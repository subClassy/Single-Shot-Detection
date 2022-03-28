import random
from tkinter import image_names
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2

from utils import visualize_pred

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    def get_min(center, total):
        return max(round(center - (total / 2.), 2), 0)
    
    def get_max(center, total):
        return min(round(center + (total / 2.), 2), 1)
    
    boxes = np.zeros(((10*10+5*5+3*3+1*1)*4, 8))
    box_count = 0

    for i, layer in enumerate(layers):
        ssize = small_scale[i]
        lsize = large_scale[i]
        
        cell_size = round(1. / layer, 2)
        cell_center_y = round(cell_size / 2., 2)
        
        while cell_center_y <= 1:
            cell_center_x = round(cell_size / 2., 2)
            while cell_center_x <= 1:
                box_1 = [cell_center_x, cell_center_y, ssize, ssize, get_min(cell_center_x, ssize), get_min(cell_center_y, ssize), get_max(cell_center_x, ssize), get_max(cell_center_y, ssize)]
                box_2 = [cell_center_x, cell_center_y, lsize, lsize, get_min(cell_center_x, lsize), get_min(cell_center_y, lsize), get_max(cell_center_x, lsize), get_max(cell_center_y, lsize)]
                
                lsize_mod_h = lsize * np.sqrt(2)
                lsize_mod_w = lsize / np.sqrt(2)

                box_3 = [cell_center_x, cell_center_y, lsize_mod_h, lsize_mod_w, get_min(cell_center_x, lsize_mod_h), get_min(cell_center_y, lsize_mod_w), get_max(cell_center_x, lsize_mod_h), get_max(cell_center_y, lsize_mod_w)]
                box_4 = [cell_center_x, cell_center_y, lsize_mod_w, lsize_mod_h, get_min(cell_center_x, lsize_mod_w), get_min(cell_center_y, lsize_mod_h), get_max(cell_center_x, lsize_mod_w), get_max(cell_center_y, lsize_mod_h)]

                boxes[box_count] = box_1
                boxes[box_count + 1] = box_2
                boxes[box_count + 2] = box_3
                boxes[box_count + 3] = box_4
                box_count += 4

                cell_center_x = round(cell_center_x + cell_size, 2)
                
            cell_center_y = round(cell_center_y + cell_size, 2)
    
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)

def iou_custom(gt_info, x_min, y_min, x_max, y_max):
    inter = np.maximum(np.minimum(gt_info[:, 2] + gt_info[:, 0], x_max) - np.maximum(gt_info[:, 0], x_min), 0) \
            * np.maximum(np.minimum(gt_info[:, 3] + gt_info[:, 1], y_max) - np.maximum(gt_info[:, 1], y_min), 0)
    area_a = gt_info[:, 2] * gt_info[:, 3]
    area_b = (x_max - x_min) * (y_max - y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    
    gx = x_min + (x_max - x_min)/2
    gy = y_min + (y_max - y_min)/2
    gw = x_max - x_min
    gh = y_max - y_min

    one_hot_vector = np.zeros((4,))
    one_hot_vector[cat_id] = 1
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    if len(boxs_default[ious_true]) == 0:
        max_iou = np.argmax(ious)
        ious_true = ious == ious[max_iou]
    
    ann_confidence[ious_true] = one_hot_vector
    bboxes = boxs_default[ious_true]
    ann_box[ious_true] = np.array([(gx - bboxes[:, 0])/bboxes[:, 2], 
                                    (gy - bboxes[:, 1])/bboxes[:, 3], 
                                    np.log(gw/bboxes[:, 2]), 
                                    np.log(gh/bboxes[:, 3])]).T
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)

    return ann_box, ann_confidence

def crop_image(image, annotation_info):
    height, width, _ = image.shape
    gt_info = []
    
    for vals in annotation_info:
        class_id = int(vals[0])
        x_min = float(vals[1])
        y_min = float(vals[2])
        w_bb = float(vals[3])
        h_bb = float(vals[4])

        bbox = [class_id, x_min, y_min, w_bb, h_bb]
        gt_info.append(bbox)
    
    gt_info = np.array(gt_info)

    while True:
        mode = random.choice((
            None,
            0.3,
            0.7,
            0.9
        ))

        if mode is None:
            return image, gt_info

        min_iou = mode

        for _ in range(50):
            w = random.randrange(int(0.3 * width), width)
            h = random.randrange(int(0.3 * height), height)

            if h / w < 0.5 or 2 < h / w:
                continue

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            new_image = np.array((l, t, l + w, t + h))

            new_iou = iou_custom(gt_info[:, 1:], l, t, l + w, t + h)
            
            if not min_iou <= new_iou.min():
                continue

            image = image[new_image[1] : new_image[3], new_image[0] : new_image[2]]
            new_gt_info = np.zeros_like(gt_info)

            new_gt_info[:, 0] = gt_info[:, 0]
            new_gt_info[:, 1] = np.maximum(gt_info[:, 1], new_image[0])
            new_gt_info[:, 2] = np.maximum(gt_info[:, 2], new_image[1])
            new_gt_info[:, 3] = np.minimum(gt_info[:, 1] + gt_info[:, 3], new_image[2]) - new_gt_info[:, 1]
            new_gt_info[:, 4] = np.minimum(gt_info[:, 2] + gt_info[:, 4], new_image[3]) - new_gt_info[:, 2]

            new_gt_info[:, 1] -= new_image[0]
            new_gt_info[:, 2] -= new_image[1]

            return image, new_gt_info


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        train_test_split = int(0.9 * len(self.img_names))
        if self.train:
            self.img_names = self.img_names[:train_test_split]
        else:
            self.img_names = self.img_names[train_test_split:]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        image = cv2.imread(img_name)

        annotation_info = np.loadtxt(ann_name, dtype=np.float32)
        
        if len(annotation_info.shape) == 1:
            annotation_info = np.expand_dims(annotation_info, 0)
        
        if self.train:
            image, annotation_info = crop_image(image, annotation_info)
        
        h, w, _ = image.shape

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose(2, 0, 1)
        
        for vals in annotation_info:
            class_id = int(vals[0])
            x_min = float(vals[1]) / w
            y_min = float(vals[2]) / h
            w_bb = float(vals[3]) / w
            h_bb = float(vals[4]) / h

            #TODO:
            #1. prepare the image [3,320,320], by reading image "img_name" first.
            #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
            #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
            #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
            
            #to use function "match":
            ann_box, ann_confidence = match(ann_box, ann_confidence, self.boxs_default, self.threshold, class_id, x_min, y_min, x_min + w_bb, y_min + h_bb)
            #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
            
            #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
            #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        # visualize_pred(img_name, ann_confidence, ann_box, ann_confidence, ann_box, image, self.boxs_default, True)
        
        return image, ann_box, ann_confidence

if __name__ == '__main__':
    boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
    dataset = COCO("data/train/images/", "data/train/annotations/", 4, boxs_default, train = False, image_size=320)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, data in enumerate(dataloader, 0):
        images_, ann_box_, ann_confidence_ = data
        
        
    
    