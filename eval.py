import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from metrics import update_precision_recall
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 1


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
overlap_thres = 0.5

#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

network.load_state_dict(torch.load('network.pth'))
network.eval()

if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    tp = 0
    fp = 0
    fn = 0
    
    for i, data in enumerate(dataloader, 0):
        images_, ann_box_, ann_confidence_, h, w, img_name = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence.detach().cpu().numpy()
        pred_box_ = pred_box.detach().cpu().numpy()
        
        picked_ids = []
        
        for imgg in range(len(pred_confidence_)):
            picked_ids.append(non_maximum_suppression(pred_confidence_[imgg], pred_box_[imgg], boxs_default, 3, 0.1, 0.5))
            out_list = []
            for s_id in picked_ids[-1]:
                gx, gy, gw, gh = recover_gt_bbox(pred_box_[i, :, :], boxs_default, s_id)
                sub_list = []
                sub_list.append(np.argmax(pred_confidence_[imgg, s_id, :]))
                sub_list.extend(((gx * w).cpu().numpy()[0], (gy * h).cpu().numpy()[0], (gw * w).cpu().numpy()[0], (gh * h).cpu().numpy()[0]))
                out_list.append(sub_list)
            print(out_list)

        tp, fp, fn = update_precision_recall(picked_ids, pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default, overlap_thres, tp, fp, fn)
    
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision=0.0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall=0.0

    F1score = 2 * precision * recall / np.maximum(precision+recall,1e-8)
    print(precision)
    print(recall)

    print(F1score)

else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        #pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, True)
        cv2.waitKey(1000)



