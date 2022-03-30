import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    pred_confidence = pred_confidence.view(pred_confidence.shape[0] * 540, -1)
    ann_confidence = ann_confidence.view(ann_confidence.shape[0] * 540, -1)

    pred_box = pred_box.view(pred_box.shape[0] * 540, -1)
    ann_box = ann_box.view(ann_box.shape[0] * 540, -1)

    object_boxes = ann_confidence[:, 3] != 1

    # l_cls = F.cross_entropy(pred_confidence[object_boxes], torch.max(ann_confidence[object_boxes], 1)[1], reduction='mean') \
    #     + 3 * F.cross_entropy(pred_confidence[~object_boxes], torch.max(ann_confidence[~object_boxes], 1)[1], reduction='mean')
    
    l_cls = F.binary_cross_entropy(pred_confidence[object_boxes], ann_confidence[object_boxes], reduction='mean') \
        + 3 * F.binary_cross_entropy(pred_confidence[~object_boxes], ann_confidence[~object_boxes], reduction='mean')
    
    l_box = F.smooth_l1_loss(pred_box[object_boxes], ann_box[object_boxes])
    
    return l_box + l_cls

class down_sample_block(nn.Module):
    
    def __init__(self, in_channels, out_channels, add_additional_layer=True):
        super(down_sample_block, self).__init__()
        
        blk = []
        blk.append(nn.Conv2d(in_channels, out_channels, 3, 2, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        
        if add_additional_layer:
            for _ in range(2):
                blk.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
                blk.append(nn.BatchNorm2d(out_channels))
                blk.append(nn.ReLU())
        
        self.block = nn.Sequential(*blk)
    
    def forward(self, x):
        return self.block(x)


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, k_size, stride, padding):
        super(conv_block, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    
    def forward(self, x):
        return self.block(x)


class conv_reshape_block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, padding):
        super(conv_reshape_block, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding))
    
    def forward(self, x):
        output = self.block(x)
        output = output.view(output.shape[0], output.shape[1], -1)
        return output

class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.base_net = nn.Sequential(
            down_sample_block(3, 64),
            down_sample_block(64, 128),
            down_sample_block(128, 256),
            down_sample_block(256, 512),
            down_sample_block(512, 256, False)   
        )

        self.layer_1_1 = nn.Sequential(
            conv_block(256, 256, 1, 1, 0),
            conv_block(256, 256, 3, 2, 1),
        )

        self.layer_1_2 = conv_reshape_block(256, 16, 3, 1, 1)

        self.layer_1_3 = conv_reshape_block(256, 16, 3, 1, 1)

        self.layer_2_1 = nn.Sequential(
            conv_block(256, 256, 1, 1, 0),
            conv_block(256, 256, 3, 1, 0),
        )

        self.layer_2_2 = conv_reshape_block(256, 16, 3, 1, 1)

        self.layer_2_3 = conv_reshape_block(256, 16, 3, 1, 1)

        self.layer_3_1 = nn.Sequential(
            conv_block(256, 256, 1, 1, 0),
            conv_block(256, 256, 3, 1, 0),
        )

        self.layer_3_2 = conv_reshape_block(256, 16, 3, 1, 1)

        self.layer_3_3 = conv_reshape_block(256, 16, 3, 1, 1)

        self.layer_4_1 = conv_reshape_block(256, 16, 1, 1, 0)

        self.layer_4_2 = conv_reshape_block(256, 16, 1, 1, 0)

        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        #TODO: define forward
        output = self.base_net(x)
        output_1 = self.layer_1_1(output)
        output_red_2 = self.layer_1_2(output)
        output_blue_4 = self.layer_1_3(output)

        output_2 = self.layer_2_1(output_1)
        output_red_3 = self.layer_2_2(output_1)
        output_blue_3 = self.layer_2_3(output_1)
        
        output_3 = self.layer_2_1(output_2)
        output_red_4 = self.layer_2_2(output_2)
        output_blue_2 = self.layer_2_3(output_2)

        output_red_1 = self.layer_4_1(output_3)
        output_blue_1 = self.layer_4_2(output_3)

        bboxes = torch.cat((output_red_1, output_red_2, output_red_3, output_red_4), 2)
        bboxes = torch.permute(bboxes, [0, 2, 1])
        bboxes = torch.reshape(bboxes, (bboxes.shape[0], -1, 4))

        confidence = torch.cat((output_blue_1, output_blue_2, output_blue_3, output_blue_4), 2)
        confidence = torch.permute(confidence, [0, 2, 1])
        confidence = torch.reshape(confidence, (confidence.shape[0], -1, self.class_num))
        confidence = F.softmax(confidence, dim=2)
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        return confidence, bboxes


if __name__ == "__main__":
    network = SSD(10)
    network.cuda()
    img = torch.randn((8, 3, 320, 320))
    img = img.cuda()
    confidence, bboxes = network(img)







