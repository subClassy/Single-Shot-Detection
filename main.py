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

num_epochs = 50
batch_size = 50


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
overlap_thres = 0.5

#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=10)
    
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    
    # optimizer = optim.Adam(network.parameters(),
    #                  lr=1e-2, eps=1,
    #                  weight_decay=0.0625,
    #                  betas=(0.9, 0.999))
    
    #feel free to try other optimizers and parameters.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, cooldown=2, patience=10, verbose=True)

    start_time = time.time()

    lowest_val_loss = 100
    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        precision_train_ = 0
        recall_train_ = 0
        precision_test_ = 0
        recall_test_ = 0
        
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            # ann_confidence = ann_confidence_.type(torch.LongTensor).cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            # precision_train_, recall_train_ = update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default, precision_train_, recall_train_, overlap_thres, 320)
            
            avg_loss += loss_net.data
            avg_count += 1

        
        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        # print('Precision: %f Recall: %f' % (precision_train_/avg_count, recall_train_/avg_count))
        start_time = time.time()
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred(str(epoch) + "_train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, True)
        
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        avg_count = 0
        avg_loss = 0
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            
            avg_loss += loss_net.data
            avg_count += 1

            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            # precision_test_, recall_test_  = update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default, precision_test_, recall_test_, overlap_thres, 320)
          
        
        scheduler.step(avg_loss/avg_count)
        
        #visualize
        print('[%d] validation loss: %f' % (epoch, avg_loss/avg_count))
        # print('Precision: %f Recall: %f' % (precision_test_/avg_count, recall_test_/avg_count))
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred(str(epoch) + "_val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, True)
        
        #optional: compute F1
        # F1score = 2*precision_test_*recall_test_/np.maximum(precision_test_ + recall_test_, 1e-8)
        # print(F1score)
        
        #save weights
        # if epoch%10==9:
        if  avg_loss/avg_count < lowest_val_loss:
            #save last network
            lowest_val_loss = avg_loss/avg_count
            print(f'saving net for epoch - {epoch}')
            torch.save(network.state_dict(), f'network.pth')

else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
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
