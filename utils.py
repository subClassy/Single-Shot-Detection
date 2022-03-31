import numpy as np
import cv2


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

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


def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, nms=False):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    h,w,_ = image.shape
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                gx, gy, gw, gh = recover_gt_bbox(ann_box, boxs_default, i, h, w)

                start_point_1 = (int(gx - (gw / 2)), int(gy - (gh / 2))) #top left corner, x1<x2, y1<y2
                end_point_1 = (int(gx + (gw / 2)), int(gy + (gh / 2))) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image1, start_point_1, end_point_1, color, thickness)

                start_point_2 = (int(boxs_default[i, 4] * w), int(boxs_default[i, 5] * h)) #top left corner, x1<x2, y1<y2
                end_point_2 = (int(boxs_default[i, 6] * w), int(boxs_default[i, 7] * h)) #bottom right corner
                cv2.rectangle(image2, start_point_2, end_point_2, color, thickness)
    
    if nms:
        picked_ids = non_maximum_suppression(pred_confidence, pred_box, boxs_default, class_num, 0.1, 0.8)
        
        if len(picked_ids) == 0:
            picked_ids = non_maximum_suppression(pred_confidence, pred_box, boxs_default, class_num, 0.1, 0.6)
        
        if len(picked_ids) == 0:
            picked_ids = non_maximum_suppression(pred_confidence, pred_box, boxs_default, class_num, 0.1, 0.5)
        
    else:
        picked_ids = range(len(pred_confidence))

    #pred
    for i in picked_ids:
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                gx, gy, gw, gh = recover_gt_bbox(pred_box, boxs_default, i, h, w)

                start_point_1 = (int(gx - (gw / 2)), int(gy - (gh / 2))) #top left corner, x1<x2, y1<y2
                end_point_1 = (int(gx + (gw / 2)), int(gy + (gh / 2))) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image3, start_point_1, end_point_1, color, thickness)

                start_point_2 = (int(boxs_default[i, 4] * w), int(boxs_default[i, 5] * h)) #top left corner, x1<x2, y1<y2
                end_point_2 = (int(boxs_default[i, 6] * w), int(boxs_default[i, 7] * h)) #bottom right corner
                cv2.rectangle(image4, start_point_2, end_point_2, color, thickness)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.


def recover_gt_bbox(box, boxs_default, i, h=1, w=1):
    tx, ty, tw, th = box[i]
    px, py, pw, ph = boxs_default[i, :4]
    gx = (pw * tx + px) * w
    gy = (ph * ty + py) * h
    gw = (pw * np.exp(tw)) * w
    gh = (ph * np.exp(th)) * h

    return gx, gy, gw, gh


def non_maximum_suppression(confidence_, box_, boxs_default, num_classes, overlap=0.5, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    pick = []
    #TODO: non maximum suppression
    for i in range(num_classes):
        sort_ids = np.argsort(confidence_[:, i])
        while len(sort_ids) > 0:
            id = sort_ids[-1]
            
            if confidence_[id, i] <= threshold:
                break
            else:
                pick.append(id)

            sort_ids = sort_ids[:-1]
            gx, gy, gw, gh = recover_gt_bbox(box_, boxs_default, id)

            gx_all = np.zeros(len(sort_ids))
            gy_all = np.zeros(len(sort_ids))
            gw_all = np.zeros(len(sort_ids))
            gh_all = np.zeros(len(sort_ids))

            for k, s_id in enumerate(sort_ids):
                gx_all[k], gy_all[k], gw_all[k], gh_all[k] = recover_gt_bbox(box_, boxs_default, s_id)
            
            remaining_boxes = np.array([gx_all, 
                                        gy_all, 
                                        gw_all, 
                                        gh_all, 
                                        gx_all - (gw_all / 2), 
                                        gy_all - (gh_all / 2), 
                                        gx_all + (gw_all / 2), 
                                        gy_all + (gh_all / 2)])
            
            remaining_boxes = remaining_boxes.T
            
            ious = iou(remaining_boxes, gx - (gw / 2), gy - (gh / 2), gx + (gw / 2), gy + (gh / 2))

            sort_ids = np.delete(sort_ids, np.where(ious > overlap)[0])

    return pick










