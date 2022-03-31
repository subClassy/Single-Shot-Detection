from cv2 import threshold
import numpy as np

from utils import iou, recover_gt_bbox

def update_precision_recall(picked_ids, pred_confidence_, pred_box_, ann_confidence_, ann_box_, boxs_default, thres, tp, fp, fn):
    _, _, class_num = pred_confidence_.shape
    class_num = class_num - 1
    
    true_positive = 0
    false_positive = 0
    tp_plus_fn = 0

    for i in range(len(pred_confidence_)):
        object_boxes = np.where(ann_confidence_[i, :, 3] != 1)[0]
        tp_plus_fn += len(object_boxes)
        for j in picked_ids[i]:
            for k in range(class_num):
                if pred_confidence_[i, j, k] >= 0.5:
                    gt_idxs = np.where(ann_confidence_[i, :, k] == 1)[0]
                    
                    if len(gt_idxs) == 0:
                        false_positive += 1
                        continue

                    gx, gy, gw, gh = recover_gt_bbox(pred_box_[i, :, :], boxs_default, j)

                    gx_all = np.zeros(len(gt_idxs))
                    gy_all = np.zeros(len(gt_idxs))
                    gw_all = np.zeros(len(gt_idxs))
                    gh_all = np.zeros(len(gt_idxs))

                    for k, s_id in enumerate(gt_idxs):
                        gx_all[k], gy_all[k], gw_all[k], gh_all[k] = recover_gt_bbox(ann_box_[i, :, :], boxs_default, s_id)
                    
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
                    ious = np.sort(ious)
                    if ious[-1] < thres:
                        false_positive += 1
                        continue
                    else:
                        true_positive += 1
                        continue

    return tp + true_positive, fp + false_positive, fn + tp_plus_fn - true_positive

