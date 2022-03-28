import numpy as np

from utils import iou, recover_gt_bbox

def update_precision_recall(pred_confidence_, pred_box_, ann_confidence_, ann_box_, boxs_default,precision_, recall_, thres):
    _, _, class_num = pred_confidence_.shape
    class_num = class_num - 1
    
    for i in range(len(pred_confidence_)):
        for j in range(class_num):
            pred_sort_ids = np.where(pred_confidence_[i, :, j] > 0.5)[0]
            gt_sort_ids = np.where(ann_confidence_[i, :, j] == 1)[0]

            pred_boxes = pred_box_[i, pred_confidence_[i, :, j] > 0.5]
            gt_boxes = ann_box_[i, ann_confidence_[i, :, j] == 1]

            if len(pred_boxes) == 0:
                tp = 0
                fp = 0
                fn = 0
                
                return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
            
            if len(gt_boxes) == 0:
                tp = 0
                fp = 0
                fn = 0
            
                return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}

            gt_idx_thr = []
            pred_idx_thr = []
            ious = []
            
            for ipb in pred_sort_ids:
                for igb in gt_sort_ids:
                    gx, gy, gw, gh = recover_gt_bbox(ann_box_[i], boxs_default, igb)
                    px, py, pw, ph = recover_gt_bbox(pred_box_[i], boxs_default, ipb)
                    iou_current = iou(np.array([px, py, pw, ph, px - (pw / 2), py - (ph / 2), px + (pw / 2), py + (ph / 2)]), 
                                      gx - (gw / 2), gy - (gh / 2), gx + (gw / 2), gy + (gh / 2))
                    
                    if iou_current > thres:
                        gt_idx_thr.append(igb)
                        pred_idx_thr.append(ipb)
                        ious.append(iou_current)
            
            iou_sort = np.argsort(ious)[::1]
            
            if len(iou_sort)==0:
                tp=0
                fp=0
                fn=0
                return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
            else:
                gt_match_idx=[]
                pred_match_idx=[]
                for idx in iou_sort:
                    gt_idx=gt_idx_thr[idx]
                    pr_idx= pred_idx_thr[idx]
                    if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                        gt_match_idx.append(gt_idx)
                        pred_match_idx.append(pr_idx)
                tp= len(gt_match_idx)
                fp= len(pred_boxes) - len(pred_match_idx)
                fn = len(gt_boxes) - len(gt_match_idx)
            
            return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive=0
    false_positive=0
    false_negative=0
    for img_id, res in image_results.items():
        true_positive +=res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision=0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall=0.0
    return (precision, recall)
