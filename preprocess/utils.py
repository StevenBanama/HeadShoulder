#coding=utf-8
import numpy as np

def IOU(box, gt_boxes):
    if gt_boxes.size == 0:
       return np.array([0.0])
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], gt_boxes[:, 0])
    yy1 = np.maximum(box[1], gt_boxes[:, 1])
    xx2 = np.minimum(box[2], gt_boxes[:, 2])
    yy2 = np.minimum(box[3], gt_boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter * 1.0 / (box_area + area - inter)
    return ovr
