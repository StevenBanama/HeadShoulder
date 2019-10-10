#coding=utf-8
import cv2
import os
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

KEY_POINTS = [(0, 'nose'), (1, 'left_eye'), (2, 'right_eye'), (3, 'left_ear'), (4, 'right_ear'),  (5, 'left_shoulder'),  (6, 'right_shoulder')]
COLUMNS = ["id", "file_name", "keypoints", "boundbox"]
#PTYPE = [(0, "neg"), (1, "part"), (2, "positive"), (3, "landmark")]


def key_points_filter(image, an, points):
    if not isinstance(an["segmentation"], list):
        return False
    # 两个肩膀必须有一个, 耳朵高于肩膀
    if an["keypoints"][3*5+0] == 0 and an["keypoints"][3*6+0] == 0:
        return False
    if an["keypoints"][3*5+1] != 0 and an["keypoints"][3*3+1] != 0 and an["keypoints"][3*5+1] < an["keypoints"][3*3+1]:
        return False
    if an["keypoints"][3*6+1] != 0 and an["keypoints"][3*4+1] != 0 and an["keypoints"][3*6+1] < an["keypoints"][3*4+1]:
        return False
    # get only 7 keypoints
    for x in range(7): # max point 17
        if an["keypoints"][3*x+2] >= 1:
            if an["keypoints"][3*x+0] == 0:
                continue
            points.append([an["keypoints"][3*x+0], an["keypoints"][3*x + 1]])
            image = cv2.putText(image, '%s'%x, (an["keypoints"][3*x+0], an["keypoints"][3*x + 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    tmp_points = np.array(points or [[0, 0]])
    xmin, xmax = min(tmp_points[:, 0]), max(tmp_points[:, 0])
    ymin, ymax = min(tmp_points[:, 1]), max(tmp_points[:, 1])
    width, height = xmax - xmin, ymax - ymin
    if width > 0 and (height / width) > 3:
        return False

    for x in range(len(an["segmentation"])):
        segmentation = an["segmentation"][x]
        if segmentation:
            for y in range(len(segmentation)//2):
                #image = cv2.circle(image,(int(segmentation[2*y+0]), int(segmentation[2*y+1])), 1, (0,0,213), -1)
                if points and len(points) >= 3:  # 关键点必须超过三个
                    if segmentation[2*y+1] < ymax and xmax + 0.5 *(xmax-xmin) > segmentation[2*y] > xmin - 0.5 *(xmax-xmin):
                        points.append([int(segmentation[2*y+0]), int(segmentation[2*y+1])])

    if len(points) < 3:
         return False
    return True

def init_coco(img_path, key_point_anno, output):
    coco = COCO(key_point_anno)
    catIds = coco.getCatIds(catNms=['person']);
    ca = coco.loadCats(catIds)
    img_keys = coco.imgs.keys()
    result = {field: [] for field in COLUMNS}
    for cid, k in enumerate(img_keys):
        if cid % 1000 == 0:
            print("------%s / %s------"%(cid,len(img_keys) ))
        img_meta = coco.imgs[k]
        img_idx = img_meta['id'] 
        file_name = os.path.join(img_path, img_meta['file_name'])
        ann_idx = coco.getAnnIds(imgIds=img_idx)
        anns = coco.loadAnns(ann_idx)
        total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
        image = cv2.imread(file_name)
        boxes, key_points = [], []
        if len(anns) == 0:
            continue
        for an in anns:
            if len(an["keypoints"]) == 0:
                continue
            points = []

            if not key_points_filter(image, an, points):
                continue

            points = np.array(points)
            box = map(int, [min(points[:, 0]), min(points[:, 1]), max(points[:, 0]), max(points[:, 1])])  # minx, miny, maxx, maxy
            key_point = [int(p) for p in an["keypoints"][:3*7]]  # x, y, visable
            image = cv2.rectangle(image, tuple([box[0], box[1]]), tuple([box[2], box[3]]), (0,255,0), 2)
            boxes.append(box)
            key_points.append(key_point)

        result["id"].append(str(img_idx))
        result["file_name"].append(file_name)
        result["keypoints"].append(np.array(key_points).dumps())
        result["boundbox"].append(np.array(boxes).dumps())

        #cv2.imwrite(img_meta['file_name'], image)
    pd_result = pd.DataFrame(result)
    pd_result.to_feather(output)

def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='mtcnn test')
    parser.add_argument(
        '-d', '--data-dir', default="/data/build/dataset/coco/train2017/",
        type=str, help='the dirname of trainning data')
    parser.add_argument(
        '-ano', '--anotation',
        default="/data/build/dataset/coco/anotation/person_keypoints_train2017.json",
        type=str,
        help='candi dataset path')
    parser.add_argument(
        '-o', '--output',
        default="./data/test.feather", type=str,
        help='head and should points')

    params = parser.parse_args()
    return params

if __name__ == "__main__":
    params = init_parse()
    img_path = params.data_dir
    key_point_anno = params.anotation
    output = params.output
    init_coco(img_path, key_point_anno, output)
