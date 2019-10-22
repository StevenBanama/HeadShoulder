#coding=utf-8
import cv2
import time
import pandas as pd
import numpy as np
from .utils import IOU

COLUMNS = ["file_name", "cropped", "norm_points", "normbox", "btype", "crop_image"]

class Btype:
   NEG = 0
   PART = 1
   POSITIVE = 2

def process(dataframe, input_size=12, output_file="pnet.feather"):
    result = pd.DataFrame(columns=COLUMNS)
 
    for idx, ano in dataframe.iterrows():
        img_path = ano.file_name
        gboxes = np.loads(ano.boundbox, encoding='bytes')
        keypoint = np.loads(ano.keypoints, encoding='bytes')

        img = cv2.imread(img_path)
        height, width = img.shape[:-1]
        # 50 negtive example for each
        neg = gen_random_neg_box(img_path, img, gboxes, input_size=input_size)
        # box by gt
        npp = gen_from_ground_truth(img_path, img, gboxes, keypoint, input_size=input_size)

        result = pd.concat([neg, npp, result], ignore_index=True)
        if (idx+1) % 1000 == 0:
            print("idx: %s"%idx)
    result.to_feather(output_file)

def gen_random_neg_box(path, img, gboxes, input_size=12, neg_num=50):
    height, width = img.shape[:-1]
    data = {x: [] for x in COLUMNS}
    while neg_num > 0: 
        neg_num -= 1
        crop_size = np.random.randint(input_size, min(width, height) / 2)
        nx1 = np.random.randint(0, width - crop_size)
        ny1 = np.random.randint(0, height - crop_size)
        nx2, ny2 = nx1 + crop_size, ny1 + crop_size 
        crop_box = np.array([nx1, ny1, nx2, ny2])
        iou_fraction = IOU(crop_box, gboxes)
        if iou_fraction.max() < 0.3:
            data["file_name"].append(path)
            data["cropped"].append(crop_box.dumps())
            data["norm_points"].append(np.array([0]*14).dumps())
            data["normbox"].append(np.array([0]*4).dumps())
            data["btype"].append(Btype.NEG)
            data["crop_image"].append(cv2.resize(img[ny1:ny2+1, nx1:nx2+1,:], (input_size, input_size)).dumps())  # mat is not big enough, so do not encode img
    return pd.DataFrame(data)

def gen_from_ground_truth(path, img, gt_boxes, keypoints, input_size=12, neg_num=50):
    height, width = img.shape[:-1]
    data = {x: [] for x in COLUMNS}

    for idx, gt_box in enumerate(gt_boxes):
        x1, y1, x2, y2 = gt_box
        w, h = x2 - x1, y2 - y1
        nums = 5
        for x in range(nums):
            size = np.random.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            # max here not really necessary
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            nx2 = nx1 + size
            ny2 = ny1 + size
            if nx2 > width or ny2 > height:
                continue
            cropped = np.array([nx1, ny1, nx2, ny2])
            thres = IOU(cropped, gt_boxes)
            if thres.max() < 0.3:
                data["file_name"].append(path)
                data["cropped"].append(cropped.dumps())
                data["norm_points"].append(np.array([0]*14).dumps())
                data["normbox"].append(np.array([0]*4).dumps())
                data["btype"].append(Btype.NEG)
                data["crop_image"].append(cv2.resize(img[ny1:ny2+1, nx1:nx2+1,:], (input_size, input_size)).dumps())  # mat is not big enough, so do not encode img

            
        for i in range(20):
            # pos and part face size [minsize*0.8,maxsize*1.25]
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h))) * 1.0

            # delta here is the offset of box center
            if w < 5:
                continue
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            center_x, center_y = x1 + w / 2, y1 + h / 2
            nx1 = int(max(center_x + delta_x - size / 2, 0))
            ny1 = int(max(center_y + delta_y - size / 2, 0))
            nx2 = int(nx1 + size)
            ny2 = int(ny1 + size)

            if nx2 > width or ny2 > height:
                continue 
            crop_box = np.array([nx1, ny1, nx2, ny2])
            norm_bound = np.array([(x1 - nx1) * 1.0 / size, (y1 - ny1) * 1.0 / size, (x2 - nx2) * 1.0 / size, (y2 - ny2) * 1.0 / size])
            norm_points = ((keypoints[idx].reshape((-1, 3,)) - np.array([nx1, ny1, 0])) * 1.0 / size)[:,:2].reshape((-1))

            iou_frac = IOU(crop_box, gt_boxes)

            dtype = None
            if iou_frac.max() >= 0.65:
                dtype = Btype.POSITIVE
            elif iou_frac.max() >= 0.4:
                dtype = Btype.PART
            if dtype:
                data["file_name"].append(path)
                data["cropped"].append(crop_box.dumps())
                data["normbox"].append(norm_bound.dumps())
                data["norm_points"].append(norm_points.dumps())
                data["btype"].append(dtype)
                data["crop_image"].append(cv2.resize(img[ny1:ny2+1, nx1:nx2+1,:], (input_size, input_size)).dumps())  # mat is not big enough, so do not encode img
                #if dtype == 2:
                #    cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
        #cv2.imwrite("img_%s.jpg"%(int(time.time())), img)
        return pd.DataFrame(data)

def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='mtcnn test')
    parser.add_argument(
        '-n', '--net', default="pnet", type=str,
        choices=['pnet', "rnet", 'onet'],
        help='pnet|rnet|onet')
    parser.add_argument(
        '-p', '--preprocess-path',
        default="./data/train_data.feather", type=str,
        help='candi dataset path')
    parser.add_argument(
        '-w', '--workers',
        default=3, type=int,
        help='workers')

    params = parser.parse_args()
    return params


def main():
    from multiprocessing.pool import Pool
    params = init_parse()
    print(params)
    dataframe = pd.read_feather(params.preprocess_path)
    net = params.net 
    if net == "pnet":
       size = 12
    elif net == "rnet":
       size = 24
    elif net == "onet":
       size = 48
    pool = Pool(params.workers)
    print(dataframe.size)
    for g, df in dataframe.groupby(np.arange(len(dataframe)) // 2000):
        pool.apply_async(process, (df, size, "./data/%s_%s.feather"%(net, g),))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
