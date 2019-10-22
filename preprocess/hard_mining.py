import argparse
import cv2
import numpy as np
import pandas as pd
from sys import version_info
from nets.detector import Detector
from nets.net import config_gpu
from preprocess.image_process import COLUMNS, IOU, Btype

def hard_mining(params):
    from multiprocessing.pool import Pool

    stage = params.net
    dfs = pd.read_feather(params.preprocess_path)
    pool = Pool(params.workers)
    print(dfs.size)
    for g, df in dfs.groupby(np.arange(len(dfs)) // 2000):
        # process(df, params.net, "./data/hard_mining/hard_%s_%s.feather"%(params.net, g))
        pool.apply_async(process, (df, params.net, "./data/hard_mining/hard_%s_%s.feather"%(params.net, g),))
    pool.close()
    pool.join()

def mining_fp_box(file_path, img, candis, gt_boxes, keypoints, input_size):
    # mining fp data
    data = {x: [] for x in COLUMNS}
    for c in candis:
        nx1, ny1, nx2, ny2 = crop_box= np.array(c[:4])
        iou_fraction = IOU(crop_box, gt_boxes)
        max_iou = iou_fraction.max()

        if max_iou > 0.65:
            continue
        nw, nh = nx2 - nx1, ny2 - ny1 
        if max_iou > 0.3:
            gt_box = x1, y1, x2, y2 = gt_boxes[iou_fraction.argmax()]
            norm_bound = np.array([(x1 - nx1) * 1.0 / nw , (y1 - ny1) * 1.0 / nh, (x2 - nx2) * 1.0 / nw, (y2 - ny2) * 1.0 / nh])

            data["btype"].append(Btype.PART)
            data["normbox"].append(norm_bound.dumps())
            data["norm_points"].append(np.array([0]*14).dumps())
        else:
            data["btype"].append(Btype.NEG)
            data["normbox"].append(np.array([0]*4).dumps())
            data["norm_points"].append(np.array([0]*14).dumps())
        data["cropped"].append(crop_box.dumps())
        data["file_name"].append(file_path)
        nx1, ny1, nx2, ny2 = list(map(int, [max(nx1, 0), max(ny1, 0), nx2, ny2]))
        data["crop_image"].append(cv2.resize(img[ny1:ny2+1, nx1:nx2+1,:], (input_size, input_size)).dumps())
    return pd.DataFrame(data)

def mining_fn_box(file_path, img, candis, gt_boxes, keypoints, input_size):
    # enhence tp
    candis = np.array(candis or [[]])[:, :4]
    data = {x: [] for x in COLUMNS}
    for idx, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = list(map(int, gt))
        iou_fraction = IOU(gt, candis)  # change order
        max_iou = iou_fraction.max()

        if max_iou > 0.65:
            continue

        w, h = x2 - x1, y2 - y1
        norm_points = ((keypoints[idx].reshape((-1, 3,)) - np.array([x1, y1, 0])) * 1.0 / np.array([w, h, 1]))[:,:2].reshape((-1))

        data["btype"].append(Btype.POSITIVE)
        data["normbox"].append(np.array([0]*4).dumps())
        data["norm_points"].append(norm_points.dumps())
        data["cropped"].append(gt.dumps())
        data["file_name"].append(file_path)
        data["crop_image"].append(cv2.resize(img[y1:y2+1, x1:x2+1,:], (input_size, input_size)).dumps())
    return pd.DataFrame(data)

def process(dataframe, stage, output_file):
    # produce data for next stage
    try:
        config_gpu()
        detector = Detector()
        print("start gen: %s!!!!"%output_file)
        result = pd.DataFrame(columns=COLUMNS)
        for idx, ano in dataframe.iterrows():
            img_path = ano.file_name
            gboxes = np.loads(ano.boundbox, encoding='bytes') if version_info.major >=3 else np.loads(ano.boundbox) # py3 encoding='bytes'
            keypoints = np.loads(ano.keypoints, encoding='bytes') if version_info.major >=3 else np.loads(ano.keypoints)
    
            img = cv2.imread(img_path)
            height, width = img.shape[:-1]
            candis = detector.predict(img, stage) or []
            input_size = {"pnet": 12, "rnet": 24, "onet": 48}.get(stage)
            fp_df = mining_fp_box(img_path, img, candis, gboxes, keypoints, input_size)
            fn_df = mining_fn_box(img_path, img, candis, gboxes, keypoints, input_size)
            result = pd.concat([result, fp_df, fn_df], ignore_index=True)
            if idx % 100 == 0:
                print("idx: %s"%idx)
        result.to_feather(output_file) 
    except Exception as ee:
        print("!!!!!%s---%s"%(process.__name__, ee))
        print(ano.boundbox, ano.keypoints)
    print("end file %s"%output_file) 


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

if __name__ == "__main__":
    params = init_parse()
    hard_mining(params) 
