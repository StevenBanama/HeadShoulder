#encoding=utf-8
import os
import re
import cv2
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

import numpy as np
import pandas as pd
import random
import keras
from tensorflow.python.keras import backend as K
import tensorflow as tf

from sys import version_info
from preprocess.image_process import Btype
from sklearn.utils import shuffle
from tensorflow.python.keras.backend import categorical_crossentropy

# Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
# total loss = \sum_{i=0}^N \sum_{j \in \{\ det, box, landmark }} \alpha_j * \beta_i^j * L_i^j
# \beta is a indicator in {0, 1}, when is background=0, ground truth =1

def cls_ohem(package_data, p_class, top=0.7, epsilon=10**-8):
    # label: true class
    # package_data: (class, box, landmarks)
    print(package_data, p_class)
    if len(p_class.get_shape()) == 4:
         package_data = tf.reshape(package_data, [-1, 19])
         p_class = tf.squeeze(p_class, axis=[1, 2])
    g_class = tf.gather(package_data, indices=[0], axis=-1)
    mask = tf.reshape(tf.equal(g_class, Btype.NEG) | tf.equal(g_class, Btype.POSITIVE), [-1])

    cate = tf.cast(tf.equal(g_class, Btype.POSITIVE), tf.int32)

    cate_one_hot = tf.squeeze(tf.one_hot(cate, 2))
    nums = tf.cast(tf.reduce_sum(tf.cast(mask, tf.float32)) * top, tf.int32)
    cross_entropy1 = -tf.reduce_sum(cate_one_hot * tf.log(p_class + epsilon), -1)
    mask_ce = tf.where(mask, cross_entropy1, tf.zeros_like(cross_entropy1))
    values, _ = tf.nn.top_k(mask_ce, nums)
    return tf.reduce_mean(values)

def bbox_ohem(package_data, p_box, top=0.7):
    print(package_data, p_box)
    if len(p_box.get_shape()) == 4:
         package_data = tf.reshape(package_data, [-1, 19])
         p_box = tf.squeeze(p_box, axis=[1, 2])

    g_class = tf.gather(package_data, indices=[0], axis=-1)
    mask = tf.reshape(tf.equal(g_class, Btype.PART) | tf.equal(g_class, Btype.POSITIVE), [-1])
    nums = tf.cast(tf.reduce_sum(tf.cast(mask, tf.float32)) * top, tf.int32)

    gbound = tf.gather(package_data, indices=[x for x in range(1, 5)], axis=-1)
    gpoints = tf.gather(package_data, indices=[x for x in range(5, 19)], axis=-1)
    values, _ = tf.nn.top_k(tf.reduce_mean(tf.abs(gbound - p_box), axis=-1), nums)
    return tf.reduce_mean(values)

def landmark_ohem(package_data, ppoint, top=0.7):
    print(package_data, ppoint)
    if len(ppoint.get_shape()) == 4:
        package_data = tf.reshape(package_data, [-1, 19])
        ppoint = tf.squeeze(ppoint, axis=[1, 2])

    g_class = tf.gather(package_data, indices=[0], axis=-1)
    mask = tf.reshape(tf.equal(g_class, Btype.POSITIVE), [-1])
    nums = tf.cast(tf.reduce_sum(tf.cast(mask, tf.float32)) * top, tf.int32)

    gbound = tf.gather(package_data, indices=[x for x in range(1, 5)], axis=-1)
    gpoints = tf.gather(package_data, indices=[x for x in range(5, 19)], axis=-1)
    gp_mask = tf.cast(tf.equal(gpoints, 0), tf.float32)  # keypoints of ground truth is 0 which means hidden
    values, _ = tf.nn.top_k(tf.reduce_mean(tf.abs(gpoints - ppoint) * gp_mask, axis=-1), nums)
    return tf.reduce_mean(values)

def accuracy(package_data, p_class):
    from keras.metrics import categorical_accuracy
    g_class = tf.gather(package_data, indices=[0], axis=-1)
    mask = tf.reshape(tf.equal(g_class, Btype.NEG) | tf.equal(g_class, Btype.POSITIVE), [-1])
    p_class = tf.boolean_mask(p_class, mask)

    cate = tf.cast(tf.reshape(tf.equal(g_class, Btype.POSITIVE), [-1]), tf.int32)
    cate = tf.boolean_mask(cate, mask)
    cate_one_hot = tf.reshape(tf.one_hot(cate, tf.constant(2, tf.int32)), [-1, 2])

    return categorical_accuracy(cate_one_hot, tf.reshape(p_class, [-1, 2]))


def rotate(img, norm_box, norm_keypoints, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    #whole image rotate
    #pay attention: 3rd param(col*row)
    img = cv2.warpAffine(img, rot_mat, (width, height))
    keypoints_ = np.asarray([
        (rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
         rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])
         for (x, y) in norm_keypoints
    ])
    return (img, norm_box, keypoints_)


def flip(img, bbox, keypoints):
    """flip"""
    flip_img = cv2.flip(img, 1)
    #mirror
    kps = np.asarray([(1-x, y) for (x, y) in keypoints])
    kps[[1, 2]] = kps[[2, 1]]  # left eye<->right eye
    kps[[3, 4]] = kps[[4, 3]]  # left ear<->right ear
    kps[[5, 6]] = kps[[6, 5]]  # left shoulder<->right shoulder
    return (flip_img, bbox, kps)

def load_data(data_dir, ptn="pnet"):
    data = []
    for dirname, a, files in os.walk(data_dir):
        for fname in files:
            if ptn and not re.search(ptn, fname):
                continue
            df = pd.read_feather(os.path.join(dirname, fname))
            data.append(df)
    dataframe = pd.concat(data, ignore_index=True)
    pos_example = dataframe[dataframe.btype == Btype.POSITIVE]
    part_example = dataframe[dataframe.btype == Btype.PART]
    neg_example = dataframe[dataframe.btype == Btype.NEG]
    pos_num = int(len(pos_example))
    cand = [(pos_example, pos_num), (part_example, pos_num), (neg_example, 3*pos_num)]
    select_example = []
    for df, num in cand:
        select_example.append(df.sample(num, replace=True, random_state=1))
    sample_data = pd.concat(select_example, ignore_index=True)
    print(sample_data.groupby("btype").agg("count"))
    sample_data = shuffle(sample_data)
    return sample_data

def image_enforcing(img, norm_box, norm_keypoints, contrast=(0.5, 2.5), bright=(-50, 50), rotation=(-15, 15)):
    flag = random.randint(0, 3)
    norm_keypoints = np.array(norm_keypoints).reshape((-1, 2))
    if flag == 1:  # trans hue
        img = cv2.convertScaleAbs(img, alpha=random.uniform(*contrast), beta=random.uniform(*bright))
    elif flag == 2:  # rotation
        alpha = random.randint(*rotation)
        img, norm_box, norm_keypoints = rotate(img, norm_box, norm_keypoints, alpha)
    elif flag == 3:  # flp 翻转
        img, norm_box, norm_keypoints = flip(img, norm_box, norm_keypoints)
    return img, norm_box, norm_keypoints.reshape((-1,)).tolist()

def image_transform(idx, row, input_size=12, is_training=True):
    if row.crop_image:
        input_img = np.loads(row.crop_image, encoding='bytes') if version_info.major >=3 else np.loads(row.crop_image)
    else:
        img = cv2.imread(row.file_name)
        cropped = np.loads(row.cropped, encoding='bytes')
        x1, y1, x2, y2 = [int(x) for x in cropped.tolist()]
        input_img = cv2.resize(img[y1:y2, x1:x2, :], (input_size, input_size,))
    btype, normbox, norm_points = row.btype, trans_numpy(row.normbox), trans_numpy(row.norm_points)
    btype = np.array([btype])
    #if btype == 2:
        #cv2.imwrite("%s.jpg"%idx, img[y1:y2,x1:x2, :])
    #if img[y1:y2, x1:x2, :].size == 0:
        #print("dddddd", cropped, y1, y2, x1, x2, img.shape, row.file_name, btype)

    if is_training:
        input_img, normbox, norm_points = image_enforcing(input_img, normbox, norm_points)
    result = np.concatenate((btype, normbox, norm_points,))  # 0: class, 1-4: boundbox, 5-19: keypoints
    return input_img, result

def trans_numpy(data):
    return np.loads(data, encoding='bytes') if version_info.major >= 3 else np.loads(data)

def generate_data_generator(dataframe, input_size=12, batch_size=32, is_training=True):
    dataframe = dataframe.reset_index(drop=True)
    all_nums = len(dataframe)
    while True:
        idxs = np.random.permutation(all_nums)
        start = 0
        while start + batch_size < all_nums:
            candis = dataframe.loc[list(idxs[start:start+batch_size])]
            result = np.array([image_transform(*x, is_training=is_training, input_size=input_size) for x in candis.iterrows()])
            imgs, concat_data = result[:,0], result[:,1]
            imgs = np.array(imgs.tolist())
            concat_data = np.array(concat_data.tolist())
            if input_size == 12:
                yield imgs, [concat_data, concat_data]
            else:
                yield imgs, [concat_data, concat_data, concat_data]
            start += batch_size

class DataGenetator(tf.keras.utils.Sequence):

    def __init__(self, dataframe, input_size=12, batch_size=32, is_training=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.input_size = input_size
        self.is_training = is_training

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        candis = self.dataframe[start:start + self.batch_size]
        result = np.array([image_transform(*x, is_training=self.is_training, input_size=self.input_size) for x in candis.iterrows()])
        imgs, concat_data = result[:,0], result[:,1]
        imgs = np.array(imgs.tolist())
        concat_data = np.array(concat_data.tolist())
        if self.input_size == 12:
            return imgs, [concat_data, concat_data]
        else:
            return imgs, [concat_data, concat_data, concat_data]

    def on_epoch_end(self):
        idxes = np.random.permutation(len(self.dataframe))
        self.dataframe = self.dataframe.take(idxes).reset_index(drop=True)


def gen_input(image, size=12, stride=12):
    input = []
    height, width = image.shape[:2]
    for y in range(0, height - size, stride):
        for x in range(0, width - size, stride):
            input.append(image[y:y+size, x:x+size:])
    return np.array(input)

def NMS(boxes, thres=0.5, ntype="union"):
    # boxes: [[x, y, x2, y2, prob]...]
    boxes = sorted(boxes, key=lambda x:x[-1], reverse=True)
    output = []
    for b in boxes:
        keep = True
        for exist in output:
            keep &= cal_nms(b, exist, thres, ntype=ntype)
            if not keep:
                break
        if keep:
            output.append(b)
    return output

def cal_nms(b1, b2, threshold, ntype="union"):
    fx1, fy1, fx2, fy2 = b1[:4]
    sx1, sy1, sx2, sy2 = b2[:4]
    inter = (min(fx2, sx2) - max(fx1, sx1)) * (min(fy2, sy2) - max(fy1, sy1))
    w1, h1, w2, h2 = (fx2 - fx1), (fy2 -fy1), (sx2 - sx1), (sy2 - sy1)
    if ntype == "union":
        return inter * 1.0 / (w1 * h1 + w2 * h2 - inter + 0.0000000001) < threshold
    elif ntype == "min":
        return inter * 1.0 / min(w1 * h1, w2 * h2) < threshold
    return True

if __name__ == "__main__":
    dataframe = load_data("./data/")
    gr = generate_data_generator(dataframe)
    print(next(gr)[1][0][:,0])
    dd = DataGenetator(dataframe, )
    print(dd[1])
