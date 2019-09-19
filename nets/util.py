#encoding=utf-8
import os
import re
import cv2
import numpy as np
import pandas as pd
from preprocess.Pnet_process import Btype
from sklearn.utils import shuffle
from keras.backend import categorical_crossentropy
import keras.backend as K
import tensorflow as tf

# Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
# total loss = \sum_{i=0}^N \sum_{j \in \{\ det, box, landmark }} \alpha_j * \beta_i^j * L_i^j
# \beta is a indicator in {0, 1}, when is background=0, ground truth =1

def cls_ohem(package_data, p_class, top=0.7):
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
    cross_entropy1 = -tf.reduce_sum(cate_one_hot * tf.log(p_class), -1)
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

    gbound = tf.gather(g_class, indices=[x for x in xrange(1, 5)], axis=-1)
    gpoints = tf.gather(g_class, indices=[x for x in xrange(5, 19)], axis=-1)
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

    gbound = tf.gather(g_class, indices=[x for x in xrange(1, 5)], axis=-1)
    gpoints = tf.gather(g_class, indices=[x for x in xrange(5, 19)], axis=-1)
    values, _ = tf.nn.top_k(tf.reduce_mean(tf.abs(gpoints - ppoint), axis=-1), nums)
    return tf.reduce_mean(values)

def accuracy(package_data, p_class):
    from keras.metrics import categorical_accuracy
    print("dddddd", p_class.get_shape())
    g_class = tf.gather(package_data, indices=[0], axis=-1)
    mask = tf.reshape(tf.equal(g_class, Btype.NEG) | tf.equal(g_class, Btype.POSITIVE), [-1])
    p_class = tf.boolean_mask(p_class, mask)

    cate = tf.cast(tf.reshape(tf.equal(g_class, Btype.POSITIVE), [-1]), tf.int32)
    cate = tf.boolean_mask(cate, mask)
    cate_one_hot = tf.reshape(tf.one_hot(cate, tf.constant(2, tf.int32)), [-1, 2])

    return categorical_accuracy(cate_one_hot, tf.reshape(p_class, [-1, 2]))

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

def image_enforcing(img, contrast=(0.5, 2.5), bright=(-50, 50), rotation=(-15, 15)):
    flag = random.randint(0, 3)
    if flag == 1:  # trans hue
        img = cv2.convertScaleAbs(img, alpha=random.uniform(*contrast), beta=random.uniform(*bright))
    elif flag == 2:  # rotation
        height, width = img.shape[:-1]
        matRotate = cv2.getRotationMatrix2D((height, width), random.randint(-15, 15), 1) # mat rotate 1 center 2 angle 3 缩放系数
        img = cv2.warpAffine(img, matRotate, (height, width))
    elif flag == 3:  # flp 翻转
        img = cv2.flip(img, 1)
    return img

def image_transform(idx, row, input_size=12, is_training=True):
    if row.crop_image:
        input_img = np.loads(row.crop_image)
    else: 
        img = cv2.imread(row.file_name)
        cropped = np.loads(row.cropped)
        x1, y1, x2, y2 = map(int, cropped.tolist())
        input_img = cv2.resize(img[y1:y2, x1:x2, :], (input_size, input_size,))
    btype, normbox, norm_points = row.btype, trans_numpy(row.normbox), trans_numpy(row.norm_points)
    btype = np.array([btype])
    #if btype == 2:
        #cv2.imwrite("%s.jpg"%idx, img[y1:y2,x1:x2, :])
    #if img[y1:y2, x1:x2, :].size == 0:
        #print("dddddd", cropped, y1, y2, x1, x2, img.shape, row.file_name, btype)
    result = np.concatenate((btype, normbox, norm_points,))  # 0: class, 1-4: boundbox, 5-19: keypoints
    return input_img, result

def trans_numpy(data):
    return np.loads(data)

def generate_data_generator(dataframe, input_size=12, batch_size=32, is_training=True):
    dataframe = dataframe.reset_index(drop=True)
    all_nums = len(dataframe)
    while True:
        idxs = np.random.permutation(all_nums)
        start = 0
        while start + batch_size < all_nums:
            candis = dataframe.loc[list(idxs[start:start+batch_size])]
            result = np.array(map(lambda x: image_transform(*x, is_training=is_training, input_size=input_size), candis.iterrows()))
            imgs, concat_data = result[:,0], result[:,1]
            imgs = np.array(imgs.tolist())
            concat_data = np.array(concat_data.tolist())
            yield imgs, [concat_data, concat_data, concat_data]
            start += batch_size

def gen_input(image, size=12, stride=12):
    input = []
    height, width = image.shape[:2]
    for y in xrange(0, height - size, stride):
        for x in xrange(0, width - size, stride):
            input.append(image[y:y+size, x:x+size:])
    return np.array(input)

def NMS(boxes, thres=0.5, ntype="union"):
    # boxes: [[x, y, w, h, prob]...]
    boxes = sorted(boxes, cmp=lambda x, y: -cmp(x[-1], y[-1]))
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
    x1, y1, w1, h1 = b1[:4]
    x2, y2, w2, h2 = b2[:4]
    inter = abs(min(x1+w1, x2+w2) - max(x1, x2)) * abs(min(y1+h1, y2+h2) - max(y1, y2))
    if ntype == "union":
        return inter * 1.0 / (w1*h1 + w2*h2 - inter + 0.0000000001) < threshold
    elif ntype == "min":
        return inter * 1.0 / min(w1*h1, w2*h2) < threshold
    return False
      

if __name__ == "__main__":
    dataframe = load_data("./data/")
    gr = generate_data_generator(dataframe)
    print(next(gr)[1][0][:,0])
