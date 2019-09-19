#coding=utf-8
import cv2
import math
import numpy as np
from net import BuildModel
from util import gen_input, NMS
import keras.backend as K


class Detector(object):

    def __init__(self):
        self.net_params = [
            ("pnet", "./models/pnet.h5", 12),
            ("rnet", "./models/rnet.h5", 24),
        ]
        self.stride = 2  # search stride
        self.models = {}  # model path
        self.scale_factor = 0.709  # image search scale
        self.min_size = 12
        self.threshold = [0.65, 0.7, 0.9]
        self.input_size = (640, 480, 3)
        self.init_net()

    def init_net(self):
        #for (net, path, size) in self.net_params:
        #    print("load net: %s"%net)
        #    self.models[net] = BuildModel(net, pretrain_path=path)
        self.models["pnet"], self.pgraph, self.psess = BuildModel("pnet", pretrain_path="./models/pnet.h5")
        self.models["rnet"], self.rgraph, self.rsess = BuildModel("rnet", pretrain_path="./models/rnet.h5")

    def predict(self, cv_img):
        with self.pgraph.as_default():
            with self.psess.as_default():
                candis = self.run_pnet(self.models["pnet"], cv_img)
        with self.rgraph.as_default():
            with self.rsess.as_default():
                rnet_candis = self.run_rnet(self.models["rnet"], cv_img, candis)
                

    def run_pnet(self, models, image, size=12):
        height, width = image.shape[:2]
        pnet_candis = []
        scale = 1
        while height * scale >= self.min_size and width * scale >= self.min_size:
            h, w = int(height * scale), int(width * scale)
            scale_img = cv2.resize(image, (w, h))
            b = models.predict(np.array([scale_img]))
            probs = b[0][:, :, :, 1]
            boxes = self.selection(image, probs[0], b[1][0], scale, size)
            nms_boxes = NMS(boxes, 0.5, "union")
            print("nms_boxes!!!", len(nms_boxes))
            pnet_candis += nms_boxes
            scale *= self.scale_factor
        pnet_candis = NMS(pnet_candis, 0.7, "union")

        '''cp_img = image.copy()
        for (rx, ry, rw, rh, _) in pnet_candis:
            cv2.rectangle(cp_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        cv2.imwrite("test_pnet.jpg", cp_img)
        '''
        return pnet_candis

    def selection(self, image, probs, boxes_reg, scale, size=12):
        candis = []
        h, w = probs.shape
        for c in xrange(w):
            for r in xrange(h):
                if probs[r][c] > self.threshold[0]:
                    reg_box = reg_x, reg_y, reg_w, reg_h = boxes_reg[r][c]
                    rx = int((c * self.stride + reg_x * size) / scale)
                    ry = int((r * self.stride + reg_y * size) / scale)

                    rw = int(size * (1 + reg_w) / scale)  # 注意暂时先这么改，  等数据远改过来一定要搞改成 
                    rh = int(size * (1 + reg_h) / scale)
                    candis.append([rx, ry, rw, rh, probs[r][c]])
        return candis

    def run_rnet(self, models, image, candis):
        feedbox = [] 
        for (rx, ry, rw, rh, _) in candis:
            feedbox.append(cv2.resize(image[ry:ry+rh, rx:rx+rw,:], (24, 24)))
        rnet_candis = models.predict(np.array(feedbox))
        mask = rnet_candis[0][:, 1] > self.threshold[1]
        mask_prob = rnet_candis[0][:, 1][mask]
        candis = np.array(candis)[mask]
        reg_box = rnet_candis[1][mask]
        rnet_candis = []
        for idx, (rx, ry, rw, rh, _) in enumerate(candis):
            rx, ry, rw, rh = map(int, [rx, ry, rw, rh])
            #cv2.rectangle(cv_img, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2)
            rx = int(rx * (1 + reg_box[idx][0]))
            ry = int(ry * (1 + reg_box[idx][1]))
            rw = int(rw * (1 + reg_box[idx][2]))
            rh = int(rh * (1 + reg_box[idx][3]))
            rnet_candis.append([rx, ry, rw, rh, mask_prob[idx]])
        rnet_candis = NMS(rnet_candis, 0.4, "min")
        print(rnet_candis)
        for (rx, ry, rw, rh, _) in rnet_candis:
            cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        cv2.imwrite("test_pri.jpg", image)
        return None
