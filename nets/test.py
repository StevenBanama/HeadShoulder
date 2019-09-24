import keras
import cv2
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.activations import softmax, sigmoid
from keras.engine.input_layer import Input
from net import BuildModel
from detector import Detector 

def gen_input(image, size=12, stride=12):
    input = []
    height, width = image.shape[:2]
    for x in xrange(0, height - size, stride):
        for y in xrange(0, width - size, stride):
            input.append(image[x:x+size, y:y+size,:])
    return np.array(input)

def video(params=None):
    cap = cv2.VideoCapture(0)
    det = Detector()

    while True:
        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.resize(img, (640, 480))
        (height, width) = img.shape[:-1]

        det.predict(img)

        if img is not None:
            cv2.imshow("result", img)
        if cv2.waitKey(3) == 27:
            break


def run_detect():
    img = cv2.imread("./assets/timg4.jpg")
    print(img)
    det = Detector()
    det.predict(img)

if __name__ == "__main__":
    #run_detect()
    video()
