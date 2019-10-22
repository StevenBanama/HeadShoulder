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
    for x in range(0, height - size, stride):
        for y in range(0, width - size, stride):
            input.append(image[x:x+size, y:y+size,:])
    return np.array(input)

def video(params=None):
    cap = cv2.VideoCapture(0)
    #cap.open("rtmp://rtmp01open.ys7.com/openlive/bd02a353615b4a12b1404f605218cb73.hd")

    K.set_learning_phase(0)
    det = Detector()

    while True:
        ret, img = cap.read()
        if not ret or len(img.shape) != 3:
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

def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='mtcnn shoulder detector')
    parser.add_argument(
        '-p', '--ptn', default="video", type=str,
        choices=['video', "one", 'save_pb'],
        help='video|one|save_pb')
    return parser.parse_args()

def save_pb():
    det = Detector()
    det.save_as_pb()

if __name__ == "__main__":
    params = init_parse()
    if params.ptn == "video":
        video()
    elif params.ptn == "one":
        run_detect()
    else:
        save_pb() 
