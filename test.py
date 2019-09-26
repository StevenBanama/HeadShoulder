#coding=utf-8
import cv2
import tensorflow as tf  
from tensorflow.python.platform import gfile
      
sess = tf.Session()
img = cv2.imread("./assets/timg.jpg")
print(img.shape)

def load_model():
    sess = tf.Session()
    sess.as_default()
    with gfile.FastGFile("./model/pnet.pb", "rb") as fd:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fd.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def)
        ia = sess.graph.get_tensor_by_name("import/input_1:0")
        output = cprob, box = sess.run(['import/class/truediv:0', u'import/box/BiasAdd:0'], feed_dict={ia: [img]})
        print("output of one-hot is : ", cprob)

print(load_model())
# ('output of one-hot is : ', array([[ 1.,  0.,  0.],
#       [ 0.,  1.,  0.],
#       [ 0.,  0.,  1.]], dtype=float32))
