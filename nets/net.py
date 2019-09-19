#coding=utf-8
import os
import keras
import keras.backend as K
import tensorflow as tf
from keras.activations import softmax, sigmoid
#from keras.engine.input_layer import Input
from keras.layers import Input
from keras.layers import Lambda, BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, Concatenate, PReLU, Reshape
from keras.layers.core import Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
from util import cls_ohem, bbox_ohem, landmark_ohem, load_data, generate_data_generator, accuracy
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback


def Squeeze(input):
    return tf.squeeze(input, axis=[1, 2])

def white_norm(input):
    return (input - tf.constant(127.5)) / 128.0

def propose_net(input):
    wn = Lambda(white_norm, name="white_norm")(input)
    conv1 = PReLU(shared_axes=[1, 2])(Conv2D(10, (3, 3), padding="valid", strides=(1, 1), name="conv1")(wn))
    block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    block2 = PReLU(shared_axes=[1, 2])(Conv2D(16, (3, 3), padding="valid", strides=(1, 1), name="conv2")(block1))
    block3 = PReLU(shared_axes=[1, 2])(Conv2D(32, (3, 3), padding="valid", strides=(1, 1), name="conv3")(block2))
    cate = Conv2D(2, (1, 1), activation="softmax", name="class")(block3)
    boxes = Conv2D(4, (1, 1), name="box")(block3)
    landmark = Conv2D(14, (1, 1), name="landmark")(block3)
    model = Model(input=input, output=[cate, boxes, landmark])
    return model

def recall_net(input):
    wn = Lambda(white_norm, name="white_norm")(input)
    conv1 = PReLU(shared_axes=[1, 2])(Conv2D(28, (3, 3), padding="valid", strides=(1, 1), name="conv1")(wn))
    block1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1) 
    conv2 = PReLU(shared_axes=[1, 2])(Conv2D(48, (3, 3), padding="valid", strides=(1, 1), name="conv2")(block1))
    block2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(conv2) 

    block3 = PReLU(shared_axes=[1, 2])(Conv2D(64, (2, 2), padding="valid", strides=(1, 1), name="conv3")(block2))
    block3 = Reshape((-1,))(block3)
    dense = PReLU()(Dense(128, name="r_feat")(block3))
    cate = Dense(2, activation=softmax, name="class")(dense)
    boxes = Dense(4, activity_regularizer=regularizers.l2(0.0005), name="box")(dense)
    landmark = Dense(14, activity_regularizer=regularizers.l2(0.0005), name="landmark")(dense)
    model = Model(input=input, output=[cate, boxes, landmark])
    return model

def output_net(input):
    wn = Lambda(white_norm, name="white_norm")(input)
    conv1 = PReLU(shared_axes=[1, 2])(Conv2D(32, (3, 3), padding="valid", strides=(1, 1), name="conv1")(wn))
    block1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1) 

    conv2 = PReLU(shared_axes=[1, 2])(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), name="conv2")(block1))
    block2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(conv2) 

    conv3 = PReLU(shared_axes=[1, 2])(Conv2D(64, (3, 3), padding="valid", strides=(1, 1), name="conv3")(block2))
    block3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)

    block4 = PReLU(shared_axes=[1, 2])(Conv2D(64, (2, 2), padding="valid", strides=(1, 1), name="conv4")(block3))
    block4 = Reshape((-1,))(block4)

    dense = PReLU(shared_axes=[1, 2], name="o_feat")(Dense(128)(block4))

    cate = Dense(2, activation=softmax, name="class")(dense)
    boxes = Dense(4, activity_regularizer=regularizers.l2(0.0005), name="box")(dense)
    landmark = Dense(14, activity_regularizer=regularizers.l2(0.0005), name="landmark")(dense)
    model = Model(input=input, output=[cate, boxes, landmark])
    return model


def BuildModel(ntype, lr=0.002, pretrain_path=None, is_train=False):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default(), session.as_default():
        if ntype == "pnet":
           input = Input(shape=[12, 12, 3] if is_train else [None, None, 3])
           model = propose_net(input)
           alpha_det, alpha_box, alpha_landmk = 1, 0.5, 0.5
    
        elif ntype == "rnet":
           model = recall_net(Input(shape=(24, 24, 3)))
           alpha_det, alpha_box, alpha_landmk = 1, 0.5, 0.5
        elif ntype == "onet":
           model = output_net(Input(shape=(48, 48, 3)))
           alpha_det, alpha_box, alpha_landmk = 1, 0.5, 1
    
        else:
           raise Exception("invalid net")
    
        model.summary()
        model.compile(optimizer=Adam(lr=lr),
            loss=[cls_ohem, bbox_ohem, landmark_ohem],
            loss_weights=[alpha_det, alpha_box, alpha_landmk],
            metrics={"class": accuracy}
        )
        if pretrain_path and os.path.exists(pretrain_path):
            print("!!!! restore path: %s"%pretrain_path)
            model.load_weights(pretrain_path, by_name=True)
    return model, graph, session

def get_size(ntype):
    if ntype == "pnet":
        return 12
    elif ntype == "rnet":
        return 24
    elif ntype == "onet":
        return 48
    raise Exception("fatal size")

def config_gpu():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='C3AE retry')
    parser.add_argument(
        '-m', '--model-path', default="./models/", type=str,
        help='the best model to load')

    parser.add_argument(
        '-log', "--log", type=str, default="./log/",
        help='log dir')

    parser.add_argument(
        '-n', '--net', default="pnet", type=str,
        choices=['pnet', "rnet", 'onet'],
        help='pnet|rnet|onet')

    parser.add_argument(
        '-lr', '--lr', default=0.002, type=float,
        help='learning rate')

    parser.add_argument(
        '-w', '--worker', default=2, type=float,
        help='mutil process worker')


    params = parser.parse_args()
    return params

def main(params):
   batch_size = 256
   sample_rate = 0.8
   seed = 2019
   save_path = os.path.join(params.model_path, "%s.h5"%params.net)
   pretrain_path = save_path
   models, graph, session = BuildModel(params.net, lr=params.lr, pretrain_path=pretrain_path, is_train=True)
   log_dir = params.log
   input_size = get_size(params.net)
   
   def get_weights(epoch, loggs):
       print(epoch, K.get_value(models.optimizer.lr))

   callbacks = [
        ModelCheckpoint(save_path, monitor='val_class_accuracy', verbose=1, save_best_only=True, mode='max'),
        TensorBoard(log_dir=log_dir, batch_size=batch_size, write_images=True, update_freq='epoch'),
        LambdaCallback(on_epoch_end=get_weights)
   ]

   dataframe = load_data("./data/", ptn=params.net)
   trainset, testset = train_test_split(dataframe, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)

   train_gen = generate_data_generator(trainset, input_size=input_size, batch_size=batch_size)
   validation_gen = generate_data_generator(testset, input_size=input_size, batch_size=batch_size)
   with session.as_default(), graph.as_default():
       history = models.fit_generator(train_gen,
           workers=params.worker, use_multiprocessing=True,
           steps_per_epoch=len(trainset) / batch_size, epochs=30,
           callbacks=callbacks, validation_data=validation_gen,
           validation_steps=len(testset) / batch_size * 3)

if __name__ == "__main__":
    params = init_parse()
    config_gpu()
    main(params)
