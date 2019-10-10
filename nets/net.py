#coding=utf-8
import os
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import softmax, sigmoid
#from keras.engine.input_layer import Input
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Lambda, BatchNormalization, Conv2D, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, Concatenate, PReLU, Reshape, Flatten
from keras.layers import ReLU
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import regularizers
from sklearn.model_selection import train_test_split
from util import cls_ohem, bbox_ohem, landmark_ohem, load_data, generate_data_generator, accuracy, DataGenetator
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback


def Squeeze(input):
    return tf.squeeze(input, axis=[1, 2])

def white_norm(input):
    return (input - tf.constant(127.5)) / 128.0

def propose_net(input):
    wn = Lambda(white_norm, name="white_norm")(input)
    conv1 = Conv2D(10, (3, 3), padding="valid", strides=(1, 1), name="conv1", activation="relu")(wn)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    block2 = Conv2D(16, (3, 3), padding="valid", strides=(1, 1), name="conv2", activation="relu")(block1)
    block3 = Conv2D(32, (3, 3), padding="valid", strides=(1, 1), name="conv3", activation="relu")(block2)
    cate = Conv2D(2, (1, 1), activation="softmax", name="class")(block3)
    boxes = Conv2D(4, (1, 1), name="box")(block3)
    #landmark = Conv2D(14, (1, 1), name="landmark")(block3)
    model = Model(inputs=input, outputs=[cate, boxes])
    return model

def recall_net(input):
    wn = Lambda(white_norm, name="white_norm")(input)
    conv1 = PReLU(shared_axes=[1, 2])(Conv2D(28, (3, 3), padding="valid", strides=(1, 1), name="conv1")(wn))
    block1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
    conv2 = PReLU(shared_axes=[1, 2])(Conv2D(48, (3, 3), padding="valid", strides=(1, 1), name="conv2")(block1))
    block2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(conv2)

    block3 = PReLU(shared_axes=[1, 2])(Conv2D(64, (2, 2), padding="valid", strides=(1, 1), name="conv3")(block2))
    block3 = Flatten()(block3)
    dense = PReLU()(Dense(128, name="r_feat")(block3))
    cate = Dense(2, activation=softmax, name="class")(dense)
    boxes = Dense(4, activity_regularizer=regularizers.l2(0.0005), name="box")(dense)
    landmark = Dense(14, activity_regularizer=regularizers.l2(0.0005), name="landmark")(dense)
    model = Model(inputs=input, outputs=[cate, boxes, landmark])
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
    block4 = Flatten()(block4)

    dense = PReLU(name="o_feat")(Dense(128)(block4))

    cate = Dense(2, activation=softmax, name="class")(dense)
    boxes = Dense(4, activity_regularizer=regularizers.l2(0.0005), name="box")(dense)
    landmark = Dense(14, activity_regularizer=regularizers.l2(0.0005), name="landmark")(dense)
    model = Model(inputs=input, outputs=[cate, boxes, landmark])
    return model


def BuildModel(ntype, lr=0.002, pretrain_path=None, is_train=False):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default(), session.as_default():
        if ntype == "pnet":
           input = Input(shape=[12, 12, 3] if is_train else [None, None, 3])
           model = propose_net(input)
        elif ntype == "rnet":
           model = recall_net(Input(shape=(24, 24, 3)))
        elif ntype == "onet":
           model = output_net(Input(shape=(48, 48, 3)))
        else:
           raise Exception("invalid net")
    
        model.summary()
        if pretrain_path and os.path.exists(pretrain_path):
            print("!!!! restore path: %s"%pretrain_path)
            model.load_weights(pretrain_path, by_name=True)
    return model, graph, session

def gen_traning_params(ntype):
    if ntype == "pnet":
        loss_weight = alpha_det, alpha_box = [1, 0.5]
        loss_func = [cls_ohem, bbox_ohem]
        input_size = 12
    elif ntype == "rnet":
        loss_weight = alpha_det, alpha_box, alpha_landmk = [1, 0.5, 0.5]
        loss_func = [cls_ohem, bbox_ohem, landmark_ohem]
        input_size = 24
    elif ntype == "onet":
        loss_weight = alpha_det, alpha_box, alpha_landmk = [1, 0.5, 1]
        loss_func = [cls_ohem, bbox_ohem, landmark_ohem] 
        input_size = 48
    else:
        raise Exception("fatal size")
    return input_size, loss_weight, loss_func

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
        '-w', '--worker', default=1, type=int,
        help='mutil process worker')

    parser.add_argument(
        '-b', '--batch-size', default=256, type=int,
        help='batch size')

    parser.add_argument('--pruned', default=False, action='store_true', help='prune model')
    params = parser.parse_args()
    print(params)
    return params


def prun_model(params):
    import tensorflow_model_optimization as tfmot
    from tensorflow_model_optimization.sparsity import keras as sparsity

    save_path = os.path.join(params.model_path, "pruned_%s.h5"%params.net)
    pretrain_path = os.path.join(params.model_path, "%s.h5"%params.net)
    models_a = tf.python.keras.models.load_model(pretrain_path, custom_objects={"tf": tf, "white_norm": white_norm, "Squeeze": Squeeze, "cls_ohem": cls_ohem, "bbox_ohem": bbox_ohem, "landmark_ohem": landmark_ohem, "accuracy": accuracy})
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.5, final_sparsity=0.9,
                        begin_step=0, end_step=5000*4)

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(models, pruning_schedule=pruning_schedule)  # 必须是tf.keras.Model， 否则返回None, 详见源码

    input_size, loss_weight, loss_func = gen_traning_params(params.net)

    model_for_pruning.compile(optimizer=Adam(lr=params.lr),
        loss=loss_func,
        loss_weights=loss_weight,
        metrics={"prune_low_magnitude_class": accuracy}
    )
    train_gen = DataGenetator(trainset, input_size=input_size, batch_size=batch_size)
    validation_gen = DataGenetator(testset, input_size=input_size, batch_size=batch_size, is_training=False)

    callbacks = [
        ModelCheckpoint(save_path, monitor='val_prune_low_magnitude_class_accuracy', verbose=1, save_best_only=True, mode='max'),
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir="./logs", profile_batch=0)
    ]

    history = model_for_pruning.fit_generator(train_gen,
        workers=params.worker, use_multiprocessing=(params.worker > 1),
        steps_per_epoch=len(trainset) / batch_size, epochs=80,
        callbacks=callbacks, validation_data=validation_gen,
        validation_steps=len(testset) / batch_size * 3)

def train(params):
   batch_size = params.batch_size
   sample_rate = 0.8
   seed = 2019
   save_path = os.path.join(params.model_path, "%s.h5"%params.net)
   pretrain_path = save_path

   input_size, loss_weight, loss_func = gen_traning_params(params.net)
   models, graph, session = BuildModel(params.net, lr=params.lr, pretrain_path=pretrain_path, is_train=True)

   print(pretrain_path)
   with session.as_default(), graph.as_default():       
       models.compile(optimizer=Adam(lr=params.lr),
           loss=loss_func,
           loss_weights=loss_weight,
            metrics={"class": accuracy}
       )

   log_dir = params.log
   
   def get_weights(epoch, loggs):
       print(epoch, K.get_value(models.optimizer.lr))

   callbacks = [
        ModelCheckpoint(save_path, monitor='val_class_accuracy', verbose=1, save_best_only=True, mode='max'),
        TensorBoard(log_dir=log_dir, batch_size=batch_size, write_images=True),
        ReduceLROnPlateau(monitor='val_class_accuracy', factor=0.1, patience=5, min_lr=0.0000001),
        LambdaCallback(on_epoch_end=get_weights),
   ]

   dataframe = load_data("./data/", ptn=params.net)
   trainset, testset = train_test_split(dataframe, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)
   

   train_gen = DataGenetator(trainset, input_size=input_size, batch_size=batch_size)
   validation_gen = DataGenetator(testset, input_size=input_size, batch_size=batch_size, is_training=False)
   with session.as_default(), graph.as_default():
       history = models.fit_generator(train_gen,
           workers=params.worker, use_multiprocessing=(params.worker > 1),
           steps_per_epoch=len(trainset) / batch_size, epochs=80,
           callbacks=callbacks, validation_data=validation_gen,
           validation_steps=len(testset) / batch_size * 3)

if __name__ == "__main__":
    params = init_parse()
    config_gpu()
    prun_model(params) if params.pruned else train(params)
