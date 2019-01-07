# coding: utf-8

import random

import numpy as np
import tensorflow as tf


def random_seed_set( seed=0 ):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return

def get_tensor_image( image_array ):
    IMAGE_WIDTH = 28  # 画像サイズ：幅
    IMAGE_HEIGHT = 28  # 画像サイズ：高さ
    IMAGE_CHANNEL = 1  # 画像チャネル数
    image = tf.decode_raw( image_array, tf.uint8)
    image = tf.cast( image, tf.float32 )
    image = image / 255  # 画像データを、0～1の範囲に変換する
    image = tf.reshape( image, [IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL])
    return image

def read_function(example):
    """
    以下で生成されたものが引数.
    fname_queue = tf.train.string_input_producer([tfrecord_fpath])
    reader = tf.TFRecordReader()
    _, example = reader.read(fname_queue)  
    """
    parsed_features = tf.parse_single_example(
        example,
        features={
            "label": tf.FixedLenFeature((), tf.int64 ),
            "anc_image": tf.FixedLenFeature((), tf.string ),
            "pos_image": tf.FixedLenFeature((), tf.string ),                
        }) # データ構造を解析
    label = tf.cast( parsed_features["label"], tf.int32)
    anc_image = get_tensor_image( parsed_features["anc_image"] )
    pos_image = get_tensor_image( parsed_features["pos_image"] ) 
 
    return anc_image, pos_image, label


def read_function2(example):
    parsed_features = tf.parse_single_example(
        example,
        features={
            "label": tf.FixedLenFeature((), tf.int64 ),
            "anc_image": tf.FixedLenFeature((), tf.string ),
            "pos_image": tf.FixedLenFeature((), tf.string ),                
        }) # データ構造を解析
    label = tf.cast( parsed_features["label"], tf.int32)
    features = {
        "anc_image": get_tensor_image( parsed_features["anc_image"] ),
        "pos_image": get_tensor_image( parsed_features["pos_image"] ),
    }
    return features, label


def create_network2( x, TARGET_SIZE, space_name="embedding", reuse=False ):
    # 2 になっているのはたまたま(初期のものがうまく行かなかった経緯がある.)
    CONV1_FILTER_SIZE = 3  # フィルターサイズ（幅、高さ）
    CONV1_FEATURES = 32  # 特徴マップ数
    CONV1_STRIDE = 1  # ストライドの設定
    MAX_POOL_SIZE1 = 2  # マップサイズ（幅、高さ）
    POOL1_STRIDE = 2  # ストライドの設定
    AFFINE1_OUTPUT_SIZE = 100  # 全結合層（１）の出力数
    
    with tf.variable_scope( space_name, reuse=reuse):  # 変数を共有
        # CONV層
        conv1_weight = tf.get_variable(
            "conv1_weight",
            shape=[CONV1_FILTER_SIZE, CONV1_FILTER_SIZE, 1, CONV1_FEATURES],
            initializer=tf.initializers.truncated_normal( stddev=0.1 ) )
        conv1_bias = tf.get_variable(
            "conv1_bias",
            shape=[CONV1_FEATURES],
            initializer=tf.constant_initializer( np.zeros(CONV1_FEATURES)))
        conv1_inner = tf.nn.conv2d(
            x, conv1_weight,
            strides=[1, CONV1_STRIDE, CONV1_STRIDE, 1], padding='SAME')
        conv1 = tf.nn.bias_add( conv1_inner, conv1_bias)
        conv1_activate = tf.nn.relu(conv1)        

        # POOL層
        pool1 = tf.nn.max_pool(
            conv1_activate, ksize=[1, MAX_POOL_SIZE1, MAX_POOL_SIZE1, 1],
            strides=[1, POOL1_STRIDE, POOL1_STRIDE, 1], padding='SAME' )
        pool1_shape = pool1.get_shape().as_list()
        pool1_flat_shape = pool1_shape[1] * pool1_shape[2] * pool1_shape[3]
        pool1_flat = tf.reshape( pool1, [-1, pool1_flat_shape] )  # 2次元に変換
        
        # 全結合層1
        W1 = tf.get_variable(
            "W1",
            shape=[pool1_flat_shape, AFFINE1_OUTPUT_SIZE],
            initializer=tf.initializers.truncated_normal() )
        b1 = tf.get_variable(
            "b1",
            shape=[AFFINE1_OUTPUT_SIZE],
            initializer=tf.constant_initializer( np.zeros(AFFINE1_OUTPUT_SIZE)))
        affine1 = tf.matmul(pool1_flat, W1) + b1
        activate1 = tf.sigmoid(affine1)

        # 全結合層2
        W2 = tf.get_variable(
            "W2",
            shape=[AFFINE1_OUTPUT_SIZE, TARGET_SIZE],
            initializer=tf.initializers.truncated_normal() )
        b2 = tf.get_variable(
            "b2",
            shape=[TARGET_SIZE],
            initializer=tf.constant_initializer( np.zeros(TARGET_SIZE)))
        affine2 = tf.matmul( activate1, W2) + b2

        # 出力層
        y = tf.nn.softmax(affine2)
            
    return y

