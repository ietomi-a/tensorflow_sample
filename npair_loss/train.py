# coding: utf-8

import time
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

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



def create_network2( x, TARGET_SIZE, space_name="embedding", reuse=False ):
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
        

# IMAGE_WIDTH = 28  # 画像サイズ：幅
# IMAGE_HEIGHT = 28  # 画像サイズ：高さ
# IMAGE_CHANNEL = 1  # 画像チャネル数

def randam_seed_set( seed=0 ):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return

def train():
    randam_seed_set(1) # seed=1 はたまたまうまくいったので使っている.
    TARGET_SIZE = 3  # 教師画像の種類数, 0 - 9 までの 10個.

    BATCH_SIZE = 20  # バッチサイズ
    MAX_STEPS = 2000  # 学習回数
    NUM_THREADS = 4  # スレッド

    INPUT_TFRECORD_TRAIN = "npair_train.tfrecord"  # TFRecordファイル名（学習用）
    INPUT_TFRECORD_TEST = "npair_test.tfrecord"  # TFRecordファイル名（評価用）
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)  # ファイル名を遅延評価するパイプを作成.
    dataset = dataset.map( read_function, NUM_THREADS) # ファイル名からデータを作成する遅延評価するパイプを作成.
    dataset = dataset.shuffle(60000)
    dataset = dataset.batch(BATCH_SIZE)  # 返すレコードの個数を指定.
    dataset = dataset.repeat(-1)  # 無限に繰り返す設定にする.

    # 型だけを設定した Iterator を用意する.
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    iter_init_op = iterator.make_initializer(dataset)  # イテレータを初期化するオペレータ.

    anc_xs, pos_xs, ys_ = iterator.get_next()  # 遅延評価されるして返される要素.

    ys1 = create_network2( anc_xs, TARGET_SIZE )
    ys2 = create_network2( pos_xs, TARGET_SIZE, reuse=True )
    loss = metric_learning.npairs_loss( ys_, ys1, ys2 )
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    start_time = time.time()

    with tf.Session() as sess:
        sess.run(iter_init_op, feed_dict={filenames: [INPUT_TFRECORD_TRAIN]})
        init = tf.global_variables_initializer()
        sess.run(init)  # 変数の初期化処理
        #x_v = sess.run(x1)
        # print(x_v.shape)
        # ys_v = sess.run(ys_)
        # print(ys_v.shape)
        # y1_v = sess.run(y1)
        # print(y1_v.shape)
        # print(y1_v)
        for step in range(MAX_STEPS):
            #print(sess.run(ys_))
            _,  loss_v = sess.run([train_op, loss])  # 最急勾配法でパラメータ更新
            duration = time.time() - start_time
            if (step+1) % 100 == 0:
                print("step={:4d}, loss={:5.2f}, time=({:.3f} sec)".format(step + 1, loss_v, duration))
    return


train()
    
#space_name = "hoge"
#holder1 = tf.placeholder( tf.float32, shape=(None,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL))
# print(holder1.shape)
# y1 = create_network2( holder1, TARGET_SIZE )
# holder2 = tf.placeholder( tf.float32, shape=(None,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL))
# y2 = create_network2( holder2, TARGET_SIZE, reuse=True )

# for v in tf.global_variables():
#     print(v.name)

# with tf.Session() as sess:
#     tf.global_

# mean=0.0, stddev=1.0
