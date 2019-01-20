# coding: utf-8

import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

from official.resnet import imagenet_main


def create_network_resnet( inputs, embedding_dim, is_training, reuse=False ):
    resnet_size = 50
    model = imagenet_main.ImagenetModel( resnet_size,
                                         num_classes=embedding_dim )
    with tf.variable_scope("", reuse=reuse):
        y = model(inputs, training=is_training)
    return y

def random_seed_set( seed=0 ):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return

def read_function4( example, image_size=224 ):
    parsed_features = tf.parse_single_example(
        example,
        features={
            "item_id": tf.FixedLenFeature((), tf.string ),
            "prod_img": tf.FixedLenFeature((), tf.string ),
            "snap_img": tf.FixedLenFeature((), tf.string ), }) # データ構造を解析
    
    ret_features = {}
    for img in [ "prod_img", "snap_img" ]:
        tmp_image = tf.image.decode_jpeg(parsed_features[img], channels=3 )
        ret_features[img] = tf.image.resize_bicubic([tmp_image], [image_size, image_size])[0]
    
    return ret_features, hash(parsed_features["item_id"])


# tf.estimator.Estimator と違って、
# tf.contrib.tpu.TPUEstimator の input_fn には params を引数とする関数を渡す必要がある.
def train_input_fn_tmp(params):
    NUM_THREADS = 4  # スレッド
    filenames = tf.constant( [ params["input_data_path"] ] )
    dataset = tf.data.TFRecordDataset(filenames)  # ファイル名を遅延評価するパイプを作成.
    dataset = dataset.map( read_function4, NUM_THREADS) # ファイル名からデータを作成する遅延評価するパイプを作成.

    # TPU においては dataset からとりだす tensor の shape を固定する必要がある。
    # 普通の実行では端数のデータのバッチサイズが変わってしまうので、その分は切り落とすように設定する必要がある.
    # params['batch_size'] は estimator の train メソッド呼び出しの際に estimator によって設定される.
    dataset = dataset.repeat().apply(
        tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
    
    # TPU が計算してる間に次のバッチの用意をするためのおまじないらしい.
    dataset.prefetch(tf.contrib.data.AUTOTUNE)
    
    return dataset


    
def model_fn_by_resnet( features, labels, mode, params ):
    if mode != tf.estimator.ModeKeys.PREDICT:    
        learning_rate = 0.01
        anc_xs = features['prod_img']
        pos_xs = features['snap_img']
        TARGET_SIZE = 128  #     
        ys1 = create_network_resnet( anc_xs, TARGET_SIZE )
        ys2 = create_network_resnet( pos_xs, TARGET_SIZE, reuse=True )
        loss = metric_learning.npairs_loss( labels, ys1, ys2 )
        optimizer = tf.train.AdamOptimizer( learning_rate )
        train_op = optimizer.minimize( loss, global_step=tf.train.get_global_step() )
    spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )
    return spec
    



def get_my_params():
    file_data_params = {
        "model_dir": "model_dir_for_resnet",
        "input_data_path" : "samples/my_record.tfrecord",
    }
    my_params = {
        "model_dir": file_data_params["model_dir"],  # モデルデータの保存場所.
        "save_steps": 1,  # 何ステップ毎にセーブするか.
        'log_step_count_steps': 1,
        "use_tpu" : False,
        "max_steps": 5,
        "input_data_path": file_data_params["input_data_path"] }
    return my_params


def main():
    random_seed_set(1) # seed=1 はたまたまうまくいったので使っている.    
    run_config = tf.contrib.tpu.RunConfig()

    batch_size = 20
    tpu_estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=my_params["use_tpu"],
        model_fn=model_fn_for_npair_loss,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        export_to_tpu=False,
        params=my_params
    )
    tpu_estimator.train( input_fn=train_input_fn3, max_steps=my_params["max_steps"] )
    print( "train ok")
    return
    

inputs = tf.placeholder( tf.float32, [None, 224,224,3] )
inputs2 = tf.placeholder( tf.float32, [None, 224,224,3] )
embedding_dim = 128
is_training = True
y = create_network_resnet( inputs, embedding_dim, is_training, reuse=False )
y2 = create_network_resnet( inputs2, embedding_dim, is_training, reuse=True )

print("model create ok")
vars_to_warm_start = '^(?!.*dense)'

# for v in tf.global_variables():
#     print(v.name, v.shape)

    
for v in tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES,
                            vars_to_warm_start ):
    print(v.name, v.shape)
