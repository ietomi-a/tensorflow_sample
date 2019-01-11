# coding: utf-8

import sys
import os

# from tensorflow.
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.losses import metric_learning


# project module
from npair_common import random_seed_set, create_inception_network, create_network3

#
# 以下のサイトは TPU 固有のハマりどころをピックアップしているのでよい.
# http://tensorflow.classcat.com/2018/04/23/tensorflow-programmers-guide-using-tpu/
# 



def get_fpath_list_in_dir( dname ):
    return [ os.path.join( dname, fname ) for fname in os.listdir(dname) ]

def create_hooks( loss, params ):
    hooks = []
    async_save_hook = async_checkpoint.AsyncCheckpointSaverHook(
        checkpoint_dir=params['model_dir'],
        save_steps=params["save_steps"] )
    hooks.append( async_save_hook )
    # save_hook = tf.train.CheckpointSaverHook(
    #     params["model_dir"], save_steps=params["save_steps"], saver=tf.train.Saver() )
    # hooks.append( save_hook )
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors= {
    #         'loss': loss,
    #     },
    #     every_n_iter=params["save_steps"] )
    # hooks.append( logging_hook )
    
    return hooks

# tf.estimator.EstimatorSpec を返す関数として定義する必要がある。
# ここで渡ってくる params というのは Estimator 作成時の引数の params.
def model_fn_for_npair_loss( features, labels, mode, params ):
    # TARGET_SIZE = 512  #
    TARGET_SIZE = 128  #     
    learning_rate = 0.01
    loss, train_op = None, None
    predictions = None
    training_hooks = []

    if mode != tf.estimator.ModeKeys.PREDICT:
        anc_xs = features['prod_img']
        pos_xs = features['snap_img']
        ys1 = create_network3( anc_xs, TARGET_SIZE )
        ys2 = create_network3( pos_xs, TARGET_SIZE, reuse=True )
        loss = metric_learning.npairs_loss( labels, ys1, ys2 )
        optimizer = tf.train.AdamOptimizer( learning_rate )
        if params["use_tpu"]:
            # tpu の場合、optimizer を以下のようにラップしてやる必要がある.
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize( loss, global_step=tf.train.get_global_step() )

        # tf.train.Saver() の生成は graph 生成後でないとエラーになるのでここで設定している. 
        training_hooks = create_hooks( loss, params )

    spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        training_hooks=training_hooks,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )
    return spec



# for gs://snap_prod_tpu_training/tfrecords_10000 .
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
def train_input_fn4(params):
    NUM_THREADS = 4  # スレッド
    fpath_list = get_fpath_list_in_dir( params["data_dir"] ) 
    filenames = tf.constant( fpath_list )
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

def get_tpu_run_config(params):
    tpu_params_for_conenct = {
        "name": "ietomi-demo-tpu",
        "zone": "us-central1-b",
        "gcp_project": "image-search-224008",
    }

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu_params_for_conenct["name"],
        zone=tpu_params_for_conenct['zone'],
        project=tpu_params_for_conenct['gcp_project']
    )

    # この作成時に account のアクセス状況をチェックしている.
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params['model_dir'],
        save_checkpoints_steps=params["save_steps"],
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations, num_shards,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2),
    )
    
    return run_config


def get_my_params( use_tpu, is_local ):
    print( "my_estimator start, use_tpu =", use_tpu )    
    if is_local:
        # model_dir: モデルデータの保存場所.
        # input_data_path : TFRecordファイル名（学習用）
        file_data_params = {
            "model_dir": "model_dir_for_npair_loss",
            "input_data_path" : "tfrecords_10000_train-00000-of-00128" }
    else:
        # TPU では VM のローカルのファイルはうまく見れないらしいので gcs のパスを指定する必要がある.
        file_data_params = {
            "model_dir": "gs://ietomi-test/test_model_log",
            "input_data_path" : "gs://ietomi-test/tfrecords_10000_train-00000-of-00128" }

    my_params = {
        "data_dir": "data_dir1",
        "model_dir": file_data_params["model_dir"],  # モデルデータの保存場所.
        "save_steps": 3,  # 何ステップ毎にセーブするか.
        'log_step_count_steps': 1,  # 何ステップでログを取るか.
        "use_tpu" : use_tpu,  # bool, False なら TPU 環境でなくとも動くようにしている.
        "max_steps": 30, # training のステップ数.
        "num_shards": 8,  # TPU 一台で 8 個の演算ユニットがあるのでデフォルトは 8 にしておく.
        "tpu_iterations": 100,  # 一回の TPU 演算において何ステップ分を計算するか.
        # バッチサイズは num_shards=8( 関数 get_my_params を参照) の倍数でないとダメ.(TPUの制限),
        # また params にわたすと、estimator の中で設定する名前とかぶるのでダメと言われるので
        # とりあえず外だしで設定する.
        "my_batch_size": 40,
        "input_data_path": file_data_params["input_data_path"] }
    return my_params

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "use_tpu":
        use_tpu = True
    else:
        use_tpu = False  # local で実行する場合はここを False にして実行する。
        
    random_seed_set(1) # seed=1 はたまたまうまくいったので使っている. dataset のshuffle に影響.

    is_local = True
    my_params = get_my_params( use_tpu, is_local )

    if my_params["use_tpu"]:
        # ここで TPU への接続情報を設定してる.
        run_config = get_tpu_run_config(my_params)
    else:
        # ローカル実行の場合はとりあえずのものを与えれば ok.
        run_config = tf.contrib.tpu.RunConfig()

    # debug code
    #
    # my_params["batch_size"] = 7
    # dataset = train_input_fn4(my_params)
    # iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    # iter_init_op = iterator.make_initializer(dataset)  # イテレータを初期化するオペレータ.

    # features, labels = iterator.get_next()  # 遅延評価されるして返される要素.
    # anc_xs = features['prod_img']
    # pos_xs = features['snap_img']
    # TARGET_SIZE = 100
    # ys1 = create_network3( anc_xs, TARGET_SIZE )
    
    # ys2 = create_network3( pos_xs, TARGET_SIZE, reuse=True )
    # loss = metric_learning.npairs_loss( labels, ys1, ys2 )
    # optimizer = tf.train.AdamOptimizer( 0.01 )
    # train_op = optimizer.minimize( loss, global_step=tf.train.get_global_step() )
    
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)  # 変数の初期化処理        
    #     sess.run(iter_init_op)
    #     #a = sess.run(tmp)
    #     #print(a)
    #     sess.run(train_op)
    #     #a = sess.run(features)
    #     #b = sess.run(labels)
        
    tpu_estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=my_params["use_tpu"],
        model_fn=model_fn_for_npair_loss,
        config=run_config,
        train_batch_size=my_params["my_batch_size"],
        eval_batch_size=my_params["my_batch_size"],
        export_to_tpu=False,
        params=my_params
    )
    
    tpu_estimator.train( input_fn=train_input_fn4,
                         max_steps=my_params["max_steps"] )
    
    #print( "train ok")
    return



if __name__ == "__main__":
    main()
