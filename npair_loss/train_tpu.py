# coding: utf-8

import sys

import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

from npair_common import random_seed_set, read_function2, create_network2

from tensorflow.contrib.tpu.python.tpu import async_checkpoint

#
# 以下のサイトは TPU 固有のハマりどころをピックアップしているのでよい.
# http://tensorflow.classcat.com/2018/04/23/tensorflow-programmers-guide-using-tpu/
# 




def create_hooks( loss, params ):
    hooks = []
    async_save_hook = async_checkpoint.AsyncCheckpointSaverHook(
        checkpoint_dir=params['model_dir'], save_steps=100 )
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
    TARGET_SIZE = 3  # 教師画像の種類数, 0 - 9 までの 10個.    
    learning_rate = 0.01
    loss, train_op = None, None
    predictions = None
    training_hooks = []

    if mode != tf.estimator.ModeKeys.PREDICT:
        anc_xs = features['anc_image']
        pos_xs = features['pos_image']
        ys1 = create_network2( anc_xs, TARGET_SIZE )
        ys2 = create_network2( pos_xs, TARGET_SIZE, reuse=True )
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


# tf.estimator.Estimator と違って、
# tf.contrib.tpu.TPUEstimator の input_fn には params を引数とする関数を渡す必要がある.
def train_input_fn3(params):
    NUM_THREADS = 4  # スレッド
    
    filenames = tf.constant( [ params["input_data_path"] ] )

    dataset = tf.data.TFRecordDataset(filenames)  # ファイル名を遅延評価するパイプを作成.
    dataset = dataset.map( read_function2, NUM_THREADS) # ファイル名からデータを作成する遅延評価するパイプを作成.
    dataset = dataset.shuffle(60000)  # MNIST のデータサイズに合わせている.

    # TPU においては dataset からとりだす tensor の shape を固定する必要がある。
    # 普通の実行では端数のデータのバッチサイズが変わってしまうので、その分は切り落とすように
    # 設定する必要がある.
    dataset = dataset.repeat().apply(
        tf.contrib.data.batch_and_drop_remainder(params['batch_size']))    
    return dataset

def get_tpu_run_config(params):
    num_shards = 8
    iterations = 50
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
        tpu_config=tf.contrib.tpu.TPUConfig( iterations, num_shards )
        # tpu_config=tf.contrib.tpu.TPUConfig(
        #     per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2),
    )

    return run_config


def get_my_params( use_tpu, is_local ):
    print( "my_estimator start, use_tpu =", use_tpu )    
    if is_local:
        # model_dir: モデルデータの保存場所.
        # input_data_path : TFRecordファイル名（学習用）
        file_data_params = {
            "model_dir": "model_dir_for_npair_loss",
            "input_data_path" : "npair_train.tfrecord" }
    else:
        # TPU では VM のローカルはうまく見れないらしいので gcs のパスを指定する必要がある.
        file_data_params = {
            "model_dir": "gs://ietomi-test/test_model_log",
            "input_data_path" : "gs://ietomi-test/npair_train.tfrecord" }

    my_params = {
        "model_dir": file_data_params["model_dir"],  # モデルデータの保存場所.
        "save_steps": 100,  # 何ステップ毎にセーブするか.
        'log_step_count_steps': 100,
        "use_tpu" : use_tpu,
        "max_steps": 1000,
        "input_data_path": file_data_params["input_data_path"] }
    return my_params

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "use_tpu":
        use_tpu = True
    else:
        use_tpu = False  # local で実行する場合はここを False にして実行する。
        
    random_seed_set(1) # seed=1 はたまたまうまくいったので使っている.

    is_local = False
    my_params = get_my_params( use_tpu, is_local )

    # バッチサイズは 8 の倍数でないとダメ.(TPUの制限),
    # また params にわたすと、 estimator の中で設定する名前とかぶるのでダメと言われるので
    # とりあえず外だしで設定する.
    batch_size = 40

    if my_params["use_tpu"]:
        # ここで TPU への接続情報を設定してる.
        run_config = get_tpu_run_config(my_params)
    else:
        # ローカル実行の場合はとりあえずのものを与えれば ok.
        run_config = tf.contrib.tpu.RunConfig()
        
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


if __name__ == "__main__":
    main()
