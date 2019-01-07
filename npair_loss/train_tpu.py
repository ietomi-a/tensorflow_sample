# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

from npair_common import random_seed_set, read_function2, create_network2

from tensorflow.contrib.tpu.python.tpu import async_checkpoint


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
        train_op = tf.train.AdamOptimizer( learning_rate ).minimize(
            loss, global_step=tf.train.get_global_step() )

        # tf.train.Saver() の生成は graph 生成後でないとエラーになるのでここで設定している.        
        training_hooks = create_hooks( loss, params )
        
    spec = tf.estimator.EstimatorSpec( mode=mode,
                                       training_hooks=training_hooks,
                                       predictions=predictions,
                                       loss=loss,
                                       train_op=train_op )
    return spec


def train_input_fn2():
    BATCH_SIZE = 20  # バッチサイズ
    NUM_THREADS = 4  # スレッド
    INPUT_TFRECORD_TRAIN = "npair_train.tfrecord"  # TFRecordファイル名（学習用）
    
    filenames = tf.constant( [ INPUT_TFRECORD_TRAIN ] )

    dataset = tf.data.TFRecordDataset(filenames)  # ファイル名を遅延評価するパイプを作成.
    dataset = dataset.map( read_function2, NUM_THREADS) # ファイル名からデータを作成する遅延評価するパイプを作成.
    dataset = dataset.shuffle(60000)
    dataset = dataset.batch(BATCH_SIZE)  # 返すレコードの個数を指定.
    dataset = dataset.repeat(-1)  # 無限に繰り返す設定にする.

    return dataset

def get_tpu_run_config(params):
    tpu_params = {
        "name": "ietomi-demo-tpu",
        "zone": "us-central1-b",
        "gcp_project": "image-search-224008",
    }
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu_params["name"],
        zone=tpu_params['zone'],
        project=tpu_params['gcp_project'])

    # この作成時に account のアクセス状況をチェックしている.
    #run_config = None
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params['model_dir'],
        save_checkpoints_steps=params["save_steps"],
        tpu_config=tf.contrib.tpu.TPUConfig(
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2),
         )
    )
    return run_cofig

def main():
    BATCH_SIZE = 20  # バッチサイズ
    random_seed_set(1) # seed=1 はたまたまうまくいったので使っている.    
    model_dir = "model_dir_for_npair_loss"
    print( "my_estimator start")
    params = {
        "model_dir": model_dir,  # モデルデータの保存場所.
        "save_steps": 100,  # 何ステップ毎にセーブするか.
        'log_step_count_steps': 100,
    }

    run_config = get_tpu_run_config(params)
        
    # my_estimator = tf.estimator.Estimator( model_fn=model_fn_for_npair_loss,
    #                                        model_dir=model_dir,
    #                                        params=params )

    tpu_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn_for_npair_loss,
        config=run_config,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        export_to_tpu=False )
    
    tpu_estimator.train( input_fn=train_input_fn2, max_steps=1000 )
    
    # my_estimator.train( input_fn=train_input_fn,
    #                     steps=1000 )
    print( "train ok")
    return

if __name__ == "__main__":
    main()    
