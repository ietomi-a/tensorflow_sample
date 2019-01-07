# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

from npair_common import random_seed_set, create_network2, read_function2


def create_hooks( loss, params ):
    hooks = []
    save_hook = tf.train.CheckpointSaverHook(
        params["model_dir"], save_steps=params["save_steps"], saver=tf.train.Saver() )
    hooks.append( save_hook )
    logging_hook = tf.train.LoggingTensorHook(
        tensors= {
            'loss': loss,
        },
        every_n_iter=params["save_steps"] )
    hooks.append( logging_hook )
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
    

def main():
    random_seed_set(1) # seed=1 はたまたまうまくいったので使っている.    
    model_dir = "model_dir_for_npair_loss"
    print( "my_estimator start")
    params = {
        "model_dir": model_dir,
        "save_steps": 100,
    }
    my_estimator = tf.estimator.Estimator( model_fn=model_fn_for_npair_loss,
                                           model_dir=model_dir,
                                           params=params )

    my_estimator.train( input_fn=train_input_fn2,
                        steps=1000 )
    print( "train ok")
    return

if __name__ == "__main__":
    main()    
    

