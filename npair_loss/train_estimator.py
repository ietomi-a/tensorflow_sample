# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

from npair_common import random_seed_set, read_function, create_network2


# tf.estimator.EstimatorSpec を返す関数として定義する必要がある。
# ここで渡ってくる params というのは Estimator 作成時の引数の params.
def model_fn_for_npair_loss( features, labels, mode, params ):
    TARGET_SIZE = 3  # 教師画像の種類数, 0 - 9 までの 10個.    
    learning_rate = 0.01
    loss, train_op = None, None
    predictions = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        anc_xs = features['anc_image']
        pos_xs = features['pos_image']
        ys1 = create_network2( anc_xs, TARGET_SIZE )
        ys2 = create_network2( pos_xs, TARGET_SIZE, reuse=True )
        loss = metric_learning.npairs_loss( labels, ys1, ys2 )
        train_op = tf.train.AdamOptimizer( learning_rate ).minimize(
            loss, global_step=tf.train.get_global_step() )
    spec = tf.estimator.EstimatorSpec( mode=mode,
                                       predictions=predictions,
                                       loss=loss,
                                       train_op=train_op )
    return spec


def train_input_fn():
    BATCH_SIZE = 20  # バッチサイズ
    NUM_THREADS = 4  # スレッド
    INPUT_TFRECORD_TRAIN = "npair_train.tfrecord"  # TFRecordファイル名（学習用）
    
    filenames = tf.constant( [ INPUT_TFRECORD_TRAIN ] )

    dataset = tf.data.TFRecordDataset(filenames)  # ファイル名を遅延評価するパイプを作成.
    dataset = dataset.map( read_function, NUM_THREADS) # ファイル名からデータを作成する遅延評価するパイプを作成.
    dataset = dataset.shuffle(60000)
    dataset = dataset.batch(BATCH_SIZE)  # 返すレコードの個数を指定.
    dataset = dataset.repeat(-1)  # 無限に繰り返す設定にする.

    #iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    iterator = dataset.make_one_shot_iterator()
    anc_xs, pos_xs, ys_ = iterator.get_next()  # 遅延評価されるして返される要素.
    features = {
        "anc_image": anc_xs,
        "pos_image": pos_xs
    }
    return features, ys_
    

def main():
    random_seed_set(1) # seed=1 はたまたまうまくいったので使っている.    
    model_dir = "model_dir_for_npair_loss"
    print( "my_estimator start")
    my_estimator = tf.estimator.Estimator( model_fn=model_fn_for_npair_loss,
                                           model_dir=model_dir,
                                           params={} )
    
    my_estimator.train( input_fn=train_input_fn, steps=3 )
    print( "train ok")
    return

if __name__ == "__main__":
    main()    
    

