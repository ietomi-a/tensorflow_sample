# coding: utf-8

import time

import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

from npair_common import random_seed_set, read_function, create_network2


def train():
    random_seed_set(1) # seed=1 はたまたまうまくいったので使っている.
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
        for step in range(MAX_STEPS):
            #print(sess.run(ys_))
            _,  loss_v = sess.run([train_op, loss])  # 最急勾配法でパラメータ更新
            duration = time.time() - start_time
            if (step+1) % 100 == 0:
                print("step={:4d}, loss={:5.2f}, time=({:.3f} sec)".format(step + 1, loss_v, duration))
    return


if __name__ == "__main__":
    train()
    
