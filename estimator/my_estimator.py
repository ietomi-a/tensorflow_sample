# coding: utf-8

import math
# import random

import numpy as np
import tensorflow as tf
import input_data

def my_input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

def create_mnist_net( images ):
    NUM_CLASSES = 10
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    hidden1_units = 128
    hidden2_units = 32
    weights1 = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights1')
    biases1 = tf.Variable(tf.zeros([hidden1_units]),
                          name='biases1')
    hidden1 = tf.nn.relu(tf.matmul(images, weights1) + biases1)

    weights2 = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights2')
    biases2 = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

    weights3 = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights3')
    biases3 = tf.Variable(tf.zeros([NUM_CLASSES]),
                          name='biases3')
    y = tf.matmul(hidden2, weights3) + biases3
    return y

# tf.estimator.EstimatorSpec を返す関数として定義する必要がある。
# ここで渡ってくる params というのは Estimator 作成時の引数の params.
def model_fn_for_mnist( features, labels, mode, params ):
    learning_rate = 0.01
    images = features['x']
    y = create_mnist_net( images )
    loss, train_op = None, None
    if mode != tf.estimator.ModeKeys.PREDICT:
        y_ = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_, logits=y, name='xentropy')
        loss = tf.reduce_mean( cross_entropy )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # train_op = optimizer.minimize(
        #     loss, global_step=tf.train.get_global_step() )
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            loss, global_step=tf.train.get_global_step() )

    spec = tf.estimator.EstimatorSpec( mode=mode,
                                       predictions=y,
                                       loss=loss,
                                       train_op=train_op )
    return spec

def get_mnist_data():
    # MNIST データの用意.
    mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
    train_x, train_y = my_input(mnist.train)[0], my_input(mnist.train)[1]
    # print(train_y[0])
    test_x, test_y = my_input(mnist.test)[0], my_input(mnist.test)[1]
    return train_x, train_y, test_x, test_y

def train(my_estimator, train_input_fn, test_input_fn):
    print( "train start")
    for epoch in range(10):
        my_estimator.train( input_fn=train_input_fn, steps=100)
        print( epoch )
        ev = my_estimator.evaluate(input_fn=test_input_fn)
        for key in sorted(ev):
            print("%s: %s" % (key, ev[key]))

    print( "train ok")
    return

def predict_eval( my_estimator, predict_input_fn, test_y ):
    predictions = my_estimator.predict(input_fn=predict_input_fn)
    ok_num = 0
    for i, num_array in enumerate(predictions):
        num_list = list(num_array)
        # print( i, test_y[i], num_list.index(max(num_list)) )
        if test_y[i] == num_list.index(max(num_list)):
            ok_num = ok_num + 1
        # print( num_list )
        #if i == 100:
        #    break
    print( "ok_num, i =", ok_num, i )
    print( "accuracy =", ok_num/(i+1) )
    return

def main():
    print( "my_estimator start")
    my_estimator = tf.estimator.Estimator( model_fn=model_fn_for_mnist,
                                           params={} )

    train_x, train_y, test_x, test_y = get_mnist_data()
    
    batch_size = 100
    
    # 訓練データを返す関数
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":  np.array(train_x, dtype='float32')} ,
        y=np.array(train_y, dtype='int32'),
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )

    # テストデータを返す関数
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_x, dtype='float32')},
        y=np.array(test_y, dtype='int32'),
        batch_size=batch_size,    
        num_epochs=1,
        shuffle=False
    )

    # 訓練.
    train(my_estimator, train_input_fn, test_input_fn)

    # 評価用のデータを返す関数. テストデータで代用.(本当はテストデータとも変えた方が良い.)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_x, dtype='float32')},
        batch_size=batch_size,    
        num_epochs=1,
        shuffle=False
    )

    # 訓練したモデルの評価
    predict_eval( my_estimator, predict_input_fn, test_y)
    return
    
if __name__ == "__main__":
    main()    
