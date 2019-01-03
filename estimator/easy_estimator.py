# coding: utf-8
import numpy as np
import tensorflow as tf

import input_data

def my_input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data")
#print(type(mnist))
#print( type(mnist.train.images))
#exit(1)

# tf.estimator.inputs.numpy_input_fn において "x" で 指定されたとして登録された
# numpy.ndarray のオブジェクトを (28,28) の形にして渡すためのオブジェクト.
# classifier の生成時に指定する.
feature_columns = [ tf.feature_column.numeric_column("xx", shape=[28, 28]) ]

#print(type(tf.feature_column.numeric_column("x", shape=[28, 28]) ))
#exit(1)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"xx": my_input(mnist.train)[0]},
    y=my_input(mnist.train)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)

print("train_input_fn define ok")

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.3,
    model_dir="./mnist_model"
)

print("classifier define ok")

classifier.train(input_fn=train_input_fn, steps=6000)

print("train ok")

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"xx": my_input(mnist.test)[0]},
    y=my_input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

print("test_input_fn define ok")

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print( "\nTest Accuracy: {0:f}%\n".format(accuracy_score*100) )
