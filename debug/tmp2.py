
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

x_ = tf.constant( [0.2,0.8] )
x = tf.constant([0.2,0.8])
y = nn.softmax_cross_entropy_with_logits(
    logits=x, labels=x_ )

y_hat = tf.convert_to_tensor(np.array([[0.5, 1.5, 0.1],[2.2, 1.3, 1.7]]))
y_true = tf.convert_to_tensor(np.array([[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]))
ent = nn.softmax_cross_entropy_with_logits(
    logits=y_hat, labels=y_true )

y_hat_softmax = tf.nn.softmax(y_hat)
tmp = y_true * tf.log(y_hat_softmax)
tmp2 = -tf.reduce_sum(tmp, [1])
total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), [1]))

with tf.Session() as sess:
    v =  sess.run(ent)
    print(v)
    v2 = sess.run(y_hat_softmax)
    print(v2)    
    # v3 = sess.run(total_loss)
    #print(v3)
    v4 = sess.run(tmp)
    print(v4)
    v5 = sess.run(tmp2)
    print(v5)
