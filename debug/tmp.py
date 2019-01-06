import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

#a = tf.constant([1, 2, 3, 4, 5, 6], shape=(2,3))
#a = tf.constant([1, 2, 3, 4, 5, 6])
a = tf.constant([1, 11, 3, 9, 5, 3, 13])
b = array_ops.shape(a)
a = array_ops.reshape(a, [b[0], 1])
b = array_ops.transpose(a)
labels_remapped = math_ops.to_float(
    math_ops.equal(a, array_ops.transpose(a)))
c = math_ops.reduce_sum(labels_remapped, 1, keepdims=True)

#b = tf.constant([1, 2, 2, 9, 5, 6], shape=(2,3))
# c = math_ops.matmul( a, b, transpose_a=False, transpose_b=True)
      
#c = math_ops.to_float(math_ops.equal( a, b ))
x =  tf.constant( [ [1.0, 2.0, 2.0, 9.0, 5.0, 6.0, 1.0],
                    [1.0, 2.0, 2.0, 9.0, 5.0, 6.0, 1.0],
                    [1.0, 2.0, 2.0, 9.0, 5.0, 6.0, 1.0],
                    [1.0, 2.0, 2.0, 9.0, 5.0, 6.0, 1.0],
                    [1.0, 2.0, 2.0, 9.0, 5.0, 6.0, 1.0],
                    [1.0, 2.0, 2.0, 9.0, 5.0, 6.0, 1.0],
                    [1.0, 2.0, 2.0, 9.0, 5.0, 6.0, 1.0] ] )

y = nn.softmax_cross_entropy_with_logits(
    logits=x, labels=labels_remapped )
with tf.Session() as sess:
    # v = sess.run(a)
    # v = sess.run(labels_remapped)
    # v =  sess.run(b)
    v =  sess.run(y)
    print(v)
    
print("tmp ok")


