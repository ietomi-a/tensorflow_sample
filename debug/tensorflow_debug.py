#coding: utf-8

import random

import tensorflow as tf
import numpy as np

random.seed(0)

training = tf.placeholder_with_default( False, shape=(), name="training" )

x = tf.placeholder( tf.float32, shape=(None,1) )

#w = tf.Variable( tf.zeros([1,1]) )
w = tf.Variable( tf.ones([1,1]) )
#b = tf.Variable( tf.zeros([1]) )
b = tf.Variable( tf.ones([1]) )
hidden_tmp = tf.matmul( x, w ) + b # 行列の掛け算
#hidden = tf.sigmoid( hidden_tmp )
bn1 = tf.layers.batch_normalization( hidden_tmp, training=training, momentum=0.9 )
hidden = tf.sigmoid( bn1 )
#print( dir(bn1) )
#exit(1)

w2 = tf.Variable( tf.ones([1,1]) )
#b2 = tf.Variable( tf.zeros([1]) )
b2 = tf.Variable( tf.ones([1]) )
f = tf.sigmoid( tf.matmul( hidden, w2 ) + b2 ) # 行列の掛け算


f_ = tf.placeholder( tf.float32, shape=(None,1) )
loss = tf.reduce_mean( tf.abs( f_ - f) )
learn_rate = 0.5
trainer = tf.train.GradientDescentOptimizer( learn_rate )
extra_update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
with tf.control_dependencies(extra_update_ops):
    trainer_op = trainer.minimize(loss)
#trainer_op = trainer.minimize( loss )

batch_size = 3
epochs = 10
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    for i in range( epochs ):
        batch_xs, batch_fs = [], []
        #print( w.eval() )
        for j in range( batch_size ):
            x1 = random.random()
            f1 = x1*x1*x1 + 1 # この関数を訓練させる！
            batch_xs.append( [ x1 ] )
            batch_fs.append( [ f1 ] )

        print( batch_xs )
        for v in tf.global_variables():
            print(v.name, v.eval() )
        print( hidden_tmp.eval( feed_dict={x: batch_xs } ) )
        print( bn1.eval( feed_dict={x: batch_xs } ) )         
        #print( "w:", w.eval() )
        #print( "b:", b.eval() )
        #result = loss.eval( feed_dict={x: batch_xs, f_: batch_fs } )
        #print( result )
        #print( extra_update_ops[1].eval( feed_dict={x: batch_xs } ) )
        #print( extra_update_ops )
        #print( bn1.graph )
        sess.run( [trainer_op, extra_update_ops],
                  feed_dict={x: batch_xs, f_: batch_fs, training:True } )
        # print("after training")
        # for v in tf.global_variables():
        #     print(v.name, v.eval() )
        # print( hidden_tmp.eval( feed_dict={x: batch_xs } ) )
        # print( bn1.eval( feed_dict={x: batch_xs } ) )         
        # sess.run( trainer_op,
        #           feed_dict={x: batch_xs, f_: batch_fs, training:True } )

        #print("after training")
        result_loss = loss.eval( feed_dict={x: batch_xs, f_: batch_fs } )
        print( "result_loss:", result_loss )
        #print( "training:", training.eval() )
        #print( "hidden:", hidden.eval() )
        #print( "bn1:", bn1.eval() )
        #print( "w2:", w2.eval() )
        #print( "b2:", b2.eval() )        

    
    batch_xs, batch_fs = [], []
    #print( w.eval() )
    for j in range( batch_size ):
        x1 = random.random()
        f1 = x1*x1*x1 + 1 # この関数を訓練させる！
        batch_xs.append( [ x1 ] )
        batch_fs.append( [ f1 ] )
    f_list = f.eval( feed_dict={x: batch_xs} )
    print( "check")
    print( batch_xs )
    print(f_list)
    print(batch_fs )
