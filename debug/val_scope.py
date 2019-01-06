import tensorflow as tf
import numpy as np

with tf.variable_scope("params"):
    w = tf.Variable(tf.constant([3.]), name='w')
    b = tf.Variable(tf.constant([1.]), name='b')

with tf.variable_scope("input"):
    x = tf.Variable(tf.constant([2.]), name='x')
    y_ = tf.Variable(tf.constant([5.]), name='y_')

with tf.variable_scope("intermediate"):
    p = w*x
    y = p+b
    s = -y
    t = s +y_
    f = t*t    


gx, gb, gw, gp, gy, gy_,gs, gt, gf = tf.gradients(f, [x, b, w, p, y, y_,s, t, f])

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="params")
#train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="input")
print( 'train_vars' )
for v in train_vars:
    print( v.name )

init = tf.initialize_all_variables()
opt = tf.train.GradientDescentOptimizer(1.0)
train = opt.minimize(f, var_list=train_vars)

with tf.Session() as sess:
    sess.run(init)
    print( 'x:%.2f, w:%.2f, b:%.2f' % (sess.run(x), sess.run(w), sess.run(b)) )
    print( 'p:%.2f, y:%.2f, y_:%.2f'% (sess.run(p), sess.run(y), sess.run(y_)) )
    print( 's:%.2f, t:%.2f, f:%.2f' % (sess.run(s), sess.run(t), sess.run(f)) )

    print( '---------- gradient ----------' )
    print( 'gx:%.2f, gw:%.2f, gb: %.2f' % (sess.run(gx), sess.run(gw), sess.run(gb)) )
    print( 'gp:%.2f, gy:%.2f, gy_:%.2f' %(sess.run(gp), sess.run(gy), sess.run(gy_)) )
    print( 'gs:%.2f, gt:%.2f, gf:%.2f' %(sess.run(gs), sess.run(gt), sess.run(gf)) )
    
    print( '---------- run GradientDescentOptimizer ----------')
    sess.run(train)
    print( 'x:%.2f, w:%.2f, b:%.2f' % (sess.run(x), sess.run(w), sess.run(b)) )
    print( 'p:%.2f, y:%.2f, y_:%.2f'% (sess.run(p), sess.run(y), sess.run(y_)))
    print( 's:%.2f, t:%.2f, f:%.2f'%(sess.run(s), sess.run(t), sess.run(f)) )

    print( '---------- gradient ----------')
    print( 'gx:%.2f, gw:%.2f, gb: %.2f' % (sess.run(gx), sess.run(gw), sess.run(gb)))
    print( 'gp:%.2f, gy:%.2f, gy_:%.2f' %(sess.run(gp), sess.run(gy), sess.run(gy_)))
    print( 'gs:%.2f, gt:%.2f, gf:%.2f' %(sess.run(gs), sess.run(gt), sess.run(gf)))
