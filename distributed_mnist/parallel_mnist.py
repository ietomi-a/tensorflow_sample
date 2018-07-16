#coding:utf-8

import os
import re
import time
from datetime import datetime

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.99     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.96  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
TOWER_NAME = 'tower'


def set_FLAG_parameter():
    tf.app.flags.DEFINE_string('train_dir', 'MNIST_Dataset',
                               """Directory where to read data, write event logs """
                               """and checkpoint.""")

    tf.app.flags.DEFINE_integer('read_thread_num', 2,
                                """Number of the threads reading the input file.""")

    tf.app.flags.DEFINE_integer('batch_size', 128,
                                """Size of the training batch.""")

    tf.app.flags.DEFINE_integer('max_steps', 2000,
                                """Number of batches to run.""")

    tf.app.flags.DEFINE_integer('num_gpus', 1,
                                """How many GPUs to use.""")

    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")


def forward_propagation( X, layer_hidden_nums, training,
                         dropout_rate=0.01, regularizer_scale=0.01 ):
    """
    Implements the forward propagation for the model
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    Returns:
    Z5 -- the output of the last LINEAR unit
    """
    he_init = tf.contrib.layers.variance_scaling_initializer()
    l1_regularizer = tf.contrib.layers.l1_regularizer(regularizer_scale)

    A_drop = X # 最初のレイヤー.
    for layer_index, layer_neurons in enumerate(layer_hidden_nums[:-1]):
        Z = tf.layers.dense( inputs=A_drop, units=layer_neurons,
                             kernel_initializer=he_init,
                             kernel_regularizer=l1_regularizer,
                             name="hidden%d" % (layer_index + 1))
        A = tf.nn.elu(Z)
        A_drop = tf.layers.dropout( A, dropout_rate, training=training,
                                    name="hidden%d_drop" % (layer_index + 1))

    # don't do normalization for the output layer
    Z_output = tf.layers.dense( inputs=A_drop, units=layer_hidden_nums[-1],
                                kernel_initializer=he_init, name="output")
    return Z_output

def set_loss( logits, labels ):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast( labels, tf.int64 )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example' )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection( 'losses', cross_entropy_mean )

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n( tf.get_collection('losses'), name='total_loss' )

def tower_loss( scope, images, labels):
    # images, labels : <class 'tensorflow.python.framework.ops.Tensor'> <class 'tensorflow.python.framework.ops.Tensor'>
    # images.shape, labels.shape :(128, 784) (128,)

    """Calculate the total loss on a single tower running the DNN model."""
    # Build inference Graph.
    layer_hidden_nums = [200, 100, 50, 25, 10] # output は MNISTなので 10.
    logits = forward_propagation( images, layer_hidden_nums, True )
    set_loss( logits, labels ) # "losses " の所に loss を付け加える
    correct = tf.nn.in_top_k( logits, labels, 1)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection( 'losses', scope )
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n( losses, name='total_loss' )

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss, correct

def average_gradients( tower_grads ):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars: # ここでは var の方は使わない.
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append( expanded_g )

        # Average over the 'tower' dimension.
        grad = tf.concat( axis=0, values=grads )
        grad = tf.reduce_mean( grad, 0 ) # 平均をとる.

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1] # v は共通なのでとりあえず一つ取ってくる。
        grad_and_var = (grad, v)
        average_grads.append( grad_and_var )
    return average_grads

#set the num_epochs to None, will cycle through the strings in string_tensor an unlimited number of times
def get_input_data( filename, batch_size, read_thread_num ):
    filename = os.path.join( FLAGS.train_dir, filename + '.tfrecords' )
    print('Reading', filename)
    with tf.name_scope('inputs'):
        filename_queue = tf.train.string_input_producer( [filename], num_epochs=None )
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read( filename_queue ) # tf.TFRecordReader().read() only accept queue as param
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'height': tf.FixedLenFeature( [], tf.int64),
                'width': tf.FixedLenFeature( [], tf.int64),
                'label': tf.FixedLenFeature( [], tf.int64),
                'image_raw': tf.FixedLenFeature( [], tf.string),
            } )
        image = tf.decode_raw( features['image_raw'], tf.uint8 )
        #height = tf.cast(features['height'], tf.int32)
        #width = tf.cast(features['width'], tf.int32)
        label = tf.cast(features['label'], tf.int32)
        image.set_shape([28 * 28])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5 # 正規化しておく.

        # Shuffle the examples and collect them into batch_size batches. (Internally uses a RandomShuffleQueue)
        # We run this in two threads to avoid being a bottleneck.
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=read_thread_num,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000 )

        return images_batch, labels_batch


def train2():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable( 'global_step', [],
                                       initializer=tf.constant_initializer(0),
                                       trainable=False )
        # Calculate the learning rate schedule.
        num_batches_per_epoch = (60000 / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch)
        # use stair learning rate decay or gradually decay
        learning_rate \
            = tf.train.exponential_decay( INITIAL_LEARNING_RATE, global_step, decay_steps,
                                          LEARNING_RATE_DECAY_FACTOR, staircase=True )
        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        images_batch, labels_batch = get_input_data( 'mnist_train',
                                                     FLAGS.batch_size,
                                                     FLAGS.read_thread_num )
        #print( type(images_batch), type(labels_batch) )
        #print( images_batch.shape, labels_batch.shape )
        # Calculate the gradients for each model tower.
        tower_grads, correct_sum = [], []
        with tf.variable_scope(tf.get_variable_scope()) as var_scope:
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i), tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    loss, correct = tower_loss( scope, images_batch, labels_batch )
                    correct_sum.append( tf.cast(correct, tf.float32) )
                    var_scope.reuse_variables()
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this tower.
                    grads = opt.compute_gradients( loss )
                    # for grad in grads:
                    #     print("grad: ", grad)
                        
                    tower_grads.append( grads )

        # accuracy is calculated summarize all reconigtion results
        correct_sum = tf.reshape( correct_sum, [-1] )
        accuracy = tf.reduce_mean( correct_sum, name='accuracy' )

        # 各サーバーからの gradient を足し合わせて平均をとる。
        grads = average_gradients( tower_grads )

        # ログ出力設定
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # 勾配による変数の値の更新
        apply_gradient_op = opt.apply_gradients( grads, global_step=global_step)  # the global_step will increase by one

        # ログ出力設定.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step )
        variables_averages_op = variable_averages.apply( tf.trainable_variables() ) 

        # Group all updates to into a single train op.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group( apply_gradient_op, variables_averages_op )

        # Create a saver.
        saver = tf.train.Saver( tf.global_variables() )

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge( summaries )

        # Build an initialization operation to run below.
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer() )

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session( config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement) )
        sess.run( init_op )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # スタート開始のログ記載.
        now  = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = FLAGS.train_dir
        log_dir = "{}/run-{}-log".format(root_logdir, now)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value, accuracy_value = sess.run( [train_op, loss, accuracy] )
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if step % 10 == 0:
                # 計算時間と精度を出力.
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch), train_accuracy = %f')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch, accuracy_value))
                
            if step % 100 == 0: # サマリーを出力
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                check_point_dir = "{}/run-{}-checkpoint".format(root_logdir, now)
                checkpoint_path = os.path.join(check_point_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
        coord.request_stop()
        coord.join(threads)
        summary_writer.close()
        sess.close()
        
    print("train2 ok")
        
if __name__ == "__main__":
    set_FLAG_parameter()
    train2()
    print("ok tmp2")
