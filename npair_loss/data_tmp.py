# coding: utf-8
import os
import glob

import numpy as np
import tensorflow as tf
from PIL import Image


def write_images_by_tfrecord( anc_image_obj, pos_image_obj,
                              label, writer ):
    anc_image, pos_image = np.array(anc_image_obj), np.array(pos_image_obj)
    height, width = anc_image.shape[0], anc_image.shape[1]
    # binary 化.
    anc_image_raw, pos_image_raw = anc_image.tostring(), pos_image.tostring() 
    example = tf.train.Example(features=tf.train.Features(feature={
        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "anc_image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[anc_image_raw])),
        "pos_image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos_image_raw]))
    }))
    # レコード書込
    writer.write( example.SerializeToString() )
    return

def create_npair_tfrecord( jpeg_dir, result_fpath ):
    with tf.python_io.TFRecordWriter( result_fpath ) as writer:
        for i in range(10):
            data_dir = os.path.join( jpeg_dir, str(i) )
            files = glob.glob( os.path.join( data_dir, "*.jpg" ) )
            #print(len(files))
            #print(files[0])
            label = i
            for j in range(0, int(len(files)/2) ):
                anc_image_obj = Image.open(files[2*j]).convert("L")
                pos_image_obj = Image.open(files[2*j+1]).convert("L")       
                write_images_by_tfrecord( anc_image_obj, pos_image_obj,
                                          label, writer )
    return


jpeg_dir = "mnist_jpeg_test2"
result_fpath = "npair_test.tfrecord"
create_npair_tfrecord( jpeg_dir, result_fpath )


jpeg_dir = "mnist_jpeg_train2"
result_fpath = "npair_train.tfrecord"
create_npair_tfrecord( jpeg_dir, result_fpath )
