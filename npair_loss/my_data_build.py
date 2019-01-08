# coding: utf-8

import os

import numpy as np
from PIL import Image
import glob

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def numpy_ndarray_to_jpeg_files( images, shape, dir_path, labals ):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    images = images * 255  # 画像データの値を0～255の範囲に変更する
    for i in range( images.shape[0] ):
        image_array = images[i].reshape(shape)
        out_img = Image.fromarray(image_array)
        out_img = out_img.convert("L")  # グレースケール
        l = str(np.argmax(labals[i]))
        if not os.path.exists( os.path.join( dir_path,l) ):
            os.mkdir( os.path.join( dir_path,l) )
        fname = str(i) + "-" + l + ".jpg"
        out_img.save( os.path.join( dir_path, l, fname ), format="JPEG")
    return

def write_image_by_tfrecord( image_obj, label, writer ):
    image = np.array(image_obj)
    height = image.shape[0]
    width = image.shape[1]
    image_raw = image.tostring()  # binary 化.
    example = tf.train.Example(features=tf.train.Features(feature={
        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
    }))
    # レコード書込
    writer.write( example.SerializeToString() )
    return

def fpath_list_to_tfrecord( fpath_list, result_fpath ):
    with tf.python_io.TFRecordWriter( result_fpath ) as writer:
        for fpath in fpath_list:
            fname = os.path.basename(fpath)
            label = int( fname[ fname.rfind("-") + 1 : -4] )  # ファイル名からラベルを取得
            with Image.open(fpath).convert("L") as image_obj:  # グレースケール
                write_image_by_tfrecord( image_obj, label, writer )
    return
 
def tfrecord_to_jpegs( tfrecord_fpath, jpeg_dir ):
    if not os.path.exists(jpeg_dir):
        os.mkdir(jpeg_dir)
    iterator = tf.python_io.tf_record_iterator(tfrecord_fpath)    
    for i,record  in enumerate(iterator):
        example = tf.train.Example()
        example.ParseFromString(record)  # バイナリデータからの読み込み
        height = example.features.feature["height"].int64_list.value[0]
        width = example.features.feature["width"].int64_list.value[0]
        label = example.features.feature["label"].int64_list.value[0]
        image_array = example.features.feature["image"].bytes_list.value[0]

        image = np.fromstring( image_array, dtype=np.uint8)
        image = image.reshape([height, width])
        img = Image.fromarray( image, "L")  # グレースケール
        fname = "tfrecords_{0}-{1}.jpg".format(i, label )
        img.save( os.path.join( jpeg_dir, fname ) )
    return


def read_and_decode_by_tensorflow(fname_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fname_queue)  # 次のレコードの key と value が返ってきます
    features = tf.parse_single_example(
        serialized_example,
        features={
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64),
            "image": tf.FixedLenFeature([], tf.string),  # binary を tf.string で受け取る.
        })
    image_raw = tf.decode_raw(features["image"], tf.uint8)
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    label = tf.cast(features["label"], tf.int32)
    image = tf.reshape( image_raw, tf.stack([height, width]) )
    return image, label


def tensorflow_read_and_write( tfrecord_fpath, result_dir ):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fname_queue = tf.train.string_input_producer([tfrecord_fpath])  # TFRecordファイルからキューを作成
    images_gen, labels_gen = read_and_decode_by_tensorflow(fname_queue)  # キューからデコードされた画像データとラベルを取得する処理を定義

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初期化
        try:
            coord = tf.train.Coordinator()  # スレッドのコーディネーター
            threads = tf.train.start_queue_runners(coord=coord)  # グラフで収集されたすべてのキューランナーを開始
            for i in range(10000):
                image, label = sess.run([images_gen, labels_gen])  # 次のキューから画像データとラベルを取得
                img = Image.fromarray( image, "L" )  # グレースケール
                fname = "tfrecords_{0}-{1}.jpg".format(str(i), label)  
                img.save( os.path.join( result_dir, fname ) )  # 画像を保存
        finally:
            coord.request_stop()  # すべてのスレッドが停止するように要求
            coord.join(threads)  # スレッドが終了するのを待つ
    return


# jpeg ファイルの作成.
mnist = read_data_sets("MNIST_data",one_hot=True)
# print(type(mnist.test.images))  # numpy.ndarray
# print(mnist.test.images.shape)  # (10000, 784)
jpeg_dir = "mnist_jpeg_train2"
numpy_ndarray_to_jpeg_files( mnist.train.images, (28,28),
                             jpeg_dir, mnist.train.labels )
# jpeg_dir = "mnist_jpeg_test2"
# numpy_ndarray_to_jpeg_files( mnist.test.images, (28,28),
#                              jpeg_dir, mnist.test.labels )

# tfrecord 作成.
#fpath_list = glob.glob( os.path.join( jpeg_dir , "*.jpg") )
#tfrecord_fpath = "mnist_train.tfrecord"
# fpath_list_to_tfrecord( fpath_list, tfrecord_fpath )

# tfrecord の読み込み.
# jpeg_dir2 = "from_tfrecord_jpeg"
# tfrecord_to_jpegs( tfrecord_fpath, jpeg_dir2 )

# tensorflow での読み込み.
# jpeg_dir3 = "from_tfrecord_jpeg_by_tf"
# tensorflow_read_and_write( tfrecord_fpath, jpeg_dir3 )
