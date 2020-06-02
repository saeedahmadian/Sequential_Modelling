import pandas as pd
import tensorflow as tf
import numpy as np
import os




def read_chunks(datadir,chunk_size=10000):
    chunks = pd.read_csv(datadir,index_col='DATE_FOR', parse_dates=True,chunksize=chunk_size)
    for chunk in chunks:
        yield chunk

def seperate_data(chunk):
    string_data = chunk.select_dtypes('object')
    float_data = chunk.select_dtypes('float')
    int_data = chunk.select_dtypes('int64')
    return string_data,float_data,int_data

def feature_byte(value):
    "value_list must be in list format"
    value_list= np.char.encode(value.astype('str').tolist(),'utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))

def feature_float(value_list):
    "value_list must be in list format"
    value_list= value_list.astype('float32')
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

def feature_int(value_list):
    "value_list must be in list format"
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))



def create_description(string_feat, float_feat, int_feat):
    string_cols = string_feat.columns.tolist()
    float_cols = float_feat.columns.tolist()
    int_cols = int_feat.columns.tolist()
    feature_description = dict()
    for name, val in zip(string_cols, np.transpose(string_feat.values)):
        feature_description[name] = tf.io.FixedLenFeature([], tf.string, default_value='')

    for name, val in zip(float_cols, np.transpose(float_feat.values)):
        feature_description[name] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)

    for name, val in zip(int_cols, np.transpose(int_feat.values)):
        feature_description[name] = tf.io.FixedLenFeature([], tf.int64, default_value=0)

    return feature_description

def create_serialize_tfrecord(string_feat,float_feat,int_feat):
    string_cols = string_feat.columns.tolist()
    float_cols = float_feat.columns.tolist()
    int_cols = int_feat.columns.tolist()
    features=dict()
    feature_description= dict()
    for name,val in zip(string_cols,np.transpose(string_feat.values)):
        features[name]= feature_byte(val)


    for name,val in zip(float_cols,np.transpose(float_feat.values)):
        features[name]= feature_float(val)


    for name,val in zip(int_cols,np.transpose(int_feat.values)):
        features[name]= feature_int(val)


    tf_features = tf.train.Features(feature=features)
    tf_example= tf.train.Example(features=tf_features)
    return tf_example.SerializeToString(),features

data_dir=os.path.join('./Data','DS_MiniProject_ANON.csv')
tf_dir= 'TFRecord'
chunks=pd.read_csv(data_dir,index_col='DATE_FOR', parse_dates=True,chunksize=10000)
for i,chunk in enumerate(chunks):
    string_feat,float_feat,int_feat= seperate_data(chunk)
    tf_record,features= create_serialize_tfrecord(string_feat,float_feat,int_feat)
    if i==0:
        feature_description=create_description(string_feat,float_feat,int_feat)
    # writer=tf.data.experimental.TFRecordWriter('{}/dataset_{}.tfrecord'.format(tf_dir,i))
    writer = tf.io.TFRecordWriter('{}/dataset_{}.tfrecord'.format(tf_dir,i))
    writer.write(tf_record)

# with tf.python_io.TFRecordWriter('TF.tfrecord') as writer:
#   writer.write(example.SerializeToString())
# feature_description = {
#     'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
# }

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

raw_dataset = tf.data.TFRecordDataset('TFRecord/dataset_0.tfrecord')

# for raw_record in raw_dataset.take(10):
#     exmple= tf.train.Example()
#     exmple.ParseFromString(raw_record.numpy())
#     print(exmple)



parsed_dataset = raw_dataset.map(_parse_function)
# parsed_dataset= parsed_dataset.batch(2)
for raw_rec in parsed_dataset.take(10):
    raw_rec
    print(repr(raw_rec))

a=1






