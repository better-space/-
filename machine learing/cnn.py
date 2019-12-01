#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cnn.py
@Contact :
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/14 22:43               1.0         None
'''
import os

import matplotlib.image as matimage
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import keras
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

CATEGORY_DICT = {'棒球场': 0, '冰岛': 1, '草地': 2, '岛屿': 3, '风力发电站': 4, '港口': 5, '高尔夫球场': 6, '工厂': 7, '公路': 8, '公园': 9,
                 '海滩': 10, '旱地': 11, '河流': 12, '湖泊': 13, '火车站': 14, '机场跑道': 15, '教堂': 16, '居民区': 17, '矿区': 18,
                 '篮球场': 19, '立交桥': 20, '林地': 21, '路边停车区': 22, '裸地': 23, '墓地': 24, '桥梁': 25, '沙漠': 26, '山地': 27,
                 '商业区': 28, '石质地': 29, '水田': 30, '太阳能发电厂': 31,
                 '梯田': 32, '铁路': 33, '停车场': 34, '停机坪': 35, '网球场': 36, '温室': 37, '稀疏灌木地': 38, '油罐区': 39, '油田': 40,
                 '游泳池': 41, '直升机场': 42, '转盘': 43, '足球场': 44}

BATCH_SIZE = 128
n_Class = 45


# one-hot编码
def to_one_hot():
    one_hot_array = np.eye(45)
    # one_hot_array = np.eye(45)
    # for index, _ in enumerate(one_hot_array):
    #     one_hot_array[index] = 0.1
    #     one_hot_array[index][index] = 0.9
    for key in CATEGORY_DICT:
        CATEGORY_DICT[key] = one_hot_array[CATEGORY_DICT[key]]
        # print(key,CATEGORY_DICT[key])


def get_image_array(src):
    folder_dirs = os.listdir(src)
    flag = 1
    x_train_, y_train_ = None, None
    for folder_dir in folder_dirs:
        category = folder_dir # 类别one-hot编码
        one_hot_value = CATEGORY_DICT[folder_dir]
        folder_dir = src + f'/{folder_dir}'
        image_dirs = os.listdir(folder_dir)  # 读取当前文件夹下的所有图片样本
        features, labels = [], []
        for i in range(130):
            img = Image.open(folder_dir + f'/{image_dirs[i]}')
            img = img.resize((32, 32))  # 统一输入图像的规格
            img_array = np.array(img)
            if flag == 1:
                x_train_, y_train_ = [np.array(img_array)], [one_hot_value]
                flag = 0
            # 重复读第一张图片，其余均不重复，考虑到图片数量巨大，影响可忽略不计
            features.append(img_array)
            labels.append(one_hot_value)
        features = np.array(features)
        labels = np.array(labels)
        x_train_, y_train_ = np.concatenate([x_train_, features]), np.concatenate([y_train_, labels])
    # 打乱次序
    from sklearn.utils import shuffle
    x_train_, y_train_ = shuffle(x_train_, y_train_, random_state=1)
    return x_train_, y_train_


# 返回一个Batch大小的训练集，包含输入数据和预测结果
# x_train_[batch_num*BATCH_SIZE*image_size],y_train_[batch_num*BATCH_SIZE*categories]
def get_train_batch(dataset):
    # print(len(dataset))
    batch_num = int(len(dataset) / BATCH_SIZE)
    # print(batch_num)
    results = []
    for i in range(batch_num):
        x_train_ = dataset[:, 0][i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        y_train_ = dataset[:, 1][i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        # plt.imshow(x_train_[0])
        # plt.show()
        # x_train_ = [i.reshape((32 * 32 * 3)) for i in x_train_]
        # print(y_train_[0].shape)
        # print("***")
        results.append([x_train_, y_train_])
        # yield
    # results = minmax.fit_transform(results)
    results = np.array(results)
    return results[:, 0], results[:, 1]  # n*batch_size*image_size,n*batch_size*10


src = r'E:\课程\研究生\Machine learning II\dataset\project4\train'
to_one_hot()
x_train_, y_train_ = get_image_array(src)
print(x_train_.shape, y_train_.shape)
# 数据重塑，因为归一化函数要求数组维数小于2，类似矩阵归一化
x_train_row = x_train_.reshape(x_train_.shape[0], 32 * 32 * 3)
# 数据归一化
x_train_ = minmax.fit_transform(x_train_row)
x_train_ = x_train_.reshape(x_train_.shape[0], 32, 32, 3)
# 划分训练集和验证集
from sklearn.model_selection import train_test_split

train_ratio = 0.8
x_train_, x_val, y_train_, y_val = train_test_split(x_train_, y_train_, train_size=train_ratio, random_state=123)

# x_train_, y_train_ = get_train_batch(images_array)

# 训练常数
img_shape = x_train_.shape
epochs = 1000
batch_size = 128


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# dropout防止过拟合
keep_prod = tf.placeholder(tf.float32)

# 输入层
inputs_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='inputs_')
targets_ = tf.placeholder(tf.float32, [None, n_Class], name='targets_')
# 第一层卷积+池化
W_conv1 = weight_variable([2, 2, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(inputs_, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积+池化
W_conv2 = weight_variable([4, 4, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 重塑输出
shape = np.prod(h_pool2.get_shape().as_list()[1:])
h_pool2 = tf.reshape(h_pool2, [-1, shape])
# print(h_pool2.shape)
# 第一层全连接
W_fc1 = weight_variable([shape, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prod)

# 第二层全连接
W_fc2 = weight_variable([1024, n_Class])
b_fc2 = bias_variable([n_Class])
prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

prediction = tf.identity(prediction, name='prediction')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets_, logits=prediction))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(targets_, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

saver = tf.train.Saver()
save_model_path = './train_save'
count = 0
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print("开始了")
    batch_num = len(x_train_)  # 迭代轮数

    for epoch in range(epochs):
        for batch_i in range(img_shape[0] // batch_size - 1):
            feature_batch = x_train_[batch_i * batch_size:(batch_i + 1) * batch_size]
            label_batch = y_train_[batch_i * batch_size:(batch_i + 1) * batch_size]
            train_loss, _ = sess.run([loss, optimizer],
                                     feed_dict={inputs_: feature_batch, targets_: label_batch, keep_prod: 0.6})
            val_acc = sess.run(accuracy, feed_dict={inputs_: x_val, targets_: y_val, keep_prod: 0.8})

            # 输出当前训练情况
            if (count % 100 == 0):
                # print("***************************************************************")
                # print(train_loss)
                # print(val_acc)
                print("Epoch:{:>2},Train loss:{:.4f},Validation Accuracy:{:4f}".format(epoch + 1, train_loss, val_acc))
                # saver_path = saver.save(sess, save_model_path + f'_{i}')
            count += 1
    saver_path = saver.save(sess, save_model_path)
