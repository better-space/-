#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cnn.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/14 22:43   gxrao      1.0         None
'''
import os

import matplotlib.image as matimage
import numpy as np

# print("jjjj")

CATEGORY_DICT = {'棒球场': 0, '冰岛': 1, '草地': 2, '岛屿': 3, '风力发电站': 4, '港口': 5, '高尔夫球场': 6, '工厂': 7, '公路': 8, '公园': 9,
                 '海滩': 10, '旱地': 11, '河流': 12, '湖泊': 13, '火车站': 14, '机场跑道': 15, '教堂': 16, '居民区': 17, '矿区': 18,
                 '篮球场': 19, '立交桥': 20, '林地': 21, '路边停车区': 22, '裸地': 23, '墓地': 24, '桥梁': 25, '沙漠': 26, '山地': 27,
                 '商业区': 28, '石质地': 29, '水田': 30, '太阳能发电厂': 31,
                 '梯田': 32, '铁路': 33, '停车场': 34, '停机坪': 35, '网球场': 36, '温室': 37, '稀疏灌木地': 38, '油罐区': 39, '油田': 40,
                 '游泳池': 41, '直升机场': 42, '转盘': 43, '足球场': 44}

BATCH_SIZE = 20


# one-hot编码
def to_one_hot():
    one_hot_array = np.eye(45)
    for key in CATEGORY_DICT:
        CATEGORY_DICT[key] = one_hot_array[CATEGORY_DICT[key]]
        # print(key,CATEGORY_DICT[key])


def get_image_array(src):
    images_array = []
    folder_dirs = os.listdir(src)
    # print(folder_dirs)
    for folder_dir in folder_dirs:
        # 类别one-hot编码
        category = folder_dir
        one_hot_value = CATEGORY_DICT[folder_dir]
        # print(one_hot_value)

        folder_dir = src + f'/{folder_dir}'
        image_dirs = os.listdir(folder_dir)  # 读取当前文件夹下的所有图片样本

        for i in range(1):
            img_array = matimage.imread(folder_dir + f'/{image_dirs[i]}')
            print(img_array.shape)
            images_array.append(np.array([img_array, one_hot_value]))
    images_array = np.array(images_array)
    print(images_array.shape)
    np.random.shuffle(images_array)
    return images_array


def get_train_batch(dataset):
    print(len(dataset))
    batch_num = int(len(dataset)/BATCH_SIZE)
    print(batch_num)
    results = []
    for i in range(3):
        x_train_ = dataset[:,0][i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        y_train_ = dataset[:,1][i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        x_train_ = x_train_.reshape((BATCH_SIZE,512*512*3))
        print(x_train_.shape,y_train_.shape)
        # results.append(x_train_,y_train_)
        # yield

    # results = np.array(results)
    # return results[:,0],results[:,1]


src = r'C:\Users\Suhe\Desktop\Machine learning II\dataset\project4\train'
to_one_hot()
images_array = get_image_array(src)
print(images_array.shape)
# images_array = np.array(images_array)
# x_train_,y_train_ = get_train_batch(images_array)
# get_train_batch(images_array)
# print(images_array[:, 1])
# print(images_array.shape)

# print(CATEGORY_DICT)
