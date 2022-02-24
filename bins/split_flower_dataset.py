# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : yiyinghu
# @date       : 2021-11-01
# @brief      : 划分flower dataset
"""
import os.path
import random
import shutil

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):# os.path.isdir(my_dir) 存在=True，不存在=False
        os.makedirs(my_dir)

def move_img(imgs, root_dir, setname):
    data_dir = os.path.join(root_dir, setname)
    my_mkdir(data_dir)
    for idx, img_paths in enumerate(imgs):
        print('{}/{}'.format(idx, len(imgs)))
        shutil.copy(img_paths, data_dir)# 将源文件复制到新的文件夹下
    print('{} dataset, copy {} images to {}'.format(setname, len(imgs), {data_dir}))


if __name__ == '__main__':
    # 0.config
    random_seed = 20210309
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    root_dir = r'F:\cv_project_notebook\第6期课件资料\图像分类_cls\cls_01_课堂课件\102flowers'

    # 1.读取list.suffle
    data_dir = os.path.join(root_dir, 'jpg')
    img_names = os.listdir(data_dir)
    img_names = filter(lambda p: p.endswith('.jpg'), img_names)
    img_paths = [os.path.join(data_dir, name) for name in img_names]
    random.seed(random_seed)
    random.shuffle(img_paths)
    # print(img_paths)

    # 2. 划分
    train_idx = int(len(img_paths) * train_ratio)
    valid_idx = int(len(img_paths) * (train_ratio + valid_ratio))

    train_set = img_paths[:train_idx]
    valid_set = img_paths[train_idx:valid_idx]
    test_set = img_paths[valid_idx:]

    #3.复制、保存到指定文件夹
    move_img(train_set, root_dir, 'train')
    move_img(valid_set, root_dir, 'valid')
    move_img(test_set, root_dir, 'test')