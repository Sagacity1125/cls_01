# -*- coding: utf-8 -*-
"""
# @file name  : flower_train.py
# @author     : yiyinghu
# @date       : 2021-11-01
# @brief      : flower数据集训练主代码
"""
import os,sys
from datetime import datetime
from torch.utils.data import dataset
from ImageClass.cls_01.cls_01_code.datasets.flower_102 import FlowerDataset
import torchvision.transforms as transforms
from torchvision.models import resnet18

BASE_DIR = os.path.relpath('.')# 获得当前文件所在目录



if __name__ == '__main__':
    # 0.config
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d-%H-%M')
    log_dir = os.path.join(BASE_DIR, '..', 'results', time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_dir = r'F:\cv_project_notebook\cv_Issue6_data\ImageClass_cls\cls_01_ppt\102flowers\train'
    vaild_dir = r'F:\cv_project_notebook\cv_Issue6_data\ImageClass_cls\cls_01_ppt\102flowers\vaild'

    # path_state_dict = r''

    # 1. 数据

    # 2. 模型

    # 3. 损失函数、优化器

    # 4. 迭代训练
