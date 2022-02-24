# -*- coding:  utf-8 -*-
"""
# @file name  : flower_102.py
# @author     : yiyinghu
# @date       : 2021-11-02
# @brief      : if __name__ == '__main__':的作用
"""

# print('this is one')
# print('__name__:',__name__)
# if __name__ == '__main__':
#     print('this is two')




import torchvision.transforms as transforms
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
parent_path = os.path.dirname(sys.path[0])
from datasets.flower_102 import FlowerDataset


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from ImageClass.cls_01.cls_01_ppt.img_cls.tools.model_trainer import ModelTrainer
from datetime import datetime
train_dir = r"F:\cv_project_notebook\cv_Issue6_data\ImageClass\cls_01\cls_01_ppt\data\102flowers\train"
train_data = FlowerDataset(root_dir=train_dir)

print(BASE_DIR)