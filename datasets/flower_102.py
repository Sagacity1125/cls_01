# -*- coding:  utf-8 -*-
"""
# @file name  : flower_102.py
# @author     : yiyinghu
# @date       : 2021-11-01
# @brief      : flower_102数据集读取类 datasets
"""
import os
import scipy.io as scio
from PIL import Image

from torch.utils.data import Dataset

class FlowerDataset(Dataset):
    cls_num = 102
    names = tuple([i for i in range(cls_num)])

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = self._get_img_info()
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        '''
        输入index，从硬盘中读数据，并处理得到to tensor，返回一个可以直接送入模型的样本
        :param index:
        :return:
        '''
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception('\ndata_dir:{} is a empty dir! please checkout your path to images!'.format(self.root_dir))
        return len(self.img_info)

    def _get_img_info(self):
        '''
        在类的初始化时调用该函数，读取数据的基本信息,存在list中，供getitem使用
        list=[(path1,label1),...,...]
        :return:
        '''
        # root_dir = r'F:\cv_project_notebook\第6期课件资料\图像分类_cls\cls_01_课堂课件\102flowers\train'
        names_imgs = os.listdir(self.root_dir)
        names_imgs = [n for n in names_imgs if n.endswith('.jpg')]# 得到图片名

        # 读取mat形式的label
        label_file = '../../cls_01_ppt/imagelabels.mat'
        # 'labels': array([[77, 77, 77, ..., 62, 62, 62]], dtype=uint8)}  1-102
        label_array = scio.loadmat(label_file)['labels'].squeeze()
        self.label_array = label_array

        # 匹配标签
        idx_imgs = [int(name[6:11]) for name in names_imgs]# int可以去掉前面的0, idx=1,...
        path_imgs = [os.path.join(self.root_dir, name) for name in names_imgs]
        img_info = [(p, int(label_array[idx-1]-1)) for p, idx in zip(path_imgs, idx_imgs)]
        return img_info

if __name__ == '__main__':
    root_dir = r'F:\cv_project_notebook\cv_Issue6_data\ImageClass\data\102flowers\train'
    flower = FlowerDataset(root_dir)
    print(len(flower))
    print(flower[0])
    # print(next(iter(flower)))