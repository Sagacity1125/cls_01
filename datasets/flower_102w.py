# -*- coding: utf-8 -*-

"""
# @file name  : flower_102_w.py
# @author     : hlb
# @date       : 2021-11-02
# @brief      : 作业：flower_102数据集读取类 datasets
"""
import os
from PIL import Image
import scipy.io
from torch.utils.data import Dataset

class FlowerDataset(Dataset):
    cls_num = 102 # 类属性，用于修改网络fc层
    name = tuple([i for i in range(cls_num)])# (0,...,101)

    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = self._get_img_info()
        self.label_array = None

    def __getitem__(self, idex):
        img_path, label = self.img_info[idex]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception('\ndata_dir:{} is a empty dir!'.format(self.root_dir))
        return len(self.img_info)

    def _get_img_info(self):
        '''
        返回一个list=[(path1,label1),...]
        :return:
        '''
        img_names = os.listdir(self.root_dir) #xx.jpg
        img_names = [n for n in img_names if n.endswith('.jpg')]

        # 读取label.mat
        label_file = 'imagelabels.mat'
        label_path = os.path.join(self.root_dir, '..', '..',label_file)
        # 'labels': array([[77, 77, 77, ..., 62, 62, 62]], dtype=uint8)}  1,2,...102
        label_array = scipy.io.loadmat(label_path)['labels'].squeeze()
        self.label_array = label_array

        # 匹配image和label
        idx_imgs = [int(name[6:11]) for name in img_names]# int可以去掉前面的0, idx=1,...,2,3...
        img_paths = [os.path.join(self.root_dir, name) for name in img_names]
        img_info = [(p, int(label_array[idx-1]-1)) for p, idx in zip(img_paths, idx_imgs)]
        return img_info

if __name__=='__main__':
    root_dir = r'F:\cv_project_notebook\cv_Issue6_data\ImageClass\cls_01\cls_01_ppt\data\102flowers\train'
    flower = FlowerDataset(root_dir)
    print(len(flower))
    print(next(iter(flower)))