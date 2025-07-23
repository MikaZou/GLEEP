import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def make_dataset():


class imagenet(data.Dataset):
    def __init__(self,root, split,transform=None,target_transform=None, download=None):
        '''只需要加载image就行了,
        原来的prepare_data中会输出
        train_loader, val_loader, trainval_loader, test_loader, all_loader

        prepare_data_source中调用get_train_test_loader,可以直接用
        get_train_test_loader中调用get_dataset加载train和test数据集
        get_dataset得改写

        source_forward_feature.py
        forward_pass函数只输出X_trainval_featur和X_output
        '''
        self.classes


