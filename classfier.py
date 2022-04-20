import os
os.environ["CUDA_VISIBLE_DEVICES"]=0


import torch
import torchvision.transforms.functional as T
import torchvision
import torch.nn as nn
import numpy as np
import cv2
import math
import random
import torchvision.models as models






#定义数据
class MyDataset:
    def __init__(self,directory):
        self.directory = directory
        self.files = os.listdir(directory)
    def __len__(self):
        return 0
    def __getitem__(self, item):
        return 0
fs=os.listdir("train")

#定义模型
# class Resnet18():



#定义DataLoader 多线程加载数据
train_dataset = MyDataset("./train")
train_dataloader =11

#定义模型实例
model = models.resnet18()
model.cuda


#定义损失函数 loss
loss_function = nn.CrossEntropyLoss()
#定义优化方法
op = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9)
#SGD Adam
cross_entropy = nn.CrossEntropyLoss()
# This is a real try for git
#执行循环优化过程
#lr迭代策略
