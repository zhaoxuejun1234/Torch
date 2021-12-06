##搭建一个Lenet
##搭建一个Resnet18
##手写分类器
##手写检测器
# import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
class LeNet(nn.Module):
    def __init__(self):
        # LeNet.__init__(self)
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,16,5)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,5)
        self.pool2=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.pool2(x)
        x=x.view(-1,32*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc(3)
        return x

#LeNet2
class LeNett(nn.Module):
    def __init__(self):
        super(LeNett,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,8,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8*5*5,20)
        self.fc2 =nn.Linear(20,10)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x =self.pool2(x)
        x= x.view(-1,8*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.relu(x)
        return x
