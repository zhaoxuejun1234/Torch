import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


resnet_18 = models.resnet18(pretrained=True)
vgg = models.vgg19(pretrained=True)
#Resnet18/34
class Resblock(nn.Module):
    expansion = 1
    def __init__(self,inchannel,outchannel,stride =1):
        super(Resblock,self).__init__()
        self.left = nn.Sequential(
        nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(inplace=True),
        nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride!=1 or inchannel!=outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self,x):
        out =self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet18(nn.Module):
    def __init__(self,Resblock,num_classes=10):
        super(Resnet18,self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU
        )
    def _make_layer(self,block,channel,blocks,stride):
        downsample = None
        if stride!=1 or self.inchannel!=channel*block.expansion:
            downsample = nn.Sequential(
                nn.conv(self.inchannel,channel*block.expansion,stride),

            )

    def forward(self,x):
        x = F.relu(self.conv1(x))