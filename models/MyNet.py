import torchvision
import torch
from torchvision import datasets,transforms
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
# -*- coding: utf-8 -*-
class MyNet(torch.nn.Module):

    def __init__(self):
        super(MyNet,self).__init__()
        # 第0个卷积层 '3'表示输入图片为多通道, '16'表示输出通道数，'7'表示卷积核为7*7
        self.conv01 = nn.Conv2d(3, 16, 3,stride =1,padding =1)
        self.conv02 = nn.Conv2d(16, 16, 3,stride =1,padding =1)
        # 第1个卷积层
        self.conv11 = nn.Conv2d(16, 32, 3,stride =1,padding =1)
        self.conv12 = nn.Conv2d(32, 32, 3,stride =1,padding =1)
        self.conv13 = nn.Conv2d(32, 48, 3,stride =1,padding =1)
        self.conv14 = nn.Conv2d(48, 48, 3,stride =1,padding =1)
        self.conv15 = nn.Conv2d(48, 64, 3,stride =1,padding =1)
        self.conv16 = nn.Conv2d(64, 64, 3,stride =1,padding =1)
        # 第2个卷积层
        self.conv21 = nn.Conv2d(64, 96, 3,stride =1,padding =1)
        self.conv22 = nn.Conv2d(96, 96, 3,stride =1,padding =1)
        self.conv23 = nn.Conv2d(96, 128, 3,stride =1,padding =1)
        self.conv24 = nn.Conv2d(128, 128, 3,stride =1,padding =1)
        # 第3个卷积层
        self.conv31 = nn.Conv2d(128, 160, 3,stride =1,padding =1)
        self.conv32 = nn.Conv2d(160, 160, 3,stride =1,padding =1)
        self.conv33 = nn.Conv2d(160, 160, 3,stride =1,padding =1)
        self.conv34 = nn.Conv2d(160, 160, 3,stride =1,padding =1)
        # reshape

        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(160 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        print("Constructor Finished")

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        PRelu = nn.ReLU()
        x = PRelu(self.conv01(x))
        x = F.max_pool2d(PRelu(self.conv02(x)), (2, 2))
        x = PRelu(self.conv11(x))
        x = PRelu(self.conv12(x))
        x = PRelu(self.conv13(x))
        x = PRelu(self.conv14(x))
        x = PRelu(self.conv15(x))
        x = F.max_pool2d(PRelu(self.conv16(x)), (2, 2))

        x = PRelu(self.conv21(x))
        x = PRelu(self.conv22(x))
        x = PRelu(self.conv23(x))
        x = F.max_pool2d(PRelu(self.conv24(x)), (2, 2))

        x = PRelu(self.conv31(x))
        x = PRelu(self.conv32(x))
        x = PRelu(self.conv33(x))
        x = PRelu(self.conv34(x))

        # reshape，‘-1’表示自适应

        x = x.view(-1,32*32*160)
        # dropout

        dp1 = torch.nn.Dropout(0.1)
        x = dp1(x)
        # full connected layer
        x = self.fc1(x)
        dp2 =torch.nn.Dropout(0.5)
        x = dp2(x)
        x = self.fc2(x)
        return x

net = MyNet()
#
# #print(net)
# torch.save(net,'hh.pth')
# model = torch.load('hh.pth')
# print(model)
# # for name, parameters in net.named_parameters():
# #     print(name, ':', parameters.size())