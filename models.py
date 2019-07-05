# encoding: utf-8

import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen
import torchsnooper

#private
class CNN(nn.Module):#lenet
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = torch.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)#4x4x2, 32 output

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)#置为bias
        self.cnn.fc2.weight.data.zero_()#置位0，shape = 32*50

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)

#普通cnn
class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

   # @torchsnooper.snoop()
    def forward(self, x):
        y = self.cnn(x)#[B,1,28,28] 2 [B,10]
        ret = F.log_softmax(y,dim = 1)#[b,10], 对1号dim变换
        return ret

class STNClsNet(nn.Module):

    def __init__(self, args):
        super(STNClsNet, self).__init__()
        self.args = args

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim = 1)#X Y， 16个坐标点4X4 ，代表9个格子
        target_control_points = torch.cat([X, Y], dim = 1)

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }[args.model]
        self.loc_net = GridLocNet(args.grid_height, args.grid_width, target_control_points)

        self.tps = TPSGridGen(args.image_height, args.image_width, target_control_points)

        self.cls_net = ClsNet()

    def forward(self, x):
        batch_size = x.size(0)
        #1
        source_control_points = self.loc_net(x)#[B,16,2]

        #2
        source_coordinate = self.tps(source_control_points)#source_coordinate size = [B,784,2]
        grid = source_coordinate.view(batch_size, self.args.image_height, self.args.image_width, 2)

        #3
        transformed_x = grid_sample(x, grid)#根据映射，采样出变换后的图片


        logit = self.cls_net(transformed_x)
        return logit


class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
