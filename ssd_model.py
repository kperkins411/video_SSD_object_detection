# from fastai.conv_learner import *
# from fastai.dataset import *
import utilities as ut
import torch


import numpy as np
'''
a simple first model that predicts what object is located in each cell of a 4x4 grid.
k= number of default boxes
'''
class StdConv(torch.nn.Module):
    def __init__(self, nin, nout,kernel_size=3, stride=2, drop=0.1,padding=1):
        '''
        :param nin: the depth of the input w*h*nin
        :param nout: the depth of the output w*h*nout
        '''
        super().__init__()
        self.conv = torch.nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(nout)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.bn(torch.nn.functional.relu(self.conv(x))))

def flatten_conv(x, k):
    bs, nf, gx, gy = x.size()
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(bs, -1, nf // k)

class OutConv(torch.nn.Module):
    def __init__(self, k, nin, bias,numb_categories):
        '''
        :param k: number of default boxes(anchor boxes), see SSD paper
        :param nin:
        :param bias: bias per neuron
        :param numb_categories:
        '''
        super().__init__()
        self.k = k

        #classes
        self.oconv1 = torch.nn.Conv2d(nin, numb_categories * k, kernel_size=3, padding=1)

        #default boxes
        self.oconv2 = torch.nn.Conv2d(nin, 4 * k, kernel_size=3, padding=1)
        self.oconv1.bias.data.zero_().add_(bias)

    def forward(self, x):
        return [flatten_conv(self.oconv1(x), self.k),
                flatten_conv(self.oconv2(x), self.k)]


class SSD_MultiHead(torch.nn.Module):
    def __init__(self, k, bias,numb_categories, drop=0.4, ):
        super().__init__()
        self.numb_categories = numb_categories
        self.drop = torch.nn.Dropout(drop)
        self.sconv0 = StdConv(512,256, stride=1, drop=drop)
        self.sconv1 = StdConv(256,256, drop=drop)
        self.sconv2 = StdConv(256,256, drop=drop)
        self.sconv3 = StdConv(256,256, drop=drop)
        self.out0 = OutConv(k, 256, bias, self.numb_categories)
        self.out1 = OutConv(k, 256, bias, self.numb_categories)
        self.out2 = OutConv(k, 256, bias, self.numb_categories)
        self.out3 = OutConv(k, 256, bias, self.numb_categories)

    def forward(self, x):
        x = self.drop(torch.nn.functional.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        o1c,o1l = self.out1(x)
        x = self.sconv2(x)
        o2c,o2l = self.out2(x)
        x = self.sconv3(x)
        o3c,o3l = self.out3(x)
        return [torch.cat([o1c,o2c,o3c], dim=1),
                torch.cat([o1l,o2l,o3l], dim=1)]

def get_model(id2cat,md,k):
    '''
    create the model
    :return: the model
    '''
    PRETRAINED_MODEL = 'resnet34'
    IMG_SZ = 224
    BATCH_SIZE = 64
    numb_categories = len(id2cat) +1

    n_act = k * (4 + n_clas)

    head_reg4 = SSD_MultiHead(k, -3., numb_categories)
    models = ConvnetBuilder(PRETRAINED_MODEL, 0, 0, 0, custom_head=head_reg4)
    learn = ConvLearner(md, models)
    learn.opt_fn = optim.Adam
    return learn