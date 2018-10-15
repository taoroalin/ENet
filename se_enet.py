# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 23:35:51 2018

@author: tao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
random.seed(1)

class SE_spatial(nn.Module):
    
    def __init__(self, c, ratio=16):
        super(SE_spatial, self).__init__()
        self.c=c
        self.l1=nn.Parameter(
                torch.normal(0, 1, (c, c//ratio)).clamp(-2, 2)*np.sqrt(2/c))
        self.l2=nn.Parameter(
                torch.normal(0, 1, (c, c//ratio)).clamp(-2, 2)*np.sqrt(2*ratio/c))
        
    def forward(self, x):
        pool = F.adaptive_avg_pool2d(x, 1).reshape((-1, 1))
        f1=(pool*self.l1).clamp(0)
        f2=torch.sigmoid(f1*self.l2)
        o=x*f2.reshape((-1, self.c, 1, 1))
        return o

class SE_depth(nn.Module):
    
    def __init__(self, c):
        super(SE_depth, self).__init__()
        self.conv1=nn.Conv2d(c, 1, (1,1))
        
    def forward(self, x):
        ex=torch.sigmoid(self.conv1(x))
        o=ex*x
        return o

class SE(nn.Module):
    
    def __Init__(self, c, ratio=16):
        super(SE, self).__init__()
        self.spatial=SE_spatial(c, ratio)
        self.depth=SE_depth(c)
        
    def forward(self, x):
        space=self.spatial(x)
        depth=self.depth(x)
        return space+depth

class Init_block(nn.Module):
    def __init__(self, i, f):
        super(Init_block, self).__init__()
        self.conv1=nn.Conv2d(i, f-1, (3, 3), padding=1)
    def forward(self, x):
        c1=self.conv1(x)
        return torch.cat([x, c1], 1)
    
class Bottleneck(nn.Module):
    def __init__(self, in_filters, out_filters, mode='regular', 
                 dilation=1, downsample=0, upsample=0, sq_ratio=2, dropout=0, sq_drop=0):
        super(Bottleneck, self).__init__()
        self.in_filters=in_filters
        self.out_filters=out_filters
        self.downs=downsample
        self.ups=upsample
        down=in_filters//sq_ratio
        if out_filters<=down:
            self.middle=down
        else:
            self.middle=down*int(np.sqrt(out_filters/down))
        middle=self.middle
        self.dropsamples = out_filters-int(out_filters*(1-dropout))
        self.sq_dropsamples = middle-int(middle*(1-sq_drop))
        if downsample!=0:
            self.ppool=nn.AdaptiveMaxPool2d(downsample, return_indices=True)
        if upsample!=0:
            if upsample[0] % 2 == 0:
                self.ppool=nn.MaxUnpool2d((2,2))
            else:
                self.ppool=lambda x, y: F.interpolate(x, upsample)
        #Conv path
        self.sq=nn.Conv2d(in_filters, down, kernel_size=(1, 1), bias=False, padding=0)
        self.bn1=nn.BatchNorm2d(out_filters)
        if mode=='regular':
            if downsample:
                if downsample[0] % 2 == 1:
                    self.m=self.m=nn.Conv2d(down, middle, kernel_size=(3, 3), dilation=dilation, stride=(2, 2), padding=1)
                else:
                    self.m1=nn.Conv2d(down, middle, kernel_size=(3, 3), dilation=dilation, stride=(2, 2), padding=0)
                    self.m=lambda x: self.m1(F.pad(x, (1, 0, 1, 0)))
            elif upsample:
                if upsample[0] % 2 == 1:
                    self.m=self.m=nn.ConvTranspose2d(down, middle, kernel_size=(3, 3), dilation=dilation, stride=(2, 2), padding=0)
                else:
                    self.m1=nn.ConvTranspose2d(down, middle, kernel_size=(3, 3), dilation=dilation, stride=(2, 2), padding=0, output_padding=(1, 1))
                    self.m=lambda x: self.m1(x[:, :, :-1, :-1])
            elif dilation==2:
                self.m=nn.Conv2d(down, middle, kernel_size=(3, 3), dilation=dilation, padding=2)
            elif dilation==4:
                self.m=nn.Conv2d(down, middle, kernel_size=(3, 3), dilation=dilation, padding=4)
            elif dilation==6:
                self.m=nn.Conv2d(down, middle, kernel_size=(3, 3), dilation=dilation, padding=6)
            elif dilation==8:
                self.m=nn.Conv2d(down, middle, kernel_size=(3, 3), dilation=dilation, padding=8)
            else:
                self.m=nn.Conv2d(down, middle, kernel_size=(3, 3), dilation=dilation, padding=1)
        elif mode=='asymmetric':
            self.m1=nn.Conv2d(down, middle, (1, 5), padding=(0, 2))
            self.m2=nn.Conv2d(middle, middle, (5, 1), padding=(2, 0))
            self.m=lambda x: self.m2(self.m1(x))
        else:
            raise ValueError(
                    "Bad mode, expected 'regular' or 'asymmetric' but got: "+str(mode))
        self.up=nn.Conv2d(middle, out_filters, kernel_size=(1, 1), bias=False, padding=0)

    def forward(self, x, i=None):
        p=x
        #Conv path
        sq=self.sq(x).clamp(0)
        if self.sq_dropsamples>0:
            sq[:, torch.randperm(self.middle)[:self.sq_dropsamples]]=0
            sq*=self.middle/(self.middle-self.sq_dropsamples)
        m=self.m(sq).clamp(0)
        u=self.up(m)
        u=self.bn1(u)
        #print('U shape', u.size())
        #print('P shape', p.size())
        if self.downs:
            p, i=self.ppool(p)
        elif self.ups:
            p2=p[:, i.shape[1]:]
            p2=F.interpolate(p2, self.ups)
            u1=u[:, :i.shape[1]]
            u2=u[:, i.shape[1]:]
            #u2=F.interpolate(u2, self.ups)
            u=torch.cat([u1, u2], 1)
            p1=p[:, :i.shape[1]]
            p1=self.ppool(p1, i)
            p=torch.cat([p1, p2], 1)
        if self.in_filters<self.out_filters:
            o1=p+u[:, :self.in_filters, :, :]
            o=torch.cat([o1, u[:, self.in_filters:, :, :]], 1)
        else:
            o = p+u
        if self.dropsamples>0:
            o[:, torch.randperm(self.out_filters)[:self.dropsamples]]=0
            o*=self.out_filters/(self.out_filters-self.dropsamples)
        if self.downs:
            return o, i
        return o
            
class SE_Enet(nn.Module):
    def __init__(self, i_chan, ch_mul, classes, drop=0.02, sqd=0.1):
        super(SE_Enet, self).__init__()
        self.initial=Init_block(i_chan, 14)
        self.conv1=nn.Sequential(Bottleneck(14, ch_mul),
                                 Bottleneck(ch_mul, ch_mul))
        self.conv2_1=Bottleneck(ch_mul, 2*ch_mul, downsample=(50, 50))
        self.conv2_2=Bottleneck(2*ch_mul, 2*ch_mul)
        self.conv3_1=Bottleneck(2*ch_mul, 4*ch_mul, downsample=(25, 25))
        self.conv3_2=nn.Sequential(Bottleneck(4*ch_mul, 4*ch_mul, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=2, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, mode='asymmetric', dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=4, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=6, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, mode='asymmetric', dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=8, dropout=drop, sq_drop=sqd))
        self.conv4=nn.Sequential(Bottleneck(4*ch_mul, 4*ch_mul, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=2, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, mode='asymmetric', dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=4, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=6, dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, mode='asymmetric', dropout=drop, sq_drop=sqd),
                                 Bottleneck(4*ch_mul, 4*ch_mul, dilation=8, dropout=drop, sq_drop=sqd))
        self.conv5_1=Bottleneck(4*ch_mul, 4*ch_mul, upsample=(50, 50))
        self.conv5_2=nn.Sequential(Bottleneck(4*ch_mul, 8*ch_mul),
                                 Bottleneck(8*ch_mul, 8*ch_mul))
        self.conv6_1=Bottleneck(8*ch_mul, 8*ch_mul, upsample=(101, 101))
        self.conv6_2=nn.Sequential(Bottleneck(8*ch_mul, 8*ch_mul), nn.Conv2d(8*ch_mul, classes, (1, 1)))
    def forward(self, x, sigmoid=True):
        initial=self.initial(x)
        conv1=self.conv1(initial)
        conv2_1, i1=self.conv2_1(conv1)
        conv2_2=self.conv2_2(conv2_1)
        conv3_1, i2=self.conv3_1(conv2_2)
        conv3_2=self.conv3_2(conv3_1)
        conv4=self.conv4(conv3_2)
        conv5_1=self.conv5_1(conv4, i2)
        conv5_2=self.conv5_2(conv5_1)
        conv6_1=self.conv6_1(conv5_2, i1)
        output=self.conv6_2(conv6_1)
        if sigmoid:
            output=torch.sigmoid(output)
        return output
def test():
    innoise=torch.rand((10, 1, 101, 101))
    b=SE_Enet(1, 16, 1)
    b=b.cuda()
    innoise=innoise.cuda()
    o=b(innoise[0].reshape(1, 1, 101, 101))
    print(o)
