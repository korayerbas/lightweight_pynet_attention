# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import torch.nn as nn
import torch

class PyNET_att(nn.Module):

    def __init__(self, level, instance_norm=True, instance_norm_level_1=False):
        super(PyNET_att, self).__init__()

        self.level = level

        self.conv_l1_d1 = ConvMultiBlock(4, 32, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv_l2_d1 = ConvMultiBlock(32, 64, 3, instance_norm=instance_norm)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv_l3_d1 = ConvMultiBlock(64, 128, 3, instance_norm=instance_norm)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # -------------------------------------

        self.conv_l4_d1 = ConvMultiBlock(128, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d2 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d3 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)
        self.conv_l4_d4 = ConvMultiBlock(256, 256, 3, instance_norm=instance_norm)

        self.conv_t3b = UpsampleConvLayer(256, 128, 3)

        self.conv_l4_out = ConvLayer(256, 3, kernel_size=3, stride=1, relu=False)
        self.output_l4 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l3_d3 = ConvMultiBlock(256, 128, 3, instance_norm=instance_norm)

        self.conv_t2b = UpsampleConvLayer(128, 64, 3)

        self.conv_l3_out = ConvLayer(128, 3, kernel_size=3, stride=1, relu=False)
        self.output_l3 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l2_d3 = ConvMultiBlock(128, 64, 3, instance_norm=instance_norm)
        
        self.conv_t1b = UpsampleConvLayer(64, 32, 3)
        self.conv_l2_out = ConvLayer(64, 3, kernel_size=3, stride=1, relu=False)
        self.output_l2 = nn.Sigmoid()

        # -------------------------------------
        self.conv_l1_d3 = ConvMultiBlock(64, 32, 3, instance_norm=instance_norm)
        
        self.conv_t0b = UpsampleConvLayer(32, 16, 3)
        self.conv_l1_out = ConvLayer(32, 3, kernel_size=3, stride=1, relu=False)
        self.output_l1 = nn.Sigmoid()

        # -------------------------------------

        self.conv_l0_d1 = ConvLayer(16, 32, kernel_size=3, stride=1, relu=True)               
        self.out_att = att_module(input_channels = 32, ratio =2, kernel_size = 3)
        
        self.conv_l0_d2 = ConvLayer(32, 16, kernel_size=3, stride=1, relu=True)
        self.conv_l0_d3 = ConvLayer(16, 16, kernel_size=1, stride=1, relu=True)
        
        self.conv_l0_d4 = ConvLayer(32, 3, kernel_size=1, stride=1, relu=False)
        self.output_l0 = nn.Tanh()
        
    def level_4(self, pool3):
        
        conv_l4_d1 = self.conv_l4_d1(pool3)
        conv_l4_d2 = self.conv_l4_d2(conv_l4_d1)
        conv_l4_d3 = self.conv_l4_d3(conv_l4_d2)
        conv_l4_d4 = self.conv_l4_d4(conv_l4_d3)
        conv_t3b = self.conv_t3b(conv_l4_d4)
        conv_l4_out = self.conv_l4_out(conv_l4_d4)
        output_l4 = self.output_l4(conv_l4_out)

        return output_l4, conv_t3b

    def level_3(self, conv_l3_d1, conv_t3b):

        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t3b], 1)
        conv_l3_d3 = self.conv_l3_d3(conv_l3_d2)
        conv_t2b = self.conv_t2b(conv_l3_d3)
        conv_l3_out = self.conv_l3_out(conv_l3_d3)
        output_l3 = self.output_l3(conv_l3_out)
        
        return output_l3, conv_t2b

    def level_2(self, conv_l2_d1, conv_t2b):

        conv_l2_d2 = torch.cat([conv_l2_d1, conv_t2b], 1)
        conv_l2_d3 = self.conv_l2_d3(conv_l2_d2)
        conv_t1b = self.conv_t1b(conv_l2_d3)
        conv_l2_out = self.conv_l2_out(conv_l2_d3)
        output_l2 = self.output_l2(conv_l2_out)

        return output_l2, conv_t1b

    def level_1(self, conv_l1_d1, conv_t1b):

        conv_l1_d2 = torch.cat([conv_l1_d1, conv_t1b], 1)
        conv_l1_d3 = self.conv_l1_d3(conv_l1_d2)
        conv_t0b = self.conv_t0b(conv_l1_d3)
        conv_l1_out = self.conv_l1_out(conv_l1_d3)
        output_l1 = self.output_l1(conv_l1_out)

        return output_l1, conv_t0b

    def level_0(self, conv_t0b):
        print('conv_t0b shape: ',conv_t0b.shape)
        conv_l0_d1 = self.conv_l0_d1(conv_t0b)       
        print('conv_l0_d1 shape: ',conv_l0_d1.shape)
        att_l0 = self.out_att (conv_l0_d1) 
        print('att_l0 shape: ',att_l0.shape)
        z1_l0 = conv_l0_d1 + att_l0
        print('z1_l0 shape: ',z1_l0.shape)
        conv_l0_d2 = self.conv_l0_d2(z1_l0)    
        print('conv_l0_d2 shape: ',conv_l0_d2.shape)
        conv_l0_d3 = self.conv_l0_d3(conv_l0_d2)
        print('conv_l0_d3 shape: ',conv_l0_d3.shape)
        cat1_l0 = torch.cat([conv_t0b, conv_l0_d3], 1)
        print('cat1_l0 shape: ',cat1_l0.shape)
        conv_l0_d4 = self.conv_l0_d4(cat1_l0)
        print('conv_l0_d4 shape: ',conv_l0_d4.shape)
        output_l0 = self.output_l0(conv_l0_d4)
        print('output_l0 shape: ',output_l0.shape)
        
        return output_l0

    def forward(self, x):

        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)

        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)

        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)

        output_l4, conv_t3b =self.level_4(pool3)

        if self.level < 4:
            output_l3, conv_t2b = self.level_3(conv_l3_d1, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1b = self.level_2(conv_l2_d1, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0b = self.level_1(conv_l1_d1, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(conv_t0b)

        if self.level == 0:
            enhanced = output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4

        return enhanced
 
class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size

        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(in_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)
            self.conv_5b = ConvLayer(out_channels, out_channels, kernel_size=5, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(in_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)
            self.conv_7b = ConvLayer(out_channels, out_channels, kernel_size=7, stride=1, instance_norm=instance_norm)

        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(in_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)
            self.conv_9b = ConvLayer(out_channels, out_channels, kernel_size=9, stride=1, instance_norm=instance_norm)

    def forward(self, x):

        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)

        if self.max_conv_size >= 5:
            out_5 = self.conv_5a(x)
            out_5 = self.conv_5b(out_5)
            output_tensor = torch.cat([output_tensor, out_5], 1)

        if self.max_conv_size >= 7:
            out_7 = self.conv_7a(x)
            out_7 = self.conv_7b(out_7)
            output_tensor = torch.cat([output_tensor, out_7], 1)

        if self.max_conv_size >= 9:
            out_9 = self.conv_9a(x)
            out_9 = self.conv_9b(out_9)
            output_tensor = torch.cat([output_tensor, out_9], 1)

        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out
    
class depthwise_conv(nn.Module):
    def __init__(self, input_channels, kernel_size):
        super(depthwise_conv, self).__init__()
        
        reflection_padding = 2 * (kernel_size//2)
        
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.dw_conv =  nn.Sequential(nn.Conv2d(input_channels, input_channels, kernel_size, dilation=2, groups=input_channels),nn.ReLU())
        self.point_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        
        y = self.reflection_pad(x)
        #print('y_depthwise shape: ',y.shape)
        conv1 = self.dw_conv(y)
        #print('depthwise_conv shape: ',conv1.shape)
        conv2 = self.point_conv(conv1)
        #print('point_conv shape: ',conv2.shape)
        out = self.sigmoid(conv2)
        return out

class SpatialAttention2(nn.Module):
    def __init__(self, input_channels, kernel_size):
        
        super(SpatialAttention2, self).__init__()
        
        self.dw = depthwise_conv(input_channels, kernel_size)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        
        #print('x_sa2 shape: ',x.shape)
        z= self.dw(x)
        #print('z_sa2 shape: ',z.shape)
        z1= self.sigmoid(z)
        out = x * z1
        #print('out_sa2 shape: ',out.shape)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, kernel_size = 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // ratio, in_channels, kernel_size= 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        #print('x_ca_avg shape: ',avg_out.shape)
        max_out = self.fc(self.max_pool(x))
        #print('x_ca_max shape: ',max_out.shape)
        out = self.sigmoid(avg_out * max_out)
        #print('ca_out1 shape: ',out.shape)
        out_ca = out * x
        #print('ca_out shape: ',out_ca.shape)
        return out_ca

class att_module(nn.Module):
    
    def __init__(self, input_channels, ratio, kernel_size, instance_norm=False):
        super(att_module, self).__init__()
        
        self.conv1 = ConvLayer(in_channels= input_channels, out_channels=input_channels*2, kernel_size=3, stride=1, relu=True)
        self.conv2 = ConvLayer(in_channels=input_channels*2, out_channels=input_channels, kernel_size=1, relu=True, stride =1)
        
        self.ca = ChannelAttention(input_channels, ratio)
        #self.sa = SpatialAttention(in_channels, kernel_size=5, dilation=2)
        self.sa = SpatialAttention2(input_channels, kernel_size)
        self.conv3 = ConvLayer(input_channels*2, input_channels, kernel_size=1, stride= 1, relu=True)
    
    def forward(self, x):
       
       conv1 = self.conv1(x)
       print('conv1_att shape: ',conv1.shape)
       conv2 = self.conv2(conv1)
       print('conv2_att shape: ',conv2.shape)
              
       z1 = self.ca(conv2)
       print('z1_att shape: ',z1.shape)
       z2 = self.sa(conv2)
       print('z2_att shape: ',z2.shape)
       out = self.conv3(torch.cat([z1, z2], 1))
       print('out_att shape: ',out.shape)
       return out
