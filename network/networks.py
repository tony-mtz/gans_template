import torch
import torch.nn as nn
import numpy as np

from utils.net_helper import *

class Gan_Discriminator(nn.Module):
    '''
    expected input [64,64,128,128], [3,3,3,3], [2,2,2,1]
    '''

    def __init__(self, filters, ks, strides):
        super(Gan_Discriminator, self).__init__()
        self.dropout = .4
        self.discr_conv = self.make_discriminator(filters, ks, strides)
        #(128,4,4) is the expected output of conv section
        self.dense = discr_dense(np.prod((128,4,4)))  

    def set_dropout(self, amount):
        self.dropout = amount
        
    def make_discriminator(self, filters, ks, strides):
        layers = []
        pad = 1 #to keep same padding for ks=3
        drop = .4
        for i in range(len(filters)):
            if i ==0:
                layers.append(conv_layer(1, filters[i] ,ks[i], strides[i], pad, self.dropout))
            else:
                layers.append(conv_layer(filters[i-1], filters[i] ,ks[i], strides[i], pad, self.dropout))
                
        return nn.Sequential(*layers)
            
    def forward(self, x):
        discriminator = self.discr_conv(x)
        dense = self.dense(discriminator)
        return dense
            

class Gan_Generator(nn.Module):
    '''
    expected input (64,7,7), [2,2,1,1], [128,64,64,1], [3,3,3,3], [1,1,1,1]
    '''

    def __init__(self, init_dense, upsample, filters, ks, strides):
        super(Gan_Generator, self).__init__()
        self.dropOUt = None
        self.z = 100
        
        self.dense_init = gen_dense(self.z, np.prod(init_dense))   
        self.up =  self.make_generator(upsample, filters, ks, strides )
        
    def make_generator(self, upsample, filters, ks, strides):
        layers = []
        
        for i in range(len(filters)):
            if i ==0: #assume first is 2 upscale                 
                layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(64, filters[i], ks[i], strides[i],padding=1)
                    )
                )
                
            elif upsample[i] == 2:
                layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(filters[i-1], filters[i], ks[i], strides[i], padding=1)
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(filters[i-1], filters[i], ks[i], strides[i], padding=1)
                    )
                )
                
            if i < len(filters) -1:
                layers.append(
                    nn.Sequential(
                            nn.BatchNorm2d(filters[i]),
                            nn.ReLU()
                        )
                )
            else:
                layers.append(nn.Tanh())

        return nn.Sequential(*layers)     
            
    def forward(self, x):
        init = self.dense_init(x)
        reshape = torch.reshape(init, (-1,64,7,7))
        up = self.up(reshape)
        return up
            