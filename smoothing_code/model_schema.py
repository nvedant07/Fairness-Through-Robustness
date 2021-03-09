#!/usr/bin/env python
# coding: utf-8

# ## Custom model implementations

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# #### Some rules:
# 1. Each model's final layer would be called fc3
# 2. For CNN based models, the model will typically have some convolution part and some fully connected part, each model class should define self.model_conv and self.model_linear which will define the two parts respectively
# 3. Doing these 2 simple things wouldn't break rest of the code

# In[ ]:


class DeepCNN(nn.Module):
    """
    Taken from http://torch.ch/blog/2015/07/30/cifar.html 
    """
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        layers_conv = []
        layers_conv.extend(self.ConvBNReLU(3, 64, dropout=0.3))
        layers_conv.extend(self.ConvBNReLU(64, 64))
        layers_conv.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        layers_conv.extend(self.ConvBNReLU(64, 128, dropout=0.4))
        layers_conv.extend(self.ConvBNReLU(128, 128))
        layers_conv.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        layers_conv.extend(self.ConvBNReLU(128, 256, dropout=0.4))
        layers_conv.extend(self.ConvBNReLU(256, 256, dropout=0.4))
        layers_conv.extend(self.ConvBNReLU(256, 256))
        layers_conv.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        layers_conv.extend(self.ConvBNReLU(256, 512, dropout=0.4))
        layers_conv.extend(self.ConvBNReLU(512, 512, dropout=0.4))
        layers_conv.extend(self.ConvBNReLU(512, 512))
        layers_conv.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        layers_conv.extend(self.ConvBNReLU(512, 512, dropout=0.4))
        layers_conv.extend(self.ConvBNReLU(512, 512, dropout=0.4))
        layers_conv.extend(self.ConvBNReLU(512, 512))
        layers_conv.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        self.model_conv = nn.Sequential(*layers_conv)
        layers_linear = [nn.Dropout(0.5),
                        nn.Linear(512,512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.5)]
        self.model_linear = nn.Sequential(*layers_linear)
        self.fc3 = nn.Linear(512, num_classes) # To be consistent with vanilla CNN

    def ConvBNReLU(self, nInput, nOutput, dropout=None):
        return [nn.Conv2d(nInput, nOutput, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
                nn.BatchNorm2d(nOutput, eps=1e-03), 
                nn.ReLU(),
                nn.Dropout(dropout)
                ] if dropout is not None else\
                [nn.Conv2d(nInput, nOutput, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
                nn.BatchNorm2d(nOutput, eps=1e-03), 
                nn.ReLU()
                ]

    def forward(self, x):
        x = self.model_conv(x)
        x = x.view(-1, 512)
        x = self.model_linear(x)
        x = self.fc3(x)
        return x
