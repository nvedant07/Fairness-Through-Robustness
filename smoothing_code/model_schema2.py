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


class MLP1(nn.Module):
    def __init__(self, inputs, num_classes):
        super(MLP1, self).__init__()
        self.n_in = inputs
        self.fc3 = nn.Linear(self.n_in,num_classes)
        
    def forward(self, x):
        x = x.reshape(-1,self.n_in)
        out = self.fc3(x)
        return out

class AdienceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AdienceClassifier, self).__init__()
        layers_conv = [nn.Conv2d(3, 96, kernel_size=7, stride=1),
                       nn.BatchNorm2d(96, eps=1e-03),
                       nn.ReLU(),
#                        nn.Dropout(0.3),
                       nn.MaxPool2d(kernel_size=3, stride=2),
                       nn.Conv2d(96, 256, kernel_size=5, stride=1),
                       nn.BatchNorm2d(256, eps=1e-03),
                       nn.ReLU(),
#                        nn.Dropout(0.3),
                       nn.MaxPool2d(kernel_size=3, stride=2),
                       nn.Conv2d(256, 384, kernel_size=3, stride=1),
                       nn.BatchNorm2d(384, eps=1e-03),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2),
                      ]

        self.model_conv = nn.Sequential(*layers_conv)
        layers_linear = [nn.Linear(221184, 512), # This looks baaaddd
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(512, 512), # This looks baaaddd
                         nn.ReLU(),
                         nn.Dropout(0.5)]
        self.model_linear = nn.Sequential(*layers_linear)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = self.model_conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.model_linear(out)
        out = self.fc3(out)
        return out

class UTKClassifier(nn.Module):
    def __init__(self, num_classes):
        super(UTKClassifier, self).__init__()
        layers_conv = [nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=2),
                       nn.ReLU(),
                       nn.Dropout(0.3),
                       nn.MaxPool2d(kernel_size=2, stride=2),
                       nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                       nn.ReLU(),
                       nn.Dropout(0.3),
                       nn.MaxPool2d(kernel_size=2, stride=2)]

        self.model_conv = nn.Sequential(*layers_conv)
        layers_linear = [nn.Linear(153664, 256), # This looks baaaddd
                         nn.ReLU(),
                         nn.Dropout(0.5)]
        self.model_linear = nn.Sequential(*layers_linear)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        out = self.model_conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.model_linear(out)
        out = self.fc3(out)
        return out
    
    
### Taken from: https://github.com/aaron-xichen/pytorch-playground/blob/master/cifar/model.py

class DeepCNNCifar100(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(DeepCNNCifar100, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.fc3 = nn.Linear(n_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)