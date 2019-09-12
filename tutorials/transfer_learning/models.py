from __future__ import print_function
import numpy as np
import scipy.io as lmd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Swish(nn.Module):
    """Swish Function: 
    Applies the element-wise function :math:`f(x) = x / ( 1 + exp(-x))`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = Swish()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Net(nn.Module):
    def __init__(self, dim=1604, droprate=0.5):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(inplace=True),
#             Swish(),
            nn.Dropout(droprate),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(inplace=True),
#             Swish(),
            nn.Dropout(droprate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
#             Swish(),
            nn.Dropout(droprate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
#             Swish(),
            nn.Dropout(droprate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
#             Swish(),
            nn.Linear(128,1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



