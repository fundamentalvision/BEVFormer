# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from AdelaiDet
#   https://github.com/aim-uofa/AdelaiDet/
import logging

import torch
from torch import nn

LOG = logging.getLogger(__name__)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Offset(nn.Module):
    def __init__(self, init_value=0.):
        super(Offset, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input + self.bias


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result
