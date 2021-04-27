""" Code from """

import torch
from torch import nn
from torch.nn import Sequential, Linear, Dropout, ReLU, Sigmoid, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, LeakyReLU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 3, 3)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = Sequential( 
            Conv2d(1, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x48x48
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x24x24
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x12x12
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x6x6
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x3x3
            Dropout(.25),
            Flatten(),
        )

        self.fc_audio = Sequential(
            Linear(1152, 1152, bias=False),
            Dropout(0.25),
        )

    def forward(self, x):
        z = self.audio_encoder(x)
        z_d = self.fc_audio(z)
        return z, z_d


class TagEncoder(nn.Module):
    def __init__(self):
        super(TagEncoder, self).__init__()

        self.tag_encoder = Sequential(
            Linear(1000, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(.25),
            Linear(512, 1152),
            BatchNorm1d(1152),
            ReLU(),
            Dropout(.25),
        )

        self.fc_tag = Sequential(
            Linear(1152, 1152, bias=False),
            Dropout(.25),
        )

    def forward(self, tags):
        z = self.tag_encoder(tags)
        z_d = self.fc_tag(z)
        return z, z_d
