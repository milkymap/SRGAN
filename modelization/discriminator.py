import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, nb_channels=64, nb_blocks=8, nb_neurons_on_dense=1024):
        super(Discriminator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, nb_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.body = []
        channels_array = np.repeat([ nb_channels * 2 ** idx for idx in range(nb_blocks // 2) ], 2)
        channels_array = list(zip(channels_array[:-1], channels_array[1:]))

        for idx, (m, n) in enumerate(channels_array):
            self.body.append(
                nn.Sequential(
                    nn.Conv2d(m, n, kernel_size=3, padding=1, stride=1 + (idx % 2 == 0)), 
                    nn.BatchNorm2d(n), 
                    nn.LeakyReLU(0.2)
                )
            )
        self.body = nn.Sequential(*self.body)
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nb_channels * nb_blocks, nb_neurons_on_dense, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nb_neurons_on_dense, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.tail(self.body(self.head(X))).view(-1)

if __name__ == '__main__':
    D = Discriminator()
    X = th.randn((2, 3, 256, 256))
    Y = D(X) 
    print(Y)
    