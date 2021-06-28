import torch as th 
import torch.nn as nn 

import numpy as np 

class RBlock(nn.Module):
    def __init__(self, nb_channels):
        super(RBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_channels), 
            nn.PReLU(), 
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_channels)
        )

    def forward(self, X):
        return X + self.body(X)

class SBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(SBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * up_scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=up_scale),
            nn.PReLU()
        )
    
    def forward(self, X):
        return self.body(X)

class Generator(nn.Module):
    def __init__(self, nb_blocks, nb_channels=64, scale_factor=4):
        super(Generator, self).__init__()
        self.upsample_block_num = int(np.log2(scale_factor))
        self.nb_blocks = nb_blocks
        self.nb_channels = nb_channels  
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nb_channels, kernel_size=9, padding=4), 
            nn.PReLU()
        ) 
        self.body = nn.Sequential(*[ RBlock(nb_channels) for _ in range(self.nb_blocks) ])
        self.botl = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_channels)
        )
        self.tail = nn.Sequential(
            *[ SBlock(nb_channels, 2) for _ in range(self.upsample_block_num) ]
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=3, kernel_size=9, padding=4), 
            nn.Tanh()
        )

    def forward(self, X0):
        X1 = self.head(X0)
        X2 = self.body(X1)
        X3 = self.botl(X2) 
        X4 = self.tail(X1 + X3)
        X5 = self.conv(X4)
        return X5 

if __name__ == '__main__':
    G = Generator(nb_blocks=8)
    X = th.randn((2, 3, 64, 64))

    print(G)
    print(G(X).shape)