import torch
import torch.nn as nn
from unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes): # nombre de canaux en entrée et nombre de classe en sortie pour Caravana 3,1 
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64) # 64 filtres => [8, 64, 512, 512] skip connexion, batch_size = 8
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        
        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels= num_classes, kernel_size=1) # pas de padding => w_out = ((512 - 1 (1-1) -1)/stride) +1 = 512, out_channels = 1
    
    def forward(self, x):                           # taille de sortie :
        down_1, p1 = self.down_convolution_1(x)     # - 256 diminue qu'avec maxpool car padding sur conv
        down_2, p2 = self.down_convolution_2(p1)    # - 128
        down_3, p3 = self.down_convolution_3(p2)    # - 64
        down_4, p4 = self.down_convolution_4(p3)    # - 32
        # down => skip connexion, p => pooling pour la suite

        b = self.bottle_neck(p4)                # - 32 juste double conv sans pool

        up_1 = self.up_convolution_1(b, down_4)     # - 64
        up_2 = self.up_convolution_2(up_1, down_3)  # - 128
        up_3 = self.up_convolution_3(up_2, down_2)  # - 256
        up_4 = self.up_convolution_4(up_3, down_1)  # - 512 Taille originale A chaque up on concat les canaux de x1 up et x2 skip connexion

        out = self.out(up_4)                        # - [8, 1, 512, 512] car 1 classe, batch_size = 8
        return out
