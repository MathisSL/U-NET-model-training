# Importation des dépendances
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module): # Partie double convolution 3x3 sur le schéma U-Net (flèches bleues) architecture
    def __init__(self, in_channels, out_channels):  # nombre de canaux, nombre de filtre
        super().__init__() 
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1), # padding pour garder la même taille d'image pour la connexion et donc la sortie du réseau
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1), # w_out = (w_in +2 padding - dilatation ((kernel_size - 1)/stride)+1 ici 512 w_in et 512 w_out pareil pour h_out
            nn.ReLU(inplace=True)            
        )

    def forward(self, x): # Applique à l'entrée l'opération séquential
        return self.conv_op(x)
    
class DownSample(nn.Module): # Classe de la partie downsample du modèle => Doubleconv et maxpool
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module): # Classe de la partie upsample => convTranspose et Doubleconv pour upscale (ne pas oublier les connexions résiduelles)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride = 2) # //2 car on diminue les caractéristiques (features) et on augmente la résolution
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): # x1 upsample x2 connexion, pour concaténer il faut qu'ils soient de la même taille
        x1 = self.up(x1) # Upconv
        x = torch.cat([x2, x1], dim=1) # Concaténer skip connexion et le upsampling, dim = 1 => les caractéristique ([batch_size, channel, heigth, width])
        return self.conv(x) # Conv de la concaténation x1,x2
