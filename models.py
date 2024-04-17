import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class AEConv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,8,4,1,0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,16,4,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8,3,5,1,1),
        )
        

    def forward(self, features):
        encoded  = self.encoder(features)
        # return encoded
        return self.decoder(encoded)
    #     return self.decoder(encoded) 



class AEConv2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,8,4,1,0),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3,2,1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3,2,1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3,2,1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,3,5,1,1),
        )
        

    def forward(self, features):
        encoded  = self.encoder(features)
        return self.decoder(encoded)

    

import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),

            
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        # down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(
                # output = s * (n-1) + k- 2*p
                nn.ConvTranspose2d(
                    feature*2, feature,kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2,feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)