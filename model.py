"""
In this file, we define the UNet architecture. 

The sole modification done is to add padding = 1 in order to not interpolate the output to get the right size for the segmentation map.

"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBloc(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bloc(x)

class Upsampling(nn.Module):
    """ In forward, x1 is the upsampled flow while x2 is the residual flow from the left side of UNet 
    that needs to be cropped"""
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #diffY = x2.size()[-2] - x1.size()[-2]
        #diffX = x2.size()[-1] - x1.size()[-1]

        ## In batch mode
        if len(x2.size()) == 4: 
        #    x2 = x2[:,:,diffY//2 : diffY//2 + x1.size()[-2], diffX//2 : diffX//2 + x1.size()[-1]]
            x = torch.cat([x1, x2], dim=1)

        elif len(x2.size()) == 3: # Not in batch mode
        #    x2 = x2[:,diffY//2 : diffY//2 + x1.size()[-2], diffX//2 : diffX//2 + x1.size()[-1]] 
            x = torch.cat([x1, x2], dim=0)
        return x


class UNet(nn.Module):
    def __init__(self):
        # Left side
        super().__init__()

        self.bloc1 = ConvBloc(3,64)
        self.bloc2 = ConvBloc(64,128)
        self.bloc3 = ConvBloc(128,256)
        self.bloc4 = ConvBloc(256,512)

        #middle
        self.bloc5 = ConvBloc(512,1024)
        self.Upbloc1 = Upsampling(1024)

        #right side
        self.bloc6 = ConvBloc(1024,512)
        self.Upbloc2 = Upsampling(512)

        self.bloc7 = ConvBloc(512,256)
        self.Upbloc3 = Upsampling(256)
        
        self.bloc8 = ConvBloc(256,128)
        self.Upbloc4 = Upsampling(128)

        self.bloc9 = ConvBloc(128,64)

        self.ra = nn.MaxPool2d(kernel_size=2)

        # Last layer
        self.conv_f = nn.Conv2d(64,1,kernel_size = 1)
        self.act_f = nn.Sigmoid()

    def forward(self,x):

        # A list containing the skip connections for every floor is defined
        skip_connections = []

        # Left side 
        x = self.bloc1(x)
        skip_connections.append(x)

        x = self.bloc2(self.ra(x))
        skip_connections.append(x)

        x = self.bloc3(self.ra(x))
        skip_connections.append(x)

        x = self.bloc4(self.ra(x))
        skip_connections.append(x)

        # Middle part
        x = self.Upbloc1(self.bloc5(self.ra(skip_connections[-1])),skip_connections[-1])

        #Right side
        x = self.Upbloc2(self.bloc6(x),skip_connections[-2])

        x = self.Upbloc3(self.bloc7(x),skip_connections[-3])

        x = self.Upbloc4(self.bloc8(x),skip_connections[-4])

        #return F.interpolate(self.act_f(self.conv_f(self.bloc9(x))),(256,256))
        return self.conv_f(self.bloc9(x))
