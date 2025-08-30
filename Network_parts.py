import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

#Networks:
UN_encoder=config['U-net']['encoder']
UN_bn=config['U-net']['bottleneck']
UN_decoder=config['U-net']['decoder']

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.block(x)

def copy_and_crop(down_layer,up_layer):
    b,ch,h,w=up_layer.shape
    crop=T.CenterCrop((h,w))(down_layer)
    return crop

class UNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UNet,self).__init__()

        self.encoder=nn.ModuleList([
            DoubleConv(in_channels,UN_encoder['E1']),
            DoubleConv(UN_encoder['E1'],UN_encoder['E2']),
            DoubleConv(UN_encoder['E2'],UN_encoder['E3']),
            DoubleConv(UN_encoder['E3'],UN_encoder['E4'])
        ])
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.bottle_neck=DoubleConv(UN_encoder['E4'],UN_bn)

        self.up_samples=nn.ModuleList([
            nn.ConvTranspose2d(UN_bn,UN_encoder['E4'],kernel_size=2,stride=2),
            nn.ConvTranspose2d(UN_encoder['E4'],UN_encoder['E3'],kernel_size=2,stride=2),
            nn.ConvTranspose2d(UN_encoder['E3'],UN_encoder['E2'],kernel_size=2,stride=2),
            nn.ConvTranspose2d(UN_encoder['E2'],UN_encoder['E1'],kernel_size=2,stride=2)
        ])

        self.decoder=nn.ModuleList([
            DoubleConv(UN_bn,UN_decoder['D1']),
            DoubleConv(UN_decoder['D1'],UN_decoder['D2']),
            DoubleConv(UN_decoder['D2'],UN_decoder['D3']),
            DoubleConv(UN_decoder['D3'],UN_decoder['D4'])
        ])

        self.final_layer=nn.Conv2d(UN_decoder['D4'],out_channels,1,1)
    def forward(self,x):
        skip_connections=[]

        for layer in self.encoder:
            x=layer(x)
            skip_connections.append(x)
            x=self.pool(x)
        
        x=self.bottle_neck(x)

        for ind,layer in enumerate(self.decoder):
            x=self.up_samples[ind](x)
            y=copy_and_crop(skip_connections.pop(),x)
            x=layer(torch.cat([y,x],dim=1))

        x=self.final_layer(x)

        return x