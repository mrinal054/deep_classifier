"""
Author @ Mrinal Kanti Dhar
October 21, 2024
"""
import sys
sys.path.append("/research/m324371/Project/adnexal/utils/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pscse_cab import PscSEWithCAB
from classification_head import ClassificationHead

# Modify the model to extract layers up to ReLU-81 (256x28x28)
class ResNet50_Upto_256x28x28(nn.Sequential):
    def __init__(self, pretrain:bool=True):
        if pretrain: resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else: resnet50 = models.resnet50(weights=None)
        
        # Initialize nn.Sequential directly with the desired layers
        super(ResNet50_Upto_256x28x28, self).__init__(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # Output size: [-1, 256, 56, 56]
            resnet50.layer2,  # Output size: [-1, 512, 28, 28]
            nn.Conv2d(512, 256, kernel_size=1),  # Convert channels 512 to 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

class ResNet50Pscse_256x28x28(nn.Module):
    def __init__(self,
                 num_classes,
                 out_channels:list=None, # for instance [1024, 512, 256]. Used in classification head
                 pretrain:bool=True,
                 dropout:float=0.3,
                 activation:str='leakyrelu',
                 reduction=16,
                ):
        super(ResNet50Pscse_256x28x28, self).__init__()

        self.clipped_model = ResNet50_Upto_256x28x28(pretrain=pretrain)

        self.pscse_cab_1 = PscSEWithCAB(in_ch=256,
                                     out_ch=512,
                                     activation=activation,
                                     dropout=dropout,
                                     reduction=reduction,
                                     use_batchnorm=True,
                                     )
        
        self.pscse_cab_2 = PscSEWithCAB(in_ch=512,
                                     out_ch=1024,
                                     activation=activation,
                                     dropout=dropout,
                                     reduction=reduction,
                                     use_batchnorm=True,
                                     )

        self.classification = ClassificationHead(num_classes=num_classes,
                                                 out_channels=out_channels,
                                                 dropout=dropout,
                                                )

    def forward(self, x):
        x = self.clipped_model(x)
        x = self.pscse_cab_1(x)
        x = self.pscse_cab_2(x)
        x = self.classification(x)

        return x
                                                 
                                                 
if __name__ == "__main__":
    inp=torch.rand(1, 3, 224, 224)
    num_classes=2
    out_channels=[1024, 512, 256]
    pretrain = True
    dropout=0.3
    activation='leakyrelu'
    reduction=16
    
    model = ResNet50Pscse_256x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction)
    
    out = model(inp)
    
    print(out.shape)
