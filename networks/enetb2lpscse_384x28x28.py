"""
Author @ Mrinal Kanti Dhar
October 22, 2024
"""

import sys
sys.path.append("/research/m324371/Project/adnexal/utils/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pscse_cab import PscSEWithCAB
from classification_head import ClassificationHead


class EfficientNetB2L_384x28x28(nn.Sequential):
    def __init__(self,
                 pretrain:bool=True,):

        # Load pretrained weights
        if pretrain: model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        else: model = models.efficientnet_v2_l(weights=None)

        super(EfficientNetB2L_384x28x28, self).__init__(
            model.features[0],
            model.features[1],
            model.features[2],
            model.features[3],
            model.features[4][0].block[0],) # using only block 0 (384x28x28)
            

# model = EfficientNetB2L_384x28x28(pretrain=True)
# inp = torch.rand(1,3,224,224)

# out = model(inp)

# print(out.shape)

class EfficientNetB2LPscse_384x28x28(nn.Module):
    def __init__(self,
                 num_classes,
                 out_channels:list=None, # for instance [1024, 512, 256]. Used in classification head
                 pretrain:bool=True,
                 dropout:float=0.3,
                 activation:str='leakyrelu',
                 reduction=16,
                ):
        super(EfficientNetB2LPscse_384x28x28, self).__init__()

        self.clipped_model = EfficientNetB2L_384x28x28(pretrain=pretrain)

        self.pscse_cab_1 = PscSEWithCAB(in_ch=384,
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
    pretrain=True
    dropout=0.3
    activation='leakyrelu'
    reduction=16
    
    model = EfficientNetB2LPscse_384x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction)
    
    out = model(inp)
    
    print(out.shape)

