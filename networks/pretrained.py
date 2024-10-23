#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def pretrained_models(name, num_classes:int=None, in_chs:int=None):
    """
    Args:
        - name: Name of the pretrained model
        - num_classes: No. of classes
        - in_chs: No. of input channels

    Return:
        - Pretrained model
    """
    pretrained_names = ['resnet18', 'efficientnet_v2_s', 'mobilenet_v3_large', 'vgg19_bn', 'maxvit_t', 'convnext']

    if name == 'resnet18':
        base_model = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept a single channel
        if in_chs == 1:
            base_model.conv1 = nn.Conv2d(1, base_model.conv1.out_channels, kernel_size=base_model.conv1.kernel_size, 
                                stride=base_model.conv1.stride, padding=base_model.conv1.padding, bias=False)
        
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.conv1.weight = torch.nn.Parameter(base_model.conv1.weight.mean(dim=1, keepdim=True))
        
        # Modify the last layer to adjust no. of classes
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes, bias=True) # for ResNet18

    elif name == 'efficientnet_v2_s':
        base_model = models.efficientnet_v2_s(pretrained=True)

        # Modify the first layer to adjust the input channels
        if in_chs == 1:
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, kernel_size=base_model.features[0][0].kernel_size, 
                                                  stride=base_model.features[0][0].stride, padding=base_model.features[0][0].padding, bias=False)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))
        
        # Modify the last layer to adjust no. of classes
        base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, num_classes, bias=True)

    elif name == 'mobilenet_v3_large':

        base_model = models.mobilenet_v3_large(pretrained=True)

        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, 
                                                  kernel_size=base_model.features[0][0].kernel_size, stride=base_model.features[0][0].stride, padding=base_model.features[0][0].padding, bias=False)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))
        
        # Modify the last layer to adjust no. of classes
        base_model.classifier[3] = nn.Linear(in_features=base_model.classifier[3].in_features, out_features=num_classes, bias=True)

    elif name == 'vgg19_bn':
        base_model = models.vgg19_bn(pretrained=True)

        if in_chs == 1:
            base_model.features[0] = nn.Conv2d(1, base_model.features[0].out_channels, kernel_size=base_model.features[0].kernel_size, 
                                               stride=base_model.features[0].stride, padding=base_model.features[0].padding)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0].weight = torch.nn.Parameter(base_model.features[0].weight.mean(dim=1, keepdim=True))
        
        # Modify the last layer to adjust no. of classes
        base_model.classifier[6] = nn.Linear(base_model.classifier[6].in_features, num_classes, bias=True)

    elif name == 'maxvit_t':
        base_model = models.maxvit_t(pretrained=True)

        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.stem[0][0] = nn.Conv2d(1, base_model.stem[0][0].out_channels, kernel_size=base_model.stem[0][0].kernel_size, 
                                              stride=base_model.stem[0][0].stride, padding=base_model.stem[0][0].padding, bias=False)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.stem[0][0].weight = torch.nn.Parameter(base_model.stem[0][0].weight.mean(dim=1, keepdim=True))
        
        # Modify the last layer to adjust no. of classes
        base_model.classifier[5] = nn.Linear(base_model.classifier[5].in_features, num_classes, bias=False)
        
    elif name == 'convnext':
        # ConvNeXt
        base_model = models.convnext_base(pretrained=True)

        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, kernel_size=base_model.features[0][0].kernel_size, stride=base_model.features[0][0].stride)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))
        
        base_model.classifier[2] = nn.Linear(in_features=base_model.classifier[2].in_features, out_features=num_classes, bias=True) 
    
    else:
        print(f"{name} is not in pretrained names: {pretrained_names}")

    return base_model
        
    
if __name__ == "__main__":
    input = torch.rand(1,3,224,224)
    name = "resnet18"
    n_classes = 2
    in_chs = input.shape[1]
    model = pretrained_models(name, n_classes, in_chs)

