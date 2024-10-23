#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This architecture uses ResNet18. It has three ResNet18 models. 

The first ResNet18 takes the image, the second ResNet18 takes the fluid component, 
and the third ResNet18 takes the solid component. 

Then, their feature maps are concatenated. A classification head is then added.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EnsembleRes18SepIn(nn.Module):
    def __init__(self, 
                 num_classes,
                 out_channels:list=None, # for instance [1024, 512, 256]. Used in classification head
                 dropout:float=0.3,
                ):
        super(EnsembleRes18SepIn, self).__init__()
        
        # Load pretrained ResNet18 and EfficientNetB2
        self.model1 = models.resnet18(pretrained=True)
        self.model2 = models.resnet18(pretrained=True)
        self.model3 = models.resnet18(pretrained=True)
        
        # Remove the fully connected layers to extract 2D feature maps
        self.model1 = nn.Sequential(*list(self.model1.children())[:-2])  # Feature map (512 x H x W)
        self.model2 = nn.Sequential(*list(self.model2.children())[:-2])  # Feature map (512 x H x W)
        self.model3 = nn.Sequential(*list(self.model3.children())[:-2])  # Feature map (512 x H x W)

        # Conv-ReLU-BN block after concatenating feature maps
        self.conv_block = nn.Sequential(
            nn.Conv2d(512 * 3, out_channels[0], kernel_size=3, padding=1),  # Convolution with 1024 output channels
            nn.ReLU(),
            nn.BatchNorm2d(out_channels[0])
        )
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (out_channels x 1 x 1)
        
        # Classification head
        layers = []
        
        # Create fully connected layers based on channel_list
        for i in range(len(out_channels) - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(out_channels[i], out_channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer for classification
        layers.append(nn.Linear(out_channels[-1], num_classes))

        self.fc = nn.Sequential(*layers)
    
        
    def forward(self, x):
        # Extract 2D feature maps from both models
        image = x[:,0:1,:,:]
        fluid = x[:,1:2,:,:]
        solid = x[:,2:3,:,:]
        
        model1_features = self.model1(x)  # Output: (512 x H_resnet x W_resnet)
        model2_features = self.model2(x)  # Output: (512 x H_resnet x W_resnet)
        model3_features = self.model3(x)  # Output: (512 x H_resnet x W_resnet)
        
        # Concatenate the feature maps along the channel dimension
        combined_features = torch.cat((model1_features, model2_features, model3_features), dim=1)  # (512*3 x H x W)
        
        # Apply the convolutional block
        conv_out = self.conv_block(combined_features)  # Output: (out_channels x H x W)
        
        # Global Average Pooling
        pooled_out = self.avg_pool(conv_out)  # Output: (out_channels x 1 x 1)
        
        # Flatten the pooled output
        flattened = pooled_out.view(pooled_out.size(0), -1)  # Output: (out_channels,)
        
        # Classification using fully connected layers
        output = self.fc(flattened)  # Output: (num_classes)
        
        return output

if __name__ == "__main__":
    # Example usage
    num_classes = 10  # Set the number of output classes 
    out_channels = [1024, 512, 256]
    dropout = 0.3
    model = EnsembleRes18SepIn(num_classes=num_classes, out_channels=out_channels, dropout=dropout)
    
    # Test with random input (batch_size=4, num_channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(output.shape)  # Should output: torch.Size([4, num_classes])


# In[ ]:




