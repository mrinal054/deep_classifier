import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EnsembleRes18Enet2(nn.Module):
    def __init__(self, 
                 num_classes,
                 out_channels:list=None, # for instance [1024, 512, 256]. Used in classification head
                 dropout:float=0.3,
                ):
        super(EnsembleRes18Enet2, self).__init__()
        
        # Load pretrained ResNet18 and EfficientNetB2
        self.resnet18 = models.resnet18(pretrained=True)
        self.efficientnetb2 = models.efficientnet_b2(pretrained=True)
        
        # Remove the fully connected layers to extract 2D feature maps
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])  # Feature map (512 x H x W)
        self.efficientnetb2 = nn.Sequential(*list(self.efficientnetb2.children())[:-2])  # Feature map (1408 x H x W)

        # Conv-ReLU-BN block after concatenating feature maps
        self.conv_block = nn.Sequential(
            nn.Conv2d(512 + 1408, out_channels[0], kernel_size=3, padding=1),  # Convolution with 1024 output channels
            nn.ReLU(),
            nn.BatchNorm2d(out_channels[0])
        )
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (out_channels x 1 x 1)
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (out_channels x 1 x 1)
        
        # Fully connected layers for classification
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
        resnet18_features = self.resnet18(x)  # Output: (512 x H_resnet x W_resnet)
        efficientnetb2_features = self.efficientnetb2(x)  # Output: (1408 x H_efficient x W_efficient)
        
        # Ensure both feature maps have the same spatial dimensions
        if resnet18_features.shape[2:] != efficientnetb2_features.shape[2:]:
            # Resize EfficientNetB2's output to match ResNet18's spatial dimensions
            efficientnetb2_features = F.interpolate(efficientnetb2_features, size=resnet18_features.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate the feature maps along the channel dimension
        combined_features = torch.cat((resnet18_features, efficientnetb2_features), dim=1)  # (1920 x H x W)
        
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
    model = EnsembleRes18Enet2(num_classes=num_classes, out_channels=out_channels, dropout=dropout)
    
    # Test with random input (batch_size=4, num_channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(output.shape)  # Should output: torch.Size([4, num_classes])
