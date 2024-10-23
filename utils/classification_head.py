import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """Classification head"""
    def __init__(self,
                 num_classes,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 dropout: float = 0.3):
        super(ClassificationHead, self).__init__()

        if out_channels is None:
            raise ValueError("out_channels cannot be None. Provide a list of output channels.")

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (out_channels x 1 x 1)
        
        # Classification head
        layers = []
        
        # Create fully connected layers based on out_channels
        for i in range(len(out_channels) - 1):
            if dropout is not None: layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(out_channels[i], out_channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer for classification
        layers.append(nn.Linear(out_channels[-1], num_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # Global average pooling
        x = self.avg_pool(x)

        # Flatten the tensor to [batch_size, out_channels]
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # Test the classification head
    model = ClassificationHead(num_classes=10, out_channels=[1024, 512, 256])
    input_tensor = torch.randn(1, 1024, 7, 7)  # Example input
    output = model(input_tensor)
    
    print('Output shape:', output.shape)  # torch.Size([1, 10])
