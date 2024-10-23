import torch.nn as nn
from pscse import PscSE2D
from activations import activation_function

class PscSEWithCAB(nn.Module):
    """Performs PscSE followed by Conv2d-Activation-Dropout-BN"""
    def __init__(self, 
                 in_ch,
                 out_ch,
                 activation:str='leakyrelu',
                 dropout:float=0.3,
                 reduction:int=16,
                 use_batchnorm:bool=True):
        
        super(PscSEWithCAB, self).__init__()
       
        # PscSE module
        self.pscse = PscSE2D(channels=in_ch, reduction=reduction, use_batchnorm=use_batchnorm)
        
        # Main convolution with stride=2 (downsampling)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Activation function
        self.activation = activation_function(activation)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_ch)
        
        # Downsample layer to match dimensions
        self.downsample = self.get_downsample_layer(in_ch, out_ch, stride=2)

    def get_downsample_layer(self, in_ch, out_ch, stride):
        """Creates a downsample layer to match dimensions if needed."""
        if stride != 1 or in_ch != out_ch:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        return None

    def forward(self, x):
        identity = x  # Save the input for the residual connection

        # Main path
        out = self.pscse(x)
        out = self.conv(out)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.bn(out)

        # Apply downsample to identity if necessary
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Add the residual connection (identity)
        out += identity
        
        # Apply final activation
        out = self.activation(out)

        return out

if __name__ == "__main__":
    
    # Test the model
    x = torch.rand(1, 128, 224, 224)  # Input tensor
    
    model = PscSEWithCAB(128, 256)  # Create the model with input channels 128, output channels 256
    
    y = model(x)  # Forward pass
    print(y.shape)  # Check output shape
