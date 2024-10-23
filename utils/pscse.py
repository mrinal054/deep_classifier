import torch
import torch.nn as nn
from torchsummary import summary

""" Reference:
    Base structure is taken from "https://github.com/qubvel-org/segmentation_models.pytorch"
"""

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None
    
class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU()

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        
#%% Modified SCSE module supporing different strategies
class SCSEModule0(nn.Module):
    def __init__(self, in_channels, reduction=16, strategy='addition'):
        super().__init__()
        self.strategy = strategy
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            # nn.SiLU(inplace=True), 
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        self.if_concat = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):

        xc = x * self.cSE(x) # cse attention
        xs = x * self.sSE(x) # sse attention
        
        if self.strategy == 'addition':
            x = xc + xs
        
        elif self.strategy == 'maxout':
            x = torch.maximum(xc, xs)
        
        elif self.strategy == 'concat':
            x = torch.cat([xc, xs], dim=1)
            x = self.if_concat(x)
        
        elif self.strategy == 'multiplication':
            x = xc * xs
        
        elif self.strategy == 'average':
            x = torch.stack((xc, xs), dim=0)
            x = torch.mean(x, dim=0)
        
        elif self.strategy == 'all-average':
            x1 = xc + xs # addition
            x2 = torch.maximum(xc, xs) # multiplication
            x3 = self.if_concat(torch.cat([xc, xs], dim=1)) # concatenation
            x4 = xc * xs # multiplication
            
            # Take average of above mentioned strategies
            x = torch.stack((x1, x2, x3, x4), dim=0)
            x = torch.mean(x, dim=0)
            
        else:
            raise ValueError("Wrong keyword for attention strategy. Choose from [addition, maxout, concat, multiplication, average, all-average]")
        
        return x
    
#%% Create P-scSE2D class
class PscSE2D(nn.Module):
    def __init__(self, 
                 channels, # out_channels will be equal to in_channels
                 reduction, 
                 use_batchnorm=True):
        
        super(PscSE2D, self).__init__()
        
        self.conv1 = Conv2dReLU(in_channels=channels, out_channels=channels,  
                                kernel_size=3,  padding=1, use_batchnorm=use_batchnorm,) # out_channels will be equal to in_channels

        # P-scSE
        self.attention1 = SCSEModule0(in_channels=channels, reduction=reduction, strategy='maxout')
        self.attention2 = SCSEModule0(in_channels=channels, reduction=reduction, strategy='addition')

    def forward(self, x):
        x1 = self.attention1(x)  # 1st attention
        x2 = self.attention2(x)  # 2nd attention

        x3 = x1 + x2

        x3 = self.conv1(x3)
        return x3
    
#%% Test
# # Define parameters
# batch_size = 2  # Number of samples in a batch
# channels = 64   # Number of channels (input/output for PscSE3D)
# height, width = 64, 64  # Input 3D volume dimensions (Depth x Height x Width)
# reduction = 16  # Reduction factor for the attention modules

# # Create dummy input data (e.g., a batch of 3D images or volumes)
# input_tensor = torch.randn(batch_size, channels, height, width)

# # Initialize the PscSE3D model
# model = PscSE2D(channels=channels, reduction=reduction, use_batchnorm=True)

# # Print model summary
# print("Model Summary:")
# summary(model, (channels, height, width))

# # Forward pass through the model
# output_tensor = model(input_tensor)

# # Print input and output shapes
# print(f"Input shape: {input_tensor.shape}")
# print(f"Output shape: {output_tensor.shape}")