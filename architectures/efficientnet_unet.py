import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EfficientNetV2SUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetV2SUNet, self).__init__()
        
        # Load EfficientNet-V2-S backbone
        self.backbone = models.efficientnet_v2_s(pretrained=pretrained)
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        
        # Get feature maps from different layers
        self.backbone_features = self.backbone.features
        
        # Feature dimensions for EfficientNet-V2-S
        # Layer 0: 24 channels
        # Layer 1: 24 channels  
        # Layer 2: 48 channels
        # Layer 3: 64 channels
        # Layer 4: 128 channels
        # Layer 5: 160 channels
        # Layer 6: 256 channels
        # Layer 7: 1280 channels
        
        # Decoder
        self.up1 = Up(1280 + 256, 256)  # 1280 + 256 -> 256
        self.up2 = Up(256 + 160, 128)   # 256 + 160 -> 128
        self.up3 = Up(128 + 128, 64)    # 128 + 128 -> 64
        self.up4 = Up(64 + 64, 32)      # 64 + 64 -> 32
        self.up5 = Up(32 + 48, 16)      # 32 + 48 -> 16
        self.up6 = Up(16 + 24, 8)       # 16 + 24 -> 8
        
        # Final classifier
        self.outc = nn.Conv2d(8, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Store features from encoder
        features = []
        
        # Extract features from each layer
        for i, layer in enumerate(self.backbone_features):
            x = layer(x)
            if i in [0, 1, 2, 3, 4, 5, 6]:  # Save intermediate features
                features.append(x)
        
        # Final feature map after layer 7
        x7 = x  # 1280 channels
        
        # Decoder path
        x = self.up1(x7, features[6])      # 1280 + 256 -> 256
        x = self.up2(x, features[5])       # 256 + 160 -> 128  
        x = self.up3(x, features[4])       # 128 + 128 -> 64
        x = self.up4(x, features[3])       # 64 + 64 -> 32
        x = self.up5(x, features[2])       # 32 + 48 -> 16
        x = self.up6(x, features[1])       # 16 + 24 -> 8
        
        # Final upsampling to original size
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Output
        logits = self.outc(x)
        
        return logits


def build_efficientnet_v2_s_unet(num_classes=1, pretrained=True):
    """Build EfficientNet-V2-S UNet model"""
    return EfficientNetV2SUNet(num_classes=num_classes, pretrained=pretrained)
