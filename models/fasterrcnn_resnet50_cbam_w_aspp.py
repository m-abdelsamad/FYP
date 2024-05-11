import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        # AdaptiveAvgPool2d: Applies a 2D adaptive average pooling over an input signal (image).
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # AdaptiveMaxPool2d: Applies a 2D adaptive max pooling over an input signal.
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Conv2d: Applies a 2D convolution over an input signal composed of several input planes.
        # Reducing the channel count to in_channels // ratio using 1x1 convolution.
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()

        # Expanding the channel count back to in_channels.
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

    def forward(self, x):
        # Apply avg_pool and max_pool, then pass through shared MLP layers (fc1 and fc2)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # Combine the output of average and max pooling features for channel attention
        out = avg_out + max_out
        return torch.sigmoid(out)  # Sigmoid activation to get attention weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Conv2d: 7x7 convolution to combine the features from avg and max pooling
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling and max pooling across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate the avg and max pooled features
        x = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and sigmoid activation for spatial attention
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Initialize channel and spatial attention modules
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention followed by spatial attention
        x = x * self.channel_attention(x)  # Element-wise multiplication with channel attention map
        x = x * self.spatial_attention(x)  # Element-wise multiplication with spatial attention map
        return x



class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates):
        super(ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


def create_model(num_classes=13, pretrained=True):
    model_backbone = torchvision.models.resnet50(weights='DEFAULT')

    conv1 = model_backbone.conv1
    bn1 = model_backbone.bn1
    relu = model_backbone.relu
    max_pool = model_backbone.maxpool
    layer1 = model_backbone.layer1
    layer2 = model_backbone.layer2
    layer3 = model_backbone.layer3
    layer4 = model_backbone.layer4
    
    cbam_block1 = CBAMBlock(in_channels=512)
    cbam_block2 = CBAMBlock(in_channels=2048)

    aspp = ASPP(in_ch=2048, out_ch=512, rates=[6, 12, 18])
    
    backbone = nn.Sequential(
        conv1, 
        bn1, 
        relu, 
        max_pool, 
        layer1, # Residual Block 1 -> out channel 256
        layer2, # Residual Block 2 -> out channel 512
        cbam_block1, 
        layer3, # Residual Block 3 -> out channel 1024
        layer4, # Residual Block 4 -> out channel 2048
        cbam_block2,
        aspp
    )

    backbone.out_channels = 512

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=14,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=13, pretrained=True)
    summary(model)