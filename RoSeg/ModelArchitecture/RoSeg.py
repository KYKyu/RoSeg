import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from backbone.DFormerv2 import DFormerv2_S as dformer
import os
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
import torchvision.transforms as T
import cv2
import numpy as np


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, filters, dilation_rate=1):
        super(ResNetBlock, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, filters, kernel_size=1, padding=0,
                                 dilation=dilation_rate, bias=False)
        
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        self.out_bn = nn.BatchNorm2d(filters)

        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.conv1x1(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        out = out + identity
        out = self.out_bn(out)

        return out

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='relu', padding='same'):
        super(ConvBNAct, self).__init__()
        if padding == 'same':
            padding_val = (kernel_size - 1) // 2
        else:
            padding_val = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding_val, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class SBA(nn.Module):
    def __init__(self, in_channels_L, in_channels_H, dim=16):
        super(SBA, self).__init__()
        self.dim = dim

        # 1x1 conv to reduce channel to `dim`
        self.L_reduce = nn.Conv2d(in_channels_L, dim, kernel_size=1, padding=0, bias=False)
        self.H_reduce = nn.Conv2d(in_channels_H, dim, kernel_size=1, padding=0, bias=False)

        # Activation for gate
        self.sigmoid = nn.Sigmoid()

        # Conv+BN+Act
        self.L_process = ConvBNAct(dim, dim, kernel_size=1)
        self.H_process = ConvBNAct(dim, dim, kernel_size=1)

        # Final convs
        self.final_conv = ConvBNAct(dim * 2, dim * 2, kernel_size=3)
        self.output_conv = nn.Conv2d(dim * 2, 1, kernel_size=1, padding=0, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.L_reduce, self.H_reduce, self.output_conv]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, L_input, H_input):
        # 1x1 reduce
        L = self.L_reduce(L_input)
        H = self.H_reduce(H_input)

        g_L = self.sigmoid(L)
        g_H = self.sigmoid(H)


        L = self.L_process(L)
        H = self.H_process(H)

        # Attention-weighted fusion (upsample with nearest to match TensorFlow)
        H_upsampled = F.interpolate(H * g_H, scale_factor=2, mode='nearest')
        L_feature = L + L * g_L + (1 - g_L) * H_upsampled

        # Resize L to match H
        _, _, h, w = H.shape
        L_resized = F.interpolate(L * g_L, size=(h, w), mode='nearest')
        H_feature = H + H * g_H + (1 - g_H) * L_resized

        # Final upsampling
        H_feature = F.interpolate(H_feature, scale_factor=2, mode='nearest')

        # Concatenate and refine
        out = torch.cat([L_feature, H_feature], dim=1)  # concat along channel
        out = self.final_conv(out)
        out = self.output_conv(out)

        return out

kernel_initializer = 'he_uniform'
interpolation = "nearest"
pretrained = os.path.dirname(__file__) + "/../backbone/pretrained/DFormerv2_Small_pretrained.pth"
depth_model_pretained = os.path.dirname(__file__) + '/../DepthAnythingV2/checkpoints/depth_anything_v2_vits.pth'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

class RoSeg(nn.Module):
    def __init__(self, input_channels, out_classes, starting_filters):
        super(RoSeg, self).__init__()
        self.dim = 32

        self.backbone = dformer(drop_path_rate = 0.1, norm_cfg = True)
        
        self.depth_model = DepthAnythingV2(**model_configs[encoder])
        self.depth_model.load_state_dict(torch.load(depth_model_pretained))
        self.depth_model.eval()
        for p in self.depth_model.parameters():
            p.requires_grad = False
        
        # Initial conv
        self.p1_conv = nn.Conv2d(input_channels, starting_filters * 2, kernel_size=3, stride=2, padding=1)

        # Reduce channels for each layer
        self.p2_conv = nn.Conv2d(64, starting_filters * 4, kernel_size=1)
        self.p3_conv = nn.Conv2d(128, starting_filters * 8, kernel_size=1)
        self.p4_conv = nn.Conv2d(256, starting_filters * 16, kernel_size=1)
        self.p5_conv = nn.Conv2d(512, starting_filters * 32, kernel_size=1)

        self.down1 = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2)
        self.down2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2)
        self.down3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2)
        self.down4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2)
        self.down5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2)

        # Residual blocks assumed to be nn.Module classes
        self.rapu0 = ResNetBlock(input_channels, starting_filters)
        self.rapu1 = ResNetBlock(starting_filters * 2, starting_filters * 2)
        self.rapu2 = ResNetBlock(starting_filters * 4, starting_filters * 4)
        self.rapu3 = ResNetBlock(starting_filters * 8, starting_filters * 8)
        self.rapu4 = ResNetBlock(starting_filters * 16, starting_filters * 16)

        # Residual and processing blocks
        self.t51 = nn.Sequential(
            ResNetBlock(starting_filters * 32, starting_filters * 32),
            ResNetBlock(starting_filters * 32, starting_filters * 32),
        )
        self.t53 = nn.Sequential(
            ResNetBlock(starting_filters * 32, starting_filters * 16),
            ResNetBlock(starting_filters * 16, starting_filters * 16),
        )

        self.outd_conv = ConvBNAct(starting_filters * 16 + starting_filters * 16 + starting_filters * 8, self.dim, kernel_size=1)
        self.outd_out = nn.Conv2d(self.dim, 1, kernel_size=1, bias=False)

        self.L_input_conv = ConvBNAct(starting_filters * 4, self.dim, kernel_size=3)
        self.H_input_conv = ConvBNAct(starting_filters * 16 + starting_filters * 16, self.dim, kernel_size=1)

        # self.sba = SBA(in_channels_L=self.dim, in_channels_H=self.dim, dim=self.dim)
        self.sba = SBA(in_channels_L=self.dim, in_channels_H=self.dim)

        self.final_conv = nn.Conv2d(1, out_classes, kernel_size=1)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained:
            print(f"Loading pretrained weights from {pretrained}")
            self.backbone.init_weights(pretrained=pretrained)

        for m in [self.p2_conv, self.p3_conv, self.p4_conv, self.p5_conv]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.p1_conv.weight)
        if self.p1_conv.bias is not None:
            nn.init.zeros_(self.p1_conv.bias)

        nn.init.xavier_uniform_(self.outd_out.weight)
        if self.outd_out.bias is not None:
            nn.init.zeros_(self.outd_out.bias)

        for m in [self.down1, self.down2, self.down3, self.down4, self.down5]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def train(self, mode=True):
        super().train(mode)
        self.depth_model.eval()
        for p in self.depth_model.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        x_e = self.depth_model(x)
        features = self.backbone(x, x_e)
        p2 = self.p2_conv(features[0]) # (B, C, H, W)
        p3 = self.p3_conv(features[1])
        p4 = self.p4_conv(features[2])
        p5 = self.p5_conv(features[3])

        p1 = self.p1_conv(x)

        t0 = self.rapu0(x)
        s1 = self.down1(t0) + p1
        t1 = self.rapu1(s1)

        s2 = self.down2(t1) + p2
        t2 = self.rapu2(s2)

        s3 = self.down3(t2) + p3
        t3 = self.rapu3(s3)

        s4 = self.down4(t3) + p4
        t4 = self.rapu4(s4)

        s5 = self.down5(t4) + p5
        t51 = self.t51(s5)
        t53 = self.t53(t51)

        # Aggregation
        t53_up4 = F.interpolate(t53, scale_factor=4, mode='nearest')
        t4_up2 = F.interpolate(t4, scale_factor=2, mode='nearest')
        outd = torch.cat([t53_up4, t4_up2, t3], dim=1)
        outd = self.outd_conv(outd)
        outd = self.outd_out(outd)

        # SBA input preparation
        L_input = self.L_input_conv(t2)
        t53_up2 = F.interpolate(t53, scale_factor=2, mode='nearest')
        H_input = torch.cat([t53_up2, t4], dim=1)
        H_input = self.H_input_conv(H_input)
        H_input = F.interpolate(H_input, scale_factor=2, mode='nearest')

        out2 = self.sba(L_input, H_input)

        out1 = F.interpolate(outd, scale_factor=8, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)

        output = out1 + out2
        output = self.final_conv(output)
        output = torch.sigmoid(output)

        return output

