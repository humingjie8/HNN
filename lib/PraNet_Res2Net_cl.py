import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s


"""
PraNet model definition (Res2Net backbone) with RFB-like modules.

This module defines the building blocks and the PraNet architecture
used for saliency/segmentation tasks. It provides:
- `BasicConv2d`: small conv -> BN -> (optionally) ReLU block used
    throughout the network;
- `RFB_modified`: a receptive-field block variant with multi-branch
    dilated convolutions;
- `aggregation`: dense aggregation module to fuse multi-scale features;
- `PraNet`: the full model combining a Res2Net backbone with RFB and
    reverse-attention branches.

Continual learning & architecture adaptation:
- This variant is adapted to support continual learning scenarios by
    allowing `task_num` parallel task-specific reverse-attention heads
    (see `PraNet.__init__` and the `last` ModuleList). The design
    enables multiple task heads and easier integration with continual
    learning strategies (e.g., per-task heads, parameter freezing,
    EWC-like regularization) without changing the core backbone.

Only descriptive docstrings are added; no functional changes.
"""

class BasicConv2d(nn.Module):
    """Simple Conv-BatchNorm block used across the network.

    Performs a 2D convolution followed by batch normalization. The
    ReLU activation is defined in `__init__` but the forward method
    returns the normalized convolution output (activation applied by
    callers where needed).
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass: conv -> bn.

        Note: callers often apply `ReLU` after this block when needed.
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    """Receptive Field Block (modified) with multi-branch dilated convs.

    The block builds four branches with increasing receptive fields
    (standard 1x1, and several asymmetric + dilated convolutions),
    concatenates their outputs, and applies a residual-like connection.
    """
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)  # 定义ReLU激活层
        
        # 分支0
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),  # 1x1卷积
        )
        
        # 分支1
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),  # 1x1卷积
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),  # 1x3卷积
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),  # 3x1卷积
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)  # 3x3卷积，膨胀率为3
        )
        
        # 分支2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),  # 1x1卷积
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),  # 1x5卷积
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),  # 5x1卷积
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)  # 3x3卷积，膨胀率为5
        )
        
        # 分支3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),  # 1x1卷积
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),  # 1x7卷积
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),  # 7x1卷积
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)  # 3x3卷积，膨胀率为7
        )
        
        # 合并卷积层
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)  # 3x3卷积
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)  # 1x1卷积，用于残差连接

    def forward(self, x):
        """Forward pass through the four branches and residual fusion."""
        # pass through each branch
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        # 合并分支输出，并通过卷积层
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        
        # 残差连接并通过ReLU激活层
        x = self.relu(x_cat + self.conv_res(x))
        return x



class aggregation(nn.Module):
    """Dense aggregation module to fuse multi-scale feature maps.

    This module upsamples and combines three input features (typically
    from different backbone depths) and produces a single fused map.
    It is similar in spirit to DSS/Amulet-style aggregation blocks.
    """
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        """Fuse three feature maps `x1`, `x2`, `x3` into one output map.

        Expected shapes correspond to progressively lower spatial
        resolutions (x1 highest-resolution among the three).
        """
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class PraNet(nn.Module):
    """PraNet model combining a Res2Net backbone with RFB and reverse-attention.

    The model extracts multi-scale features from a Res2Net backbone,
    passes them through RFB modules, aggregates them, and applies a
    series of reverse-attention branches. `task_num` controls how many
    parallel task-specific reverse-attention heads are created for the
    second reverse-attention stage.
    """
    def __init__(self, task_num,channel=32,):
        super(PraNet, self).__init__()
        self.taskinf = task_num
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # 为每个任务动态创建特定的结构
        self.last = nn.ModuleList()
        for t in range(self.taskinf):

            # ---- reverse attention branch 2 ----
            ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
            ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

            self.last.append(nn.Sequential(ra2_conv1, ra2_conv2, ra2_conv3, ra2_conv4))

    def forward(self, x):
        """Forward pass through backbone, RFB modules, aggregation and RA heads.

        Returns four lateral maps: high-level aggregated map and three
        reverse-attention outputs. The last output is a list with length
        equal to `task_num`, containing task-specific maps.
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)

        # Select the specific module sequence based on task index
        ra2_feat = []
        lateral_map_2 = []
        for t in range(self.taskinf):
            # print(x.shape)
            ra_module = self.last[t]
            
            y = ra_module[0](x)
            y = F.relu(ra_module[1](y))
            y = F.relu(ra_module[2](y))
            y = ra_module[3](y)
            ra2_feat.append (y)
            
        for item in ra2_feat:
            out = item + crop_2
            lateral_map_2 .append(F.interpolate(out, scale_factor=8, mode='bilinear'))   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


if __name__ == '__main__':
    ras = PraNet(task_num=5).cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    task_num = 5

    out1, out2, out3, out4 = ras(input_tensor)
    print(out4[1])