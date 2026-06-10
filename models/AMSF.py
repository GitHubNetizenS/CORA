"""
AMSF.py文件
　　该文件定义了AMSF-Net的主网络架构，通过渐进式多尺度对称融合策略，实现高光谱图像与多光谱图像的融合重建，并输出空间边缘与光谱边缘特征以辅助训练。
"""

import torch                as torch
import torch.nn             as nn
import torch.nn.functional  as F
from mymodels   import MFF
from functools  import partial

nonlinearity = partial(F.relu, inplace=True)


"""
AMSF类
　　该类定义了渐进式多尺度对称融合网络，接收LRHSI和HRMSI，输出重建的HRHSI及其空间边缘与光谱边缘特征。
"""
class AMSF(nn.Module):
    """
    __init__函数
    　　该函数用于初始化网络各模块，设定尺度因子、波段数等超参数。

    n_select_bands  多光谱图像波段数
    n_bands         高光谱图像波段数
    """
    def __init__(self, n_select_bands, n_bands):
        super(AMSF, self).__init__()

        self.conv_spat = nn.Sequential(
            # 定义空间细化卷积块，用于增强空间细节。
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1)
        )
        # self.U3 = nn.Sequential(
        #     # 定义上采样特征恢复模块1。
        #     # 使用Chikusei数据集则把通道数改为128。
        #     nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )
        # self.U2 = nn.Sequential(
        #     # 定义上采样特征恢复模块2。
        #     # 使用Chikusei数据集通道数改为256。
        #     nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )
        # self.U1 = nn.Sequential(
        #     # 定义上采样特征恢复模块3。
        #     # 使用Chikusei数据集则把通道数改为256。
        #     nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        self.U3 = nn.Identity()
        self.U2 = nn.Identity()
        self.U1 = nn.Identity()
        self.conv3 = nn.Sequential(
            # 重建头部卷积块，生成最终高光谱图像。
            # 使用Chikusei数据集则把通道数改为128。
            nn.Conv2d(32, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_n_select_bands = nn.Sequential(
            # 将HRMSI映射到32通道。
            # 使用Chikusei数据集则把通道数改为128。
            nn.Conv2d(n_select_bands, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_n_bands = nn.Sequential(
            # 将LRHSI映射到32通道。
            nn.Conv2d(n_bands, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # 定义多级特征融合模块MFF。
        # 若使用Chikusei数据集则把通道数均改为128。
        self.MFF_3 = MFF.MFF(32, 32)
        self.MFF_2 = MFF.MFF(32, 32)
        self.MFF_1 = MFF.MFF(32, 32)
        # self.MFF_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.MFF_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.MFF_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # 通道注意力与空间注意力串联的注意力模块，用于增强特征表达。
        self.att = nn.Sequential(ChannelAttention(32), SpatialAttention())
        # 定义渐进跨尺度空间感知模块PCP。
        # self.PCP_3 = PCP(96, nn.BatchNorm2d, 3, 1, [1, 2, 1, 2], [1, 2, 1, 2])
        # self.PCP_2 = PCP(192, nn.BatchNorm2d,9,4,[1, 2, 3, 6],[1, 2, 3, 6])
        # self.PCP_1 = PCP(192, nn.BatchNorm2d, 9, 4, [1, 2, 3, 6], [1 , 2 , 3 , 6])
        self.PCP_3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.PCP_2 = nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1)
        self.PCP_1 = nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1)
        # self.endmember_num = 30
        # self.decoder_msi = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, self.endmember_num, kernel_size=3, stride=1, padding=1),
        #     nn.Softmax(dim=1),
        #     nn.Conv2d(self.endmember_num, n_select_bands, kernel_size=1, stride=1, padding=0, bias=False)
        # )
        # self.decoder_hsi = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, self.endmember_num, kernel_size=3, stride=1, padding=1),
        #     nn.Softmax(dim=1),
        #     nn.Conv2d(self.endmember_num, n_bands, kernel_size=1, stride=1, padding=0, bias=False)
        # )

    """
    spatial_edge函数
    　　该函数用于计算图像在高度和宽度方向上的空间边缘（相邻像素差值）。

    x   输入图像（形状为B×C×H×W。）
    """

    def spatial_edge(self, x):
        _ = self
        edge1 = x[::1, ::1, 0:x.size(2)-1:1, ::1] - x[::1, ::1, 1:x.size(2):1, ::1]
        edge2 = x[::1, ::1, ::1, 0:x.size(3) - 1:1] - x[::1, ::1, ::1, 1:x.size(3):1]

        return edge1, edge2

    """
    spectral_edge函数
    　　该函数用于计算图像在光谱维上的边缘（相邻波段差值）。

    x   输入图像（形状为B×C×H×W。）
    """

    def spectral_edge(self, x):
        _ = self
        edge = x[::1, 0:x.size(1)-1:1, ::1, ::1] - x[::1, 1:x.size(1):1, ::1, ::1]

        return edge


    """
    forward函数
    　　该函数用于执行前向传播，生成重建的高分辨率高光谱图像及空间或光谱边缘特征。
    
    x_lr    LRHSI
    x_hr    HRMSI
    """

    def forward(self, x_lr, x_hr):
        msi_base = self.conv_n_select_bands(x_hr)
        hsi_base = self.conv_n_bands(x_lr)
        msi_L1 = msi_base
        msi_L2 = self.att(F.interpolate(msi_L1, scale_factor=0.25, mode="bilinear", align_corners=False))
        msi_L3 = self.att(F.interpolate(msi_L2, scale_factor=0.25, mode="bilinear", align_corners=False))
        hsi_L1 = self.att(F.interpolate(hsi_base, size=msi_L1.shape[2::1], mode="bilinear", align_corners=False))
        hsi_L2 = self.att(F.interpolate(hsi_base, size=msi_L2.shape[2::1], mode="bilinear", align_corners=False))
        hsi_L3 = self.att(F.interpolate(hsi_base, size=msi_L3.shape[2::1], mode="bilinear", align_corners=False))
        spat3, spec3, fused3 = self.MFF_3(msi_L3, hsi_L3)
        feat_L3 = torch.cat((msi_L3, hsi_L3, fused3), dim=1)
        feat_L3 = self.PCP_3(feat_L3)
        feat_L3_up = self.U3(F.interpolate(feat_L3, scale_factor=4, mode="bilinear", align_corners=False))
        spat2, spec2, fused2 = self.MFF_2(msi_L2, hsi_L2)
        feat_L2 = torch.cat((msi_L2, hsi_L2, fused2, feat_L3_up), dim=1)
        feat_L2 = self.PCP_2(feat_L2)
        feat_L2_up = self.U2(F.interpolate(feat_L2, scale_factor=4, mode="bilinear", align_corners=False))
        spat1, spec1, fused1 = self.MFF_1(msi_L1, hsi_L1)
        feat_L1 = torch.cat((msi_L1, hsi_L1, fused1, feat_L2_up), dim=1)
        feat_L1 = self.PCP_1(feat_L1)
        feat_L1_out = self.U1(feat_L1)
        # 最后重建步骤的残差连接。
        out_init = self.conv3(feat_L1_out+hsi_L1)
        out_final = out_init + self.conv_spat(out_init)
        spat_edge1, spat_edge2 = self.spatial_edge(out_final)
        spec_edge = self.spectral_edge(out_final)

        return out_final, spat_edge1, spat_edge2, spec_edge
        # # 最右边一列的具体步骤。
        # e = self.MFF_3(b, c)
        # # e = self.MFF_3(b)
        # f3 = torch.cat((torch.cat((b, c), 1), e), 1)
        # f3 = self.PCP_3(f3)
        # # 上采样是为了对齐空间分辨率。
        # f3 = F.interpolate(f3, scale_factor=4, mode="bilinear")
        # # 卷积
        # f3 = self.U3(f3)
        # # 中间一列的具体步骤。
        # x_lr_nands_up = F.interpolate(x_lr_nands, size=(a.shape[2], a.shape[3]), mode="bilinear")
        # g = self.MFF_2(a, x_lr_nands_up)
        # # g = self.MFF_2(a)
        # f2 = torch.cat((torch.cat((a, F.interpolate(self.conv_n_bands(x_lr),
        #                                                            size=(a.shape[2], a.shape[3]),
        #                                                            mode="bilinear")), 1), g), 1)
        # f2 = torch.cat((f2, f3),1)
        # f2 = self.PCP_2(f2)
        # f2 = F.interpolate(f2, scale_factor=4, mode="bilinear")
        # f2 = self.U2(f2)
        # # 最左边一列的具体步骤。
        # h = self.MFF_1(d, x_hr_nands)
        # # h = self.MFF_1(d)
        # f1 = torch.cat((d, self.conv_n_select_bands(x_hr), h), 1)
        # f1 = torch.cat((f1, f2), 1)
        # f1 = self.PCP_1(f1)
        # f1 = self.U1(f1)
        # # Reconstruct重建步骤。
        # x = self.conv3(f1+d)
        # x = x + self.conv_spat(x)
        # spat_edge1, spat_edge2 = self.spatial_edge(x)
        # spec_edge = self.spectral_edge(x)
        #
        # return x, spat_edge1, spat_edge2, spec_edge


"""
SynchronizedBatchNorm2d类
　　该类是同步批归一化（Synchronized Batch Normalization）的占位符，用于在多GPU分布式训练中同步所有设备上的统计量，但在当前实现中为空类。
"""
class SynchronizedBatchNorm2d:
    pass


"""
ChannelAttention类
　　该类定义了通道注意力模块，通过自适应池化与共享全连接层计算通道权重，对输入特征进行通道维度的重标定。
"""
class ChannelAttention(nn.Module):
    """
    __init__函数
    　　该函数用于初始化通道注意力模块。

    in_planes   输入特征图的通道数
    """
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        # 分别定义自适应平均池化和自适应最大池化，并将输出结果分辨率压缩至1×1。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 1×1卷积，将通道数降低为原来的1/4。
        self.fc1 = nn.Conv2d(in_planes, in_planes//4, 1, bias=False)
        self.relu1 = nn.ReLU()
        # 1×1卷积，将通道数恢复。
        self.fc2 = nn.Conv2d(in_planes//4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    """
    forward函数
    　　该函数用于计算通道注意力权重，并加权到输入特征上。
    
    x   输入特征
    """
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = x * self.sigmoid(out)

        return out


"""
SpatialAttention类
　　该类定义了空间注意力模块，通过对输入特征在通道维度进行平均与最大池化后，利用卷积生成空间权重图，实现对空间位置的动态重标定。
"""
class SpatialAttention(nn.Module):
    """
    __init__函数
    　　该函数用于初始化空间注意力模块。

    kernel_size     卷积核尺寸
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    """
    forward函数
    　　该函数用于计算空间注意力权重，并加权到输入特征上。
    
    x0  输入特征
    """
    def forward(self, x0):
        avg_out = torch.mean(x0, dim=1, keepdim=True)
        max_out, _ = torch.max(x0, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = x0 * self.sigmoid(x)

        return out


"""
PCP类
　　该类定义了渐进式跨尺度空间感知模块，通过多分支空洞卷积与双向条状卷积（BSP）的渐进式融合，从不同感受野逐步提取并细化空间边缘特征，最终聚合多尺度信
息以增强特征的空间结构。
"""
class PCP(nn.Module):
    """
    __init__函数
    　　该函数用于初始化PCP模块，设置多分支空洞卷积与BSP模块，定义权重初始化方式。

    in_channels     输入特征图的通道数
    BatchNorm       批归一化类
    k               BSP模块中条状卷积的核尺寸
    p               条状卷积的填充大小
    dilation        4个扩张卷积的膨胀率
    padding         4个扩张卷积的填充值
    """
    def __init__(self, in_channels, BatchNorm, k, p, dilation, padding):
        super(PCP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = BatchNorm(in_channels)
        self.relu1 = nn.ReLU()
        # 定义4个扩张卷积块。
        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=dilation[0], padding=padding[0])
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=dilation[1], padding=padding[1])
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=dilation[2], padding=padding[2])
        self.dilate4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3,  dilation=dilation[3], padding=padding[3]))
        # 定义4个双向条状卷积模块。
        self.BSP1 = BSP(in_channels, in_channels, k, p)
        self.BSP2 = BSP(in_channels, in_channels, k, p)
        self.BSP3 = BSP(in_channels, in_channels, k, p)
        self.BSP4 = BSP(in_channels, in_channels, k, p)
        # 定义4个卷积块。
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self.Conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm(in_channels),
            nn.ReLU()
        )
        self._init_weight()

    """
    forward函数
    　　该函数执行前向传播，逐步融合多尺度空间特征并输出增强后的特征图。
    
    x   输入特征图
    """
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 定义第1分支。
        dilate1_out = nonlinearity(self.dilate1(x))
        Fea1 = self.BSP1(dilate1_out)
        F1 = self.Conv1(Fea1)
        # 定义第2分支。
        dilate2_out = nonlinearity(self.dilate2(x))
        Fea2 = self.BSP2(dilate2_out+Fea1)
        F2 = self.Conv2(Fea2)
        # 定义第3分支。
        dilate3_out = nonlinearity(self.dilate3(x))
        Fea3 = self.BSP3(dilate3_out+Fea2)
        F3 = self.Conv3(Fea3)
        # 定义第4分支。
        dilate4_out = nonlinearity(self.dilate4(x))
        Fea4 = self.BSP4(dilate4_out+Fea3)
        F4 = self.Conv4(Fea4)
        # 汇总4个分支。
        F = F1 + F2 + F3 + F4
        out = self.Conv_out(F)

        return out

    """
    _init_weight函数
    　　该函数递归初始化模块中所有卷积层和批归一化层的权重与偏置。
    """
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


"""
BSP类
　　该类定义双向条状卷积模块，将输入特征沿通道维度拆分为两组，分别通过水平与垂直方向的条状卷积提取方向性边缘特征，经深度可分离卷积与通道注意力增强后，
与原始输入相加，实现边缘感知的特征细化。
"""
class BSP(nn.Module):
    """
    __init__函数
    　　该函数初始化BSP模块，定义2个条状卷积、2个深度可分离卷积及通道注意力。
    """
    def __init__(self, in_channel, out_channel, k, p):
        super(BSP, self).__init__()
        # 定义水平条状卷积。
        self.deconv1 = nn.Conv2d(in_channel//2, out_channel//2, (1, k), padding=(0, p))
        # 定义垂直条状卷积。
        self.deconv2 = nn.Conv2d(in_channel//2, out_channel//2, (k, 1), padding=(p, 0))
        # 定义深度可分离卷积，用于水平分支。
        self.depthwise_conv1 = DepthWiseConv(in_channel=out_channel//2, out_channel=out_channel//2)
        # 定义深度可分离卷积，用于垂直分支。
        self.depthwise_conv2 = DepthWiseConv(in_channel=out_channel//2, out_channel=out_channel//2)
        # 定义通道注意力模块。
        self.ca = ChannelAttention(out_channel)
        # super(BSP, self).__init__()
        #
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )

    """
    forward函数
    　　该函数执行前向传播，输出增强后的特征图。
    
    input_feature_map   输入特征图
    """
    def forward(self, input_feature_map):
        B, C, H, W = input_feature_map.shape
        input_1, input_2 = torch.split(input_feature_map, C//2, dim=1)
        x1 = self.deconv1(input_1)
        x2 = self.deconv2(input_2)
        x3 = self.depthwise_conv1(x1)
        x4 = self.depthwise_conv2(x2)
        x = torch.cat((x3, x4), 1)
        out = self.ca(x)

        return out+input_feature_map
        # return self.conv(input_feature_map)


"""
DepthWiseConv类
　　该类定义深度可分离卷积模块，先通过逐通道卷积对每个通道独立进行空间卷积，再通过逐点卷积进行跨通道特征融合，实现高效的空间特征混合。
"""
class DepthWiseConv(nn.Module):
    """
    __init__函数
    　　该函数初始化深度可分离卷积模块，定义逐通道卷积与逐点卷积层。

    in_channel      输入特征图的通道数
    out_channel     输出特征图的通道数
    """
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积模块。
        # 当groups=in_channel时，表示做逐通道卷积。
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1,
                                    groups=in_channel)
        # 逐点卷积模块
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0,
                                    groups=1)

    """
    forward函数
    　　该函数执行前向传播，先逐通道卷积后逐点卷积，输出融合后的特征图。
    
    input_feature_map   输入特征图
    """
    def forward(self, input_feature_map):
        out = self.depth_conv(input_feature_map)
        out = self.point_conv(out)

        return out