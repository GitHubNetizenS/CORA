"""
MFF.py文件
　　该文件实现多级特征融合模块（MFF），通过局部特征提取、残差块和融合卷积交互高光谱与多光谱特征，输出融合后的特征图。
"""
import torch                    as torch
import torch.nn                 as nn
from collections                import OrderedDict
from models.DAConv.DAConv_CUDA  import DAConvPack


"""
_make_pair函数
　　该函数将输入值转换为二元组，若输入为整数则扩展为两个相同值的元组，否则原样返回。

value   待转换的值
"""
def _make_pair(value):
    if isinstance(value, int):
        value = (value, ) * 2

    return value


"""
conv_layer函数
　　该函数封装二维卷积层，自动根据卷积核尺寸计算填充量以实现“same”填充（输出尺寸与输入相同），避免手动指定填充值。

in_channels     输入特征图的通道数
out_channels    输出特征图的通道数
kernel_size     卷积核尺寸
bias            是否使用偏置项
"""
def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0]-1)/2), int((kernel_size[1]-1)/2))

    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


"""
activation函数
　　该函数根据指定的激活类型，返回对应的激活层实例，支持ReLU、LeakyReLU和PReLU这3种选项。

act_type    激活函数类型
inplace     是否原地执行操作
neg_slope   负半轴的斜率（用于LeakyReLU和PReLU的初始值）
n_prelu     PReLU 的可学习参数数量
"""
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()

    if "relu"==act_type:
        layer = nn.ReLU(inplace)
    elif "lrelu"==act_type:
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif "prelu"==act_type:
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError("激活层[{:s}]不存在！请检查传入的激活函数类型。".format(act_type))

    return layer


"""
sequential函数
　　该函数构建并返回1个nn.Sequential容器，自动展开参数中的nn.Sequential实例，并将多个模块按顺序组合。

*args   可变数量的模块（每个参数应为nn.Module或其子类，如nn.Sequential。）
"""
def sequential(*args):
    if 1==len(args):
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('“sequential”函数不支持“OrderedDict”类型的输入！')

        return args[0]

    modules = []

    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)

    return nn.Sequential(*modules)


class DAConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DAConv3D, self).__init__()
        pad = kernel_size // 2
        self.conv_z = DAConvPack(1, 1, kernel_size=kernel_size, padding=pad, axis=0)
        self.conv_y = DAConvPack(1, 1, kernel_size=kernel_size, padding=pad, axis=1)
        self.conv_x = DAConvPack(1, 1, kernel_size=kernel_size, padding=pad, axis=2)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_3d = x.unsqueeze(1)
        out_z = self.conv_z(x_3d)
        out_y = self.conv_y(x_3d)
        out_x = self.conv_x(x_3d)
        out_3d = out_z + out_y + out_x
        out_2d = out_3d.squeeze(1)

        return self.project(out_2d)

class DAConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(DAConv2D, self).__init__()
        pad = kernel_size // 2
        # 垂直与水平正交方向的高频纹理感知
        self.conv_y = DAConvPack(in_channels, out_channels, kernel_size=kernel_size, padding=pad, axis=1)
        self.conv_x = DAConvPack(in_channels, out_channels, kernel_size=kernel_size, padding=pad, axis=2)

    def forward(self, x):
        x_3d = x.unsqueeze(2)
        out_y_3d = self.conv_y(x_3d)
        out_x_3d = self.conv_x(x_3d)
        out_2d = (out_y_3d+out_x_3d).squeeze(2)
        return out_2d


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 这里的1x1卷积就是在频域中调节各频率分量振幅和相位的“全局均衡器”。
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.conv_spatial(x)

        # FFT：转换到频域
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm="ortho")
        # 将复数拆分为实部和虚部并在通道维度拼接
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3::1])

        # 频域全局交互
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # IFFT：转换回空域
        ffted = ffted.view((batch, -1, 2, )+ffted.size()[2:])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfft2(ffted, s=(h, w), norm="ortho")

        # 频域全局特征 + 空域局部特征
        return output+spatial_feat


"""
LFE类
　　　该类定义了局部特征提取模块，通过下采样与上采样的结构生成空间注意力，对主路径卷积提取的特征进行逐元素调制，增强局部显著区域。
"""
class LFE(nn.Module):
    """
    __init__函数
    　　该函数用于初始化LFE模块，定义注意力分支和主路径特征提取分支。

    esa_channels    注意力分支内部通道数
    n_feats         输入特征图的通道数
    """
    def __init__(self, esa_channels, n_feats):
        super(LFE, self).__init__()

        f = esa_channels
        # 定义1×1卷积层，用于统一通道数。
        self.conv0 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # 定义LFE下分支的残差卷积。
        c1_r = conv_layer(n_feats, n_feats, 3)
        c2_r = conv_layer(n_feats, n_feats, 3)
        act = activation("lrelu", neg_slope=0.05)
        self.conv1 = sequential(c1_r, act, c2_r, act)
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

    """
    forward函数
    　　该函数用于生成空间注意力，对主路径特征进行调制后输出。
    
    x   输入特征张量
    """
    def forward(self, x):
        # c1_ = self.conv0(x)
        # # 这里池化和上采样的顺序与论文中的顺序相反。
        # # 定义池化操作。
        # v_max = F.max_pool2d(c1_, kernel_size=2, stride=2)
        # c3 = self.conv3(v_max)
        # # 定义上采样操作。
        # c3 = F.interpolate(c3, scale_factor=2, mode="bilinear")
        # cf = self.conv4(c1_+c3)
        # m = self.sigmoid(cf)
        # x = self.conv1(x)
        # out = x * m
        #
        # return out
        return self.conv(x)


"""
ResBlock类
　　该类定义了轻量级残差块，使用两个1×1卷积和残差缩放因子实现恒等映射基础上的非线性变换，但当前实现中存在通道维度不一致的潜在问题。
"""
class ResBlock(nn.Module):
    """
    __init__函数
    　　该函数初始化残差块，定义两个1×1卷积与激活函数。

    in_channels     输入特征图的通道数
    out_channels    输出特征图的通道数
    stride          卷积步长
    res_scale       残差缩放因子
    """
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)
        self.relu = activation("lrelu", neg_slope=0.05)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)

    """
    forward函数
    　　该函数执行前向传播。

    x   输入张量
    """
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out*self.res_scale + x1

        return out
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
#         self.act = nn.LeakyReLU(0.05, inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
#
#     def forward(self, x):
#         res = self.conv1(x)
#         res = self.act(res)
#         res = self.conv2(res)
#
#         return x + res

"""
MFF类
　　　该类定义了多级特征融合模块，通过局部特征提取、残差融合与双分支空间注意力，对２个同尺寸输入特征图进行渐进式融合，输出增强后的特征图。
"""
class MFF(nn.Module):
    """
    __init__函数
    　　该函数初始化MFF模块，创建局部特征提取、残差融合、通道变换与空间注意力等子层。

    in_channels     输入特征图的通道数
    esa_channels    LFE模块内部通道压缩维度
    """
    def __init__(self, in_channels, esa_channels=16):
        super(MFF, self).__init__()
        # 定义LFE（局部特征增强模块）。
        # 上分支对应LFE1和LFE3，下分支对应LFE2和LFE4。
        # self.LFE1 = LFE(esa_channels, in_channels)
        # self.LFE2 = LFE(esa_channels, in_channels)
        self.LFE2 = DAConv3D(in_channels, in_channels, kernel_size=5)
        # self.LFE3 = LFE(esa_channels, in_channels)
        # self.LFE4 = LFE(esa_channels, in_channels)
        self.LFE4 = DAConv3D(in_channels, in_channels, kernel_size=3)
        self.LFE1 = DAConv2D(in_channels, in_channels, kernel_size=9)
        # 022 原始 baseline 消融：LFE1-LFE4 使用最简单的单层 3x3 卷积。
        # self.LFE1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.LFE2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.LFE2 = nn.Sequential(      #     nn.Conv2d(in_channels, in_channels, kernel_size=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.LFE3 = DAConv2D(in_channels, in_channels, kernel_size=5)
        # self.LFE3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.LFE4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.LFE4 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        # 定义激活函数LeakyReLU。
        act = activation("lrelu", neg_slope=0.05)
        self.Resblock1 = sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act,
            ResBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act
        )
        self.Resblock2 = sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act,
            ResBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act
        )
        # self.Resblock1 = nn.Sequential(
        #     nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1),
        #     act,
        #     ResBlock(in_channels)
        # )
        # self.Resblock2 = ResBlock(in_channels)
        self.c1_spat = sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            act
        )
        self.c1_spec = sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            act
        )
        self.c3 = sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=3, stride=1, padding=1),
            act
        )
        self.fusion_refine = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.fusion_refine = nn.Sequential(
        #     ChannelAttention(in_channels),
        #     SpatialAttention()
        # )
        self.fourier_unit = FourierUnit(in_channels=in_channels, out_channels=in_channels)

    """
    forward函数
    　　该函数执行前向融合，输出融合后的特征图。
    
    x   第1个输入特征图
    y   第2个输入特征图
    """
    def forward(self, x, y):
        # 第二版MFF
        # 浅层特征提取。
        x1 = self.LFE1(x)
        y1 = self.LFE2(y)
        # 空间与光谱特征进入完全独立的深层非线性映射。
        x2 = self.LFE3(self.c1_spat(x1))
        y2 = self.LFE4(self.c1_spec(y1))
        # 融合支路前馈。
        z = torch.cat((x, y), dim=1)
        z1 = self.Resblock1(z)
        z2 = self.Resblock2(z1+x1+y1)
        # z_freq = self.fourier_unit(z2)
        out_spat = x2 + x
        out_spec = y2 + y
        # 022 原始 baseline 消融：去掉两个分支的残差连接。
        # out_spat = x2
        # out_spec = y2
        # 融合支路终端合成。
        z3 = torch.cat((out_spat, out_spec, z2), dim=1)
        z4 = self.c3(z3)
        out_fused = self.fusion_refine(z4)
        # 彻底解耦输出，为主干网络的循环重构提供特征源。

        return out_spat, out_spec, out_fused

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
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        # 1×1卷积，将通道数恢复。
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
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
