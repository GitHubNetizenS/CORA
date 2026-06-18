"""
preprocessing_utils.py文件
　　该Python文件提供了高光谱图像融合任务所需的核心工具函数，包括退化模型的构建（如Gauss模糊核和光谱响应矩阵等）、图像空间下采样、学习率调整、指标统
计以及测试阶段的重建拼接。
"""

import os                   as os
import re                   as re
import glob                 as glob
import numpy                as np
import torch                as torch
import random               as random
import torch.nn.functional  as F
from math           import exp
from numpy          import *
from scipy          import signal
from torch          import nn
from scipy.signal   import savgol_coeffs
from torch.autograd import Variable

# 在程序运行时，防止某些异常或信号导致执行中断并弹出调试器，避免程序运行时被调试器捕获而暂停。
os.environ["FOR_IGNORE_EXCEPTIONS"] = '1'
# 创建损失函数对象，“mean”是指对各个样本误差取均值。
loss_func_re = nn.L1Loss(reduction="mean").cuda()


"""
set_seed函数
　　该函数用于设定全局种子。

seed    种子值
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""
savitzky_golay_kernel函数
　　该函数用于生成SG一阶导数卷积核。

window_size     滑动窗口大小（奇数）
poly_order      多项式阶数
"""
def savitzky_golay_kernel(window_size, poly_order):
    # 生成SG一阶导数系数，deriv=1表示一阶导数。
    coeffs = savgol_coeffs(window_size, poly_order, deriv=1)
    kernel = torch.tensor(coeffs, dtype=torch.float32)

    return kernel

"""
spectral_gradient_sg函数
　　该函数用于使用SG滤波器计算光谱梯度。

x               输入张量
window_size     滑动窗口大小（奇数）
poly_order      多项式阶数
"""
def spectral_gradient_sg(x, window_size=5, poly_order=2):

    kernel = savitzky_golay_kernel(window_size, poly_order).to(x.device)
    # kernel形状为(window_size,)，需要扩展为1D卷积格式。
    # 将H和W维度合并，在C维度上做1D卷积。
    B, C, H, W = x.shape
    # 将x重排为(B*H*W, 1, C)，在C维度上做1D卷积。
    x_reshaped = x.permute(0, 2, 3, 1).reshape(B*H*W, 1, C)
    kernel_1d = kernel.view(1, 1, -1)
    pad = window_size // 2
    # 对C维度做对称填充。
    x_padded = F.pad(x_reshaped, (pad, pad), mode="reflect")
    grad = F.conv1d(x_padded, kernel_1d)
    # 恢复形状为B×C×H×W。
    grad = grad.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return grad



"""
sobel_gradient函数
　　该函数用于对每个通道独立计算Sobel梯度。

x   输入的特征图
"""
def sobel_gradient(x):
    # x形状为B×C×H×W，对每个通道独立计算Sobel梯度。
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(x.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(x.size(1), 1, 1, 1)
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))

    return grad_x, grad_y


"""
mkdir函数
　　该函数创建目录，如果目录已存在则不做任何修改并给出提示。

path    要创建的目录路径（支持多级目录，若上级目录不存在会一并创建。）
"""
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        # 判断是否存在文件夹，如果不存在则创建为文件夹。
        # 创建文件时如果路径不存在会创建这个路径。
        os.makedirs(path, exist_ok=True)
        print("文件夹{}已创建。".format(path))
    else:
        print("文件夹{}已存在。".format(path))


"""
findLastCheckpoint函数
　　该函数在指定目录中查找最新保存的模型检查点文件，返回其对应的epoch编号。

save_dir    要搜索的目录路径（函数会在此目录下查找以“model_”开头、以“.pth”结尾的文件。）
"""
def findLastCheckpoint(save_dir):
    # 根据匹配模式查找所有的文件名，*、[]、?三种常用通配符。
    file_list = glob.glob(os.path.join(save_dir, "model_*.pth"))

    if file_list:
        epochs_exist = []

        for file_ in file_list:
            # 捕获模型文件后面跟的数字，()中为捕获组。
            result = re.findall(".*model_(.*).pth.*", file_)

            epochs_exist.append(int(result[0]))

        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0

    return initial_epoch


"""
get_filename_list函数
　　该函数获取指定目录下的所有文件名（排除子目录），并可选择随机打乱顺序。

path        要扫描的目录路径
shuffle     是否随机打乱返回的文件名列表（默认为False，即保持原顺序。）
"""
def get_filename_list(path, shuffle=False):
    filename_list = os.listdir(path)
    filename_list = [item for item in filename_list if os.path.isfile(os.path.join(path, item))]

    if shuffle:
        random.shuffle(filename_list)

    return filename_list

"""
fspecial函数
　　该函数生成指定大小和标准差的Gauss模糊核。

func_name       滤波器类型（如gaussian表示Gauss模糊核。）
kernel_size     Gauss核的边长（边长为整数。）
sigma           Gauss核的标准差（用于控制平滑程度。）
"""
def fspecial(func_name, kernel_size, sigma):
    if "gaussian"==func_name:
        print("传入的滤波器类型为“gaussian”。")

        # 　　通常而言，Gauss核的边长为奇数。对于一个k×k的Gauss核，其下标范围为“(0, 0)~(k-1, k-1)”。为了求得Gauss核的中心元素坐标(m, n)，
        # 需要使用公式“[0+(k-1)]/2”计算中心元素坐标((k-1)/2, (k-1)/2)。当然若边长不为奇数，也可以成立，只不过中央2×2的位置会成为中心。
        m = (kernel_size-1.0) / 2.0
        n = (kernel_size-1.0) / 2.0
        # y的范围为[-m, m]，x的范围为[-n, n]，二维范围为“(-n, m)~(n, -m)”，中心为(0, 0)。
        y, x = ogrid[-m:m+1:1, -n:n+1:1]
        h = exp(-(x*x+y*y) / (2.0*sigma*sigma))
        # 消除因浮点计算产生的极小的非零值。
        h[h<finfo(h.dtype).eps*h.max()] = 0
        # 进行归一化。
        sumh = h.sum()

        if sumh != 0:
            h /= sumh

        return h
    print("传入的模糊核类型有误！目前模糊核类型仅支持“gaussian”，请检查您的代码。")

    return None


"""
Gaussian_downsample函数
　　该函数对输入图像（支持单波段或多波段）应用给定的点扩散函数（PSF）进行二维卷积（即空间模糊），然后按指定因子进行下采样，生成低分辨率图像。

x       输入的图像（可以是2维的“高度×宽度”，也可以是3维的“通道×高度×宽度”。）
psf     点扩散函数（如Gauss模糊核。）
s       下采样因子
"""
def Gaussian_downsample(x, psf, s):
    if 2==x.ndim:
        # 如果输入图像x是2维（即单通道图像），则在最前面增加1个维度，使其变为3维(1, H, W)，便于后续统一处理。
        x = np.expand_dims(x, axis=0)

    # 创建全零数组y，用于存放下采样后的结果。其维度为(通道数, 原高度//s, 原宽度//s)。这里假设原高度和宽度都能被s整除（通常由外部保证）。
    y = np.zeros((x.shape[0], int(x.shape[1]/s), int(x.shape[2]/s)))

    for i in range(0, x.shape[0], 1):
        # 依次取出每个通道的图像，做卷积。
        x1 = x[i, ::1, ::1]
        x2 = signal.convolve2d(x1, psf, boundary="symm", mode="same")
        y[i, ::1, ::1] = x2[0::s, 0::s]

    return y


"""
create_F函数
　　该函数生成一个归一化的光谱响应矩阵，用于模拟多光谱传感器对高光谱图像的光谱下采样过程。
"""
def create_F():
    F = np.array(
        [[2 , 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 2 , 6 , 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
                [1 , 1 , 1 , 1 , 1 , 1 , 2 , 4 , 6 , 8 , 11, 16, 19, 21, 20, 18, 16, 14, 11, 7 , 5 , 3 , 2 , 2 , 1 , 1 , 2 , 2 , 2 , 2 , 2 ],
                [7 , 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]],
        dtype=float64)

    for band in range(0, 3, 1):
        div = np.sum(F[band][: : 1])

        # 对光谱响应矩阵进行归一化处理。
        for i in range(0, 31, 1):
            F[band][i] = F[band][i] / div

    return F


"""
reconstruction函数
　　该函数在验证或测试阶段，对整幅高分辨率图像进行滑动窗口重建，通过模型预测每个图像块并加权平均拼接得到完整的高分辨率高光谱图像，同时计算每个预测块与
真实块之间的损失并累计。

net2                已训练好的PyTorch模型（接受低分辨率高光谱图像块和高分辨率多光谱图像块作为输入，输出预测的高分辨率高光谱图像块。）
R                   光谱响应矩阵（形状为(3, 31)，用于从HRHSI下采样到HRMSI。）
HSI_LR              低分辨率高光谱图像（形状为“1×C×h×w”。）
MSI_HR              高分辨率多光谱图像（形状为“1×c×H×W”。）
HSI_HR              高分辨率高光谱图像（形状为“1×C×H×W”。）
downsample_factor   空间下采样因子
training_size       模型输入块的尺寸
stride              滑动窗口的步长
val_loss            所有预测块的损失值
"""
def reconstruction(net2, R, HSI_LR, MSI_HR, HSI_HR, downsample_factor, training_size, stride, val_loss):
    # 创建index_matrix和abundance_t两个张量，尺寸都为“C×H×W”。
    # index_matrix用于记录每个像素被覆盖的次数，abundance_t用于累加每个像素的预测值。
    device = MSI_HR.device
    index_matrix = torch.zeros((R.shape[1], MSI_HR.shape[2], MSI_HR.shape[3]), device=device)
    abundance_t = torch.zeros((R.shape[1], MSI_HR.shape[2], MSI_HR.shape[3]), device=device)
    # 生成高分辨率图像行方向（H）的起始坐标列表。
    a = []

    for j in range(0, MSI_HR.shape[2] - training_size + 1, stride):
        a.append(j)
    # 为避免边界值没有被取到，故手动添加。
    if a[-1]!=MSI_HR.shape[2]-training_size:
        a.append(MSI_HR.shape[2]-training_size)

    # 生成高分辨率图像列方向（W）的起始坐标列表。
    b = []

    for j in range(0, MSI_HR.shape[3] - training_size + 1, stride):
        b.append(j)
    # 为避免边界值没有被取到，故手动添加。
    if b[-1]!=MSI_HR.shape[3]-training_size:
        b.append(MSI_HR.shape[3]-training_size)
    for j in a:
        for k in b:
            # 此次窗口覆盖的HRMSI范围。
            temp_hrms = MSI_HR[::1, ::1, j:j + training_size:1,
                                      k:k+training_size:1]
            # 此次窗口覆盖的LRHSI范围。
            temp_lrhs = HSI_LR[::1, ::1, int(j/downsample_factor):int((j+training_size)/downsample_factor):1,
                                         int(k/downsample_factor):int((k+training_size)/downsample_factor):1]
            # 此次窗口覆盖的HRHSI范围。
            temp_hrhs = HSI_HR[::1, ::1, j:j + training_size:1,
                                        k:k+training_size:1]

            with torch.no_grad():
                # 禁用梯度计算。
                # out的尺寸为“1×C×training_size×training_size”，为预测的HRHSI片段。
                out, _, _, _ = net2(temp_lrhs, temp_hrms)
                # 将预测的HRHSI片段与真实的HRHSI片段比较。
                loss_temp = loss_func_re(out, temp_hrhs)

                val_loss.update(loss_temp)
                # 去掉第1维（batch维度）。
                HSI = out.squeeze()
                # 将预测值裁剪到[0, 1]范围内。
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[::1, j:j+training_size:1, k:k+training_size:1] = abundance_t[::1, j:j+training_size:1,
                                                                                              k:k+training_size:1] + HSI
                index_matrix[::1, j:j+training_size:1, k:k+training_size:1] = index_matrix[::1, j:j+training_size:1,
                                                                                                k:k+training_size:1] + 1

    # 由于同一位置可能被多个块覆盖，取平均可减少块边界效应。
    HSI_recon = abundance_t / index_matrix

    assert 0==torch.isnan(HSI_recon).sum()

    return HSI_recon, val_loss


"""
AverageMeter类
　　该类用于跟踪和计算一系列数值的平均值、总和及当前值，常用于训练过程中记录损失或指标的动态平均值。
"""
class AverageMeter(object):
    """
    __init__函数
    　　该函数用于初始化AverageMeter实例，调用reset()将所有计数器置零。

    count   参与计算的总样本数（或总权重）
    sum     所有传入值的总和（或加权总和）
    avg     到目前为止所有传入值的平均值（或加权平均值）
    val     最近1次更新传入的值
    """
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()


    """
    reset函数
    　　该函数用于重置所有统计量为初始状态。
    """
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    """
    update函数
    　　该函数用于更新统计量。
    
    val     当前传入的值
    n       该值对应的权重或样本总数
    """
    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += (val*n)
        self.count += n
        self.avg = self.sum / self.count


"""
Loss_PSNR类
　　该类计算两幅图像之间的峰值信噪比（PSNR），常用于评估图像重建质量。
"""
class Loss_PSNR(nn.Module):
    """
    __init__函数
    　　该函数构造函数，调用父类初始化。
    """
    def __init__(self):
        super(Loss_PSNR, self).__init__()


    """
    forward函数
    　　该函数用于计算PSNR。
    
    im_true     真实图像张量（形状任意，像素值应在[0,1]范围内）
    im_fake     预测图像张量（与im_true形状相同，像素值同样应在[0,1]范围内）
    data_range  图像像素值范围（默认为255（对应8位图像），用于将归一化像素值还原到原始尺度）
    """
    def forward(self, im_true, im_fake, data_range=255):
        _ = self
        # 首先规范像素值范围到[0, 1]，然后乘以data_range恢复到原尺度。
        Itrue = im_true.clamp(0.0, 1.0) * data_range
        Ifake = im_fake.clamp(0.0, 1.0) * data_range
        # 计算像素级的误差。
        # err = Itrue - Ifake
        # err = torch.pow(err, 2)
        # err = torch.mean(err, dim=0)
        # err = torch.mean(err, dim=0)
        mse = torch.mean((Itrue-Ifake)**2, dim=[-2, -1])
        psnr = 10.0 * torch.log10((data_range**2)/mse)
        psnr = torch.mean(psnr)

        return psnr.item()


"""
Loss_RMSE类
　　该类计算两幅图像之间的均方根误差（RMSE），常用于评估图像重建的精度。
"""
class Loss_RMSE(nn.Module):
    """
    __init__函数
    　　该函数构造函数，调用父类初始化。
    """
    def __init__(self):
        super(Loss_RMSE, self).__init__()


    """
    forward函数
    　　该函数用于计算RMSE。
    
    outputs     预测图像张量（像素值应在 [0,1] 范围内。）
    label       真实图像张量（像素值应在 [0,1] 范围内。）
    """
    def forward(self, outputs, label):
        _ = self

        assert outputs.shape==label.shape

        error = outputs.clamp(0.0, 1.0)*255 - label.clamp(0.0, 1.0)*255
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))

        return rmse.item()


"""
Loss_SAM类
　　该类计算两幅高光谱图像之间的平均光谱角（SAM），用于评估光谱信息保持程度。
"""
class Loss_SAM(nn.Module):
    """
    __init__函数
    　　该函数构造函数，调用父类初始化。
    """
    def __init__(self):
        super(Loss_SAM, self).__init__()

        self.eps = 2.2204e-16


    """
    forward函数
    　　该函数用于计算两张图像的平均光谱角。
    
    im1     第1幅图像
    im2     第2幅图像
    """
    def forward(self, im1, im2):
        assert im1.shape==im2.shape

        H, W, C = im1.shape
        im1 = np.reshape(im1, (H*W, C))
        im2 = np.reshape(im2, (H*W, C))
        core = np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1, 1)))

        return np.mean(sam)


"""
Loss_SSIM类
　　该类计算两幅图像之间的结构相似性指数（SSIM），作为损失函数或评估指标。SSIM越接近1表示两幅图像越相似。
"""
class Loss_SSIM(nn.Module):
    """
    __init__函数
    　　该函数为构造函数。
    """
    def __init__(self):
        super(Loss_SSIM, self).__init__()
        pass

    """
    forward函数
    　　该函数为前向传播函数，接收两幅图像和可选参数，返回SSIM值。
    
    img1            第1张输入图像（4D张量，形状为(N, C, H, W)。）
    img2            第2张输入图像
    window_size     Gauss窗口的大小（默认为11。)
    size_average    是否对每个样本和通道的SSIM取平均（若为True返回标量，否则返回每个样本的SSIM值。）
    """
    def forward(self, img1, img2, window_size=11, size_average=True):
        _ = self
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)

        if img1.is_cuda:
            # 如果输入在GPU上，将窗口移到同一设备，将窗口数据类型转换为与img1一致。
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    """
    _ssim函数
    　　该函数执行实际的SSIM计算。
    
    img1            第1张输入图像（4D张量，形状为(N, C, H, W)。）
    img2            第2张输入图像
    window          预计算的Gauss窗口
    window_size     Gauss窗口的大小（默认为11。)
    channel         图像通道数
    size_average    是否对每个样本和通道的SSIM取平均（若为True返回标量，否则返回每个样本的SSIM值。）
    """
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        _ = self
        # 使用卷积对图像进行加权平均（即局部均值估计）。padding保持输出尺寸不变，groups=channel实现逐通道卷积。
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        # 计算均值的平方和均值乘积。
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        # 计算方差和协方差：通过卷积得到平方的均值，减去均值的平方得到方差。
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        # 定义稳定常数，防止分母为零。通常基于像素值动态范围（这里假设像素值在[0, 1]范围内）。
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        # 根据SSIM公式计算每个像素的SSIM值，得到一张与输入空间尺寸相同的SSIM图。
        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))

        if size_average:
            # 若size_average=True，则对所有元素取平均，返回标量。

            return ssim_map.mean().item()
        else:
            # 　　否则，对每个样本（batch）在空间维度和通道维度取平均，返回形状为(N, )的张量。注意mean(1).mean(1).mean(1)依次对通道、高、宽
            # 取平均。

            return ssim_map.mean(1).mean(1).mean(1).cpu().numpy()


    """
    gaussian函数
    　　该函数生成1维Gauss核。
    
    window_size     Gauss窗口的大小（默认为11。)
    sigma           Gauss标准差
    """
    def gaussian(self, window_size, sigma):
        _ = self
        gauss = torch.Tensor([exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(0, window_size, 1)])

        return gauss/gauss.sum()


    """
    create_window函数
    　　该函数创建2维Gauss窗口，扩展至多通道形式。
    
    window_size     窗口大小
    channel         图像通道数
    """
    def create_window(self, window_size, channel):
        _ = self
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

        return window


"""
Loss_ERGAS类
　　该类为相对全局无量纲误差（ERGAS）损失函数类，用于评估融合图像与真实图像在各波段的综合误差。
"""
class Loss_ERGAS(nn.Module):
    """
    __init__函数
    　　该函数为构造函数。
    """
    def __init__(self):
        super(Loss_ERGAS, self).__init__()

    """
    forward函数
    　　前向传播函数，计算两张图像之间的ERGAS值。
    
    img_tgt     目标图像（真实高分辨率高光谱图像，形状为(1, C, H, W)的4D张量。）
    img_fus     融合图像（模型预测的高分辨率高光谱图像，形状与img_tgt相同。）
    """
    def forward(self, img_tgt, img_fus):
        _ = self
        scale = 8
        # 去除batch维度。
        img_tgt = img_tgt.squeeze(0).data.cpu().numpy()
        img_fus = img_fus.squeeze(0).data.cpu().numpy()

        assert 3==img_tgt.ndim, f"目标图像img_tgt的维度为{img_tgt.ndim}维，不是3维！"
        assert 3==img_fus.ndim, f"融合图像img_fus的维度为{img_tgt.ndim}维，不是3维！"

        C = img_tgt.shape[0]
        # 将空间维度(H, W)展平为1维，结果形状为(C, H*W)，每1行对应1个波段的所有像素。
        img_tgt = img_tgt.reshape(C, -1)
        img_fus = img_fus.reshape(C, -1)
        rmse = np.mean((img_tgt-img_fus)**2, axis=1)
        rmse = np.sqrt(rmse)
        mean = np.mean(img_tgt, axis=1)
        ergas = np.mean((rmse/mean)**2)
        ergas = 100 / scale * ergas**0.5

        return ergas


"""
SpatialDownsample类
　　该类实现空间下采样操作，利用给定的点扩散函数（PSF）作为卷积核，通过带步长的卷积对输入的多通道图像进行下采样。
"""
class SpatialDownsample(nn.Module):
    """
    __init__函数
    　　该函数为SpatialDownsample的构造函数，将psf转换为PyTorch张量，注册kernel为模块缓冲区并保存步长。

    psf                 点扩散函数（Gauss模糊核）
    downsample_factor   下采样因子
    """
    def __init__(self, psf, downsample_factor):
        super().__init__()

        kernel = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0)

        self.register_buffer("kernel", kernel)

        self.stride = downsample_factor
        k = kernel.shape[-1]
        self.pad_left = k // 2
        self.pad_right = k//2 - 1


    """
    forward函数
    　　该函数执行前向传播。
    
    x   输入张量（形状为B×C×H×W。）
    """
    def forward(self, x):
        B, C, H, W = x.shape
        kernel = self.kernel.repeat(C, 1, 1, 1)
        x_pad = self._symm_pad(x)
        out = F.conv2d(x_pad, kernel, stride=self.stride, groups=C)

        return out

    def _symm_pad(self, x):
        def pad_1d(t, pl, pr, dim):
            idx_l = [slice(None)] * t.ndim
            idx_l[dim] = slice(0, pl)
            left = t[tuple(idx_l)].flip(dims=[dim])
            sz = t.shape[dim]
            idx_r = [slice(None)] * t.ndim
            idx_r[dim] = slice(sz - pr, sz)
            right = t[tuple(idx_r)].flip(dims=[dim])

            return torch.cat([left, t, right], dim=dim)

        x = pad_1d(x, self.pad_left, self.pad_right, dim=3)  # W 方向
        x = pad_1d(x, self.pad_left, self.pad_right, dim=2)  # H 方向

        return x


def fspecial_gauss(kernel_size, sigma):
    m = n = (kernel_size - 1.0) / 2.0
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h = h / h.sum()
    return torch.from_numpy(h).float()


class BlurDownsample(nn.Module):
    """
    与train_DDL.py中的blur_down保持一致：先逐通道高斯模糊，再用双线性插值下采样。
    """
    def __init__(self, kernel_size=5, sigma=1.5, scale_factor=8, channels=3):
        super().__init__()
        self.scale_factor = scale_factor
        kernel = fspecial_gauss(kernel_size, sigma)
        kernel = kernel.expand(channels, 1, kernel.size(0), kernel.size(1))
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        x = F.conv2d(x, self.kernel, padding=self.kernel.size(-1) // 2, groups=x.shape[1])
        x = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear", align_corners=False)
        return x


"""
SpectralDownsample类
　　该类利用光谱响应矩阵对高光谱图像进行光谱下采样，生成多光谱图像。通过1×1卷积实现每个像素的光谱维线性组合。
"""
class SpectralDownsample(nn.Module):
    """
    __init__函数
    　　该函数为SpectralDownsample的构造函数，将光谱响应矩阵转换为PyTorch张量，注册kernel为模块缓冲区。

    R   光谱响应矩阵
    """
    def __init__(self, R):
        super().__init__()

        self.register_buffer("R", torch.from_numpy(R).float())


    """
    forward函数
    　　该函数执行前向传播。
    
    x   输入张量（形状为B×C_in×H×W。）
    """
    def forward(self, x):

        return torch.einsum("rc, bchw -> brhw", self.R, x)


class UpsampleBlur(nn.Module):
    """
    LearnableUpsampleBlur类
    　　该类保留train_DDL.py中的可学习空间拉回思想：先上采样，再用逐通道可学习模糊核进行抗锯齿平滑。
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1):
        super(UpsampleBlur, self).__init__()
        self.scale_factor = scale_factor
        self.channels = channels
        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        self.blur_kernel = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0).expand(
            channels, 1, kernel.size(0), kernel.size(1)
        ), requires_grad=True)

    def apply_upsample_blur(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return F.conv2d(x_up, weight=self.blur_kernel, padding=1, groups=self.channels)

    def forward(self, x):
        return self.apply_upsample_blur(x)


class ConstrainedKernelUpsampleBlur(nn.Module):
    """
    Spatial pullback with a weakly learnable blur kernel.

    The effective kernel is constrained near the legacy [1, 2, 1] separable
    kernel, so it can adapt slightly without drifting far from the physical
    pullback prior.
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1, alpha=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels
        self.alpha = float(alpha)

        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer("fixed_kernel", kernel)
        self.raw_kernel = nn.Parameter(torch.log(kernel.clamp_min(1e-6)))

    def current_kernel(self):
        b, c, h, w = self.raw_kernel.shape
        learned_kernel = torch.softmax(self.raw_kernel.view(b, c, -1), dim=-1).view(b, c, h, w)
        return (1.0 - self.alpha) * self.fixed_kernel + self.alpha * learned_kernel

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        kernel = self.current_kernel()
        return F.conv2d(x_up, weight=kernel, padding=kernel.size(-1) // 2, groups=self.channels)


class PSFMatchedUpsampleBlur(nn.Module):
    """
    Fixed Gaussian spatial pullback with a wider kernel than legacy UpsampleBlur.

    This tests whether the LRHSI pullback should be closer to the LRTN Gaussian
    degradation trend instead of the simple 3x3 [1, 2, 1] kernel.
    """
    def __init__(self, scale_factor=8, channels=1, kernel_size=5, sigma=1.5):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels

        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel / kernel.sum()
        self.register_buffer(
            "blur_kernel",
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
        )

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return F.conv2d(
            x_up,
            weight=self.blur_kernel,
            padding=self.blur_kernel.size(-1) // 2,
            groups=self.channels,
        )


class ResidualGatedUpsampleBlur(nn.Module):
    """
    A nonlinear spatial pullback that keeps the original upsample-blur path as
    the initialization point, then learns a gated residual refinement.
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels

        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        self.blur_kernel = nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
            requires_grad=True,
        )

        hidden_channels = channels * 2
        self.residual_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Start from the legacy UpsampleBlur behavior; the residual path is
        # learned only after gradients update this zero-initialized output conv.
        nn.init.zeros_(self.residual_refine[-1].weight)
        nn.init.zeros_(self.residual_refine[-1].bias)

    def apply_upsample_blur(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        blurred = F.conv2d(x_up, weight=self.blur_kernel, padding=1, groups=self.channels)
        residual = self.residual_refine(blurred)
        gate = self.gate(blurred)

        return blurred + gate * residual

    def forward(self, x):
        return self.apply_upsample_blur(x)


class DACGLiteUpsampleBlur(nn.Module):
    """
    DACG-lite spatial pullback.

    Adapted from DACG-IR Context_Gating_DualDomain_Modulation, MIT License.
    The full DACG-IR restoration network is not imported; this keeps only the
    lightweight idea of context-gated spatial/frequency residual modulation.
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels

        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        self.blur_kernel = nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
            requires_grad=True,
        )

        hidden_channels = channels * 2
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_out = nn.Conv2d(channels, channels, kernel_size=1)

        # Start exactly from legacy UpsampleBlur. The DACG-lite residual branch
        # becomes active only after this zero-initialized projection is updated.
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def apply_upsample_blur(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        blurred = F.conv2d(x_up, weight=self.blur_kernel, padding=1, groups=self.channels)

        context = self.context_branch(blurred)
        spatial_feat = self.spatial_branch(blurred)

        freq = torch.fft.rfft2(blurred, norm="ortho")
        freq_amp = torch.abs(freq)
        freq_gate = self.frequency_branch(freq_amp)
        freq_feat = torch.fft.irfft2(freq * freq_gate, s=blurred.shape[-2:], norm="ortho")

        fused = self.fusion(torch.cat([spatial_feat, freq_feat, context], dim=1))
        gate = self.gate(torch.cat([blurred, context], dim=1))
        residual = self.residual_out(fused)

        return blurred + gate * residual

    def forward(self, x):
        return self.apply_upsample_blur(x)


class RemoteSensingAdaptiveGatedFusion(nn.Module):
    """
    Adapted from DACG-IR Adaptive_Gated_Fusion, MIT License.

    The AGF structure is kept, while GroupNorm and channel reduction are
    adjusted for 31-band HSI features.
    """
    def __init__(self, channels):
        super().__init__()
        hidden_channels = channels // 2
        if hidden_channels < 8:
            hidden_channels = 8
        self.channels = channels
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, f_enc, f_dec):
        combined = torch.cat([f_enc, f_dec], dim=1)
        spatial_logit = self.spatial_gate(combined)

        b, c, _, _ = combined.shape
        channel_logit = self.channel_gate(self.avg_pool(combined).view(b, c))
        channel_logit = channel_logit.view(b, self.channels, 1, 1)

        atten_weight = torch.sigmoid(spatial_logit + channel_logit)
        f_enc_filtered = f_enc * atten_weight

        return self.fusion_conv(torch.cat([f_enc_filtered, f_dec], dim=1))


class RemoteSensingAGFUpsampleBlur(nn.Module):
    """
    Remote-sensing AGF spatial pullback.

    This module reuses the DACG-IR AGF information-gating structure and keeps
    the residual candidate branch lightweight for HSI spatial pullback.
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels

        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        self.blur_kernel = nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
            requires_grad=True,
        )

        hidden_channels = channels * 2
        self.residual_candidate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.agf = RemoteSensingAdaptiveGatedFusion(channels)
        self.residual_projection = nn.Conv2d(channels, channels, kernel_size=1)

        # Start exactly from legacy UpsampleBlur. The AGF path is learned only
        # after this zero-initialized residual projection is updated.
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def apply_upsample_blur(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        blurred = F.conv2d(x_up, weight=self.blur_kernel, padding=1, groups=self.channels)
        candidate = self.residual_candidate(blurred)
        fused = self.agf(candidate, blurred)
        residual = self.residual_projection(fused)

        return blurred + residual

    def forward(self, x):
        return self.apply_upsample_blur(x)


# Keep the old class name as an alias for compatibility with earlier checkpoints/code.
LearnableUpsampleBlur = UpsampleBlur


def spectral_transform(hsi, R, inverse=False):
    """
    Match train_DDL.py spectral_transform: project HSI to MSI, or pull MSI
    back to HSI with the pseudo-inverse of R when inverse=True.
    """
    B, C_in, H, W = hsi.shape
    device = hsi.device

    if not isinstance(R, torch.Tensor):
        R = torch.from_numpy(R).float().to(device)
    else:
        R = R.to(device)

    if inverse:
        R_inv = torch.pinverse(R)
        R_expanded = R_inv.view(1, -1, R.shape[0], 1, 1)
    else:
        R_expanded = R.view(1, -1, R.shape[1], 1, 1)

    hsi_expanded = hsi.unsqueeze(1)
    transformed = torch.sum(R_expanded * hsi_expanded, dim=2)
    return transformed



class LearnableSpatialPullback(nn.Module):
    """
    Ordinary learnable spatial pullback for the final DDL design.

    It keeps the legacy upsample-blur path as the initialization point and adds
    only a zero-initialized residual refinement. No AGF is used here.
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels

        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        self.blur_kernel = nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
            requires_grad=True,
        )

        hidden_channels = channels * 2
        self.residual_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.residual_refine[-1].weight)
        nn.init.zeros_(self.residual_refine[-1].bias)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        blurred = F.conv2d(x_up, weight=self.blur_kernel, padding=1, groups=self.channels)
        residual = self.residual_refine(blurred)

        return blurred + residual


class DegradationEdgeAttentionSpatialPullback(nn.Module):
    """
    Degradation-edge guided spatial pullback.

    The module keeps the original upsample-blur result as a stable base, then
    uses fixed Sobel/Laplacian edge cues and Gaussian residual cues to guide a
    lightweight local attention refinement.
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels

        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        self.blur_kernel = nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
            requires_grad=True,
        )
        self.register_buffer(
            "gaussian_kernel",
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
        )

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        sobel_y = sobel_x.t()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ) / 4.0
        self.register_buffer(
            "sobel_x",
            sobel_x.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
        )
        self.register_buffer(
            "laplacian",
            laplacian.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
        )

        hidden_channels = channels * 2
        attention_hidden = hidden_channels // 2
        if attention_hidden < 8:
            attention_hidden = 8

        self.local_embed = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, attention_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attention_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Start from the previous LearnableSpatialPullback behavior. The edge
        # attention path becomes active only after this projection is updated.
        nn.init.zeros_(self.residual_refine[-1].weight)
        nn.init.zeros_(self.residual_refine[-1].bias)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        blurred = F.conv2d(x_up, weight=self.blur_kernel, padding=1, groups=self.channels)

        low_pass = F.conv2d(x_up, weight=self.gaussian_kernel, padding=1, groups=self.channels)
        degradation_residual = x_up - low_pass

        edge_x = F.conv2d(x_up, weight=self.sobel_x, padding=1, groups=self.channels)
        edge_y = F.conv2d(x_up, weight=self.sobel_y, padding=1, groups=self.channels)
        edge_lap = F.conv2d(x_up, weight=self.laplacian, padding=1, groups=self.channels)
        edge_cue = torch.abs(edge_x) + torch.abs(edge_y) + 0.5 * torch.abs(edge_lap)

        features = self.local_embed(torch.cat([blurred, degradation_residual, edge_cue], dim=1))
        attention = self.channel_attention(features) * self.spatial_attention(features)
        residual = self.residual_refine(features * attention)

        return blurred + self.residual_scale * residual


class DegradationEdgeResidualSpatialPullback(nn.Module):
    """
    Lightweight degradation-edge residual spatial pullback.

    Compared with DegradationEdgeAttentionSpatialPullback, this module removes
    channel/spatial attention and keeps only Sobel edge cues plus a zero-init
    residual branch. It is used to test whether degradation-edge information is
    useful without introducing a heavy attention block.
    """
    def __init__(self, blur_kernel=[1, 2, 1], scale_factor=8, channels=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.channels = channels
        self.residual_scale = 0.05

        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.outer(kernel, kernel)
        self.blur_kernel = nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
            requires_grad=True,
        )

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        sobel_y = sobel_x.t()
        self.register_buffer(
            "sobel_x",
            sobel_x.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
        )

        hidden_channels = channels * 2
        self.residual_refine = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.residual_refine[-1].weight)
        nn.init.zeros_(self.residual_refine[-1].bias)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        blurred = F.conv2d(x_up, weight=self.blur_kernel, padding=1, groups=self.channels)

        edge_x = F.conv2d(x_up, weight=self.sobel_x, padding=1, groups=self.channels)
        edge_y = F.conv2d(x_up, weight=self.sobel_y, padding=1, groups=self.channels)
        edge_cue = torch.abs(edge_x) + torch.abs(edge_y)

        residual = self.residual_refine(torch.cat([blurred, edge_cue], dim=1))

        return blurred + self.residual_scale * residual


class LearnableSpectralPullback(nn.Module):
    """
    Learnable spectral pullback initialized from the physical pinv(R) mapping.

    The residual branch is zero-initialized, so the module starts exactly from
    spectral_transform(..., inverse=True), then learns a light spectral correction.
    """
    def __init__(self, R, hidden_channels=None):
        super().__init__()
        response = torch.as_tensor(R, dtype=torch.float32)
        self.register_buffer("R", response)
        channels = response.shape[1]
        hidden_channels = hidden_channels or channels * 2

        self.residual_refine = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.residual_refine[-1].weight)
        nn.init.zeros_(self.residual_refine[-1].bias)

    def forward(self, x):
        pinv_base = spectral_transform(x, self.R, inverse=True)
        residual = self.residual_refine(pinv_base)

        return pinv_base + residual


class RemoteSensingDualPullbackFusion(nn.Module):
    """
    Remote-sensing adapted single AGF for fusing two pullback HR-HSI estimates.

    This is not a direct copy of DACG-IR AGF. It adds spectral-angle and spatial
    gradient disagreement cues that fit the HSI-MSI dual-pullback setting.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        gate_in_channels = channels * 2 + 2
        hidden_channels = channels
        channel_hidden = channels // 2
        if channel_hidden < 8:
            channel_hidden = 8

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_gate = nn.Sequential(
            nn.Linear(gate_in_channels, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, channels),
        )
        self.residual_refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

        # Zero logits make gate=sigmoid(0)=0.5 at initialization.
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)
        nn.init.zeros_(self.spectral_gate[-1].weight)
        nn.init.zeros_(self.spectral_gate[-1].bias)
        nn.init.zeros_(self.residual_refine[-1].weight)
        nn.init.zeros_(self.residual_refine[-1].bias)

    @staticmethod
    def _spatial_gradient(x):
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        return grad_x, grad_y

    def _disagreement_cue(self, spatial_hr, spectral_hr):
        cosine = F.cosine_similarity(spatial_hr, spectral_hr, dim=1, eps=1e-6)
        cue_sam = (1.0 - cosine).unsqueeze(1).clamp(0.0, 2.0)

        spatial_grad_x, spatial_grad_y = self._spatial_gradient(spatial_hr)
        spectral_grad_x, spectral_grad_y = self._spatial_gradient(spectral_hr)
        cue_grad = (
            torch.abs(spatial_grad_x - spectral_grad_x)
            + torch.abs(spatial_grad_y - spectral_grad_y)
        ).mean(dim=1, keepdim=True)

        return torch.cat([cue_sam, cue_grad], dim=1), cue_sam, cue_grad

    def forward(self, spatial_hr, spectral_hr):
        fused_input = torch.cat([spatial_hr, spectral_hr], dim=1)
        disagreement_cue, cue_sam, cue_grad = self._disagreement_cue(spatial_hr, spectral_hr)
        gate_input = torch.cat([fused_input, disagreement_cue], dim=1)

        spatial_logit = self.spatial_gate(gate_input)
        b, c, _, _ = gate_input.shape
        spectral_logit = self.spectral_gate(self.avg_pool(gate_input).view(b, c))
        spectral_logit = spectral_logit.view(b, self.channels, 1, 1)
        gate = torch.sigmoid(spatial_logit + spectral_logit)

        residual = self.residual_refine(fused_input)
        fused_hr = gate * spatial_hr + (1.0 - gate) * spectral_hr + residual
        fusion_items = {
            "cue_sam_mean": cue_sam.detach().mean(),
            "cue_grad_mean": cue_grad.detach().mean(),
            "gate_mean": gate.detach().mean(),
            "gate_min": gate.detach().amin(),
            "gate_max": gate.detach().amax(),
        }

        return fused_hr, fusion_items


class DualLearningLoss(nn.Module):
    """Dual learning loss with legacy modes and final dual-pullback AGF mode."""
    def __init__(self, R, downsample_factor=8,
                 psf=None,
                 lambda_cycle_spatial=0.01,
                 lambda_cycle_spectral=0.001,
                 lambda_jac_spatial=0.0,
                 lambda_cycle_fused=0.0,
                 lambda_jac_spectral=0.0,
                 cycle_reliability_enable=False,
                 cycle_reliability_mode="target_average",
                 cycle_reliability_tau=2.0,
                 cycle_reliability_min=0.2,
                 cycle_reliability_normalize=True,
                 cycle_reliability_apply_to_branches=False,
                 observation_target_correction_enable=False,
                 observation_target_correction_eta=0.05,
                 observation_target_correction_clamp=True,
                 spectral_pullback_learnable=True,
                 pullback_mode="legacy"):
        super().__init__()
        response = torch.from_numpy(R).float()
        self.register_buffer("R", response)
        self.register_buffer("R_pinv", torch.linalg.pinv(response))
        if psf is None:
            psf = fspecial("gaussian", downsample_factor, 3)
        self.reliability_spatial_down = SpatialDownsample(psf, downsample_factor)
        self.pullback_mode = pullback_mode
        self.use_dual_pullback_fusion = pullback_mode in (
            "dual_pullback_agf_v1",
            "dual_pullback_edge_attention_v1",
            "dual_pullback_edge_residual_v1",
            "dual_pullback_legacy_spatial_v1",
            "dual_pullback_fixed_legacy_spatial_v1",
            "dual_pullback_constrained_kernel_v1",
            "dual_pullback_psf_matched_v1",
        )
        self.spectral_pullback_learnable = bool(spectral_pullback_learnable)
        self.spectral_pullback = None
        self.pullback_fusion = None

        if pullback_mode == "legacy":
            self.upsample_blur = UpsampleBlur(scale_factor=downsample_factor, channels=response.shape[1])
        elif pullback_mode == "residual_gate_v1":
            self.upsample_blur = ResidualGatedUpsampleBlur(scale_factor=downsample_factor, channels=response.shape[1])
        elif pullback_mode == "dacg_lite_v1":
            self.upsample_blur = DACGLiteUpsampleBlur(scale_factor=downsample_factor, channels=response.shape[1])
        elif pullback_mode == "rs_agf_v1":
            self.upsample_blur = RemoteSensingAGFUpsampleBlur(scale_factor=downsample_factor, channels=response.shape[1])
        elif pullback_mode == "dual_pullback_agf_v1":
            self.upsample_blur = LearnableSpatialPullback(scale_factor=downsample_factor, channels=response.shape[1])
            # 018 消融：光谱拉回可固定为 pinv(R)，用于验证可学习光谱 residual 是否引入后期漂移。
            if self.spectral_pullback_learnable:
                self.spectral_pullback = LearnableSpectralPullback(response)
            self.pullback_fusion = RemoteSensingDualPullbackFusion(response.shape[1])
        elif pullback_mode == "dual_pullback_edge_attention_v1":
            self.upsample_blur = DegradationEdgeAttentionSpatialPullback(
                scale_factor=downsample_factor,
                channels=response.shape[1],
            )
            if self.spectral_pullback_learnable:
                self.spectral_pullback = LearnableSpectralPullback(response)
            self.pullback_fusion = RemoteSensingDualPullbackFusion(response.shape[1])
        elif pullback_mode == "dual_pullback_edge_residual_v1":
            self.upsample_blur = DegradationEdgeResidualSpatialPullback(
                scale_factor=downsample_factor,
                channels=response.shape[1],
            )
            if self.spectral_pullback_learnable:
                self.spectral_pullback = LearnableSpectralPullback(response)
            self.pullback_fusion = RemoteSensingDualPullbackFusion(response.shape[1])
        elif pullback_mode == "dual_pullback_legacy_spatial_v1":
            self.upsample_blur = UpsampleBlur(scale_factor=downsample_factor, channels=response.shape[1])
            if self.spectral_pullback_learnable:
                self.spectral_pullback = LearnableSpectralPullback(response)
            self.pullback_fusion = RemoteSensingDualPullbackFusion(response.shape[1])
        elif pullback_mode == "dual_pullback_fixed_legacy_spatial_v1":
            self.upsample_blur = UpsampleBlur(scale_factor=downsample_factor, channels=response.shape[1])
            self.upsample_blur.blur_kernel.requires_grad_(False)
            if self.spectral_pullback_learnable:
                self.spectral_pullback = LearnableSpectralPullback(response)
            self.pullback_fusion = RemoteSensingDualPullbackFusion(response.shape[1])
        elif pullback_mode == "dual_pullback_constrained_kernel_v1":
            self.upsample_blur = ConstrainedKernelUpsampleBlur(
                scale_factor=downsample_factor,
                channels=response.shape[1],
                alpha=0.1,
            )
            if self.spectral_pullback_learnable:
                self.spectral_pullback = LearnableSpectralPullback(response)
            self.pullback_fusion = RemoteSensingDualPullbackFusion(response.shape[1])
        elif pullback_mode == "dual_pullback_psf_matched_v1":
            self.upsample_blur = PSFMatchedUpsampleBlur(
                scale_factor=downsample_factor,
                channels=response.shape[1],
                kernel_size=5,
                sigma=1.5,
            )
            if self.spectral_pullback_learnable:
                self.spectral_pullback = LearnableSpectralPullback(response)
            self.pullback_fusion = RemoteSensingDualPullbackFusion(response.shape[1])
        else:
            raise ValueError(f"Unsupported pullback_mode: {pullback_mode}")

        # Previous DDL internal and global weighting mechanisms are kept as
        # comments for traceability, but the final design uses fixed weights.
        # self.weights = nn.Parameter(torch.ones(4))
        self.weights = nn.Parameter(torch.ones(4), requires_grad=False)
        # self.global_loss_weights = nn.Parameter(torch.log(torch.tensor([0.8, 0.2])))
        # self.raw_lambda_ddl = nn.Parameter(initial_raw_lambda_ddl)
        # self.raw_lambda_cycle = nn.Parameter(...)
        # self.raw_lambda_jac = nn.Parameter(...)

        self.register_buffer("lambda_cycle_spatial", torch.tensor(lambda_cycle_spatial, dtype=torch.float32))
        self.register_buffer("lambda_cycle_spectral", torch.tensor(lambda_cycle_spectral, dtype=torch.float32))
        self.register_buffer("lambda_cycle_fused", torch.tensor(lambda_cycle_fused, dtype=torch.float32))
        self.register_buffer("lambda_jac", torch.tensor(lambda_jac_spatial, dtype=torch.float32))
        self.register_buffer("lambda_jac_spatial", torch.tensor(lambda_jac_spatial, dtype=torch.float32))
        self.register_buffer("lambda_jac_spectral", torch.tensor(lambda_jac_spectral, dtype=torch.float32))
        self.cycle_reliability_enable = bool(cycle_reliability_enable)
        self.cycle_reliability_mode = str(cycle_reliability_mode)
        self.cycle_reliability_tau = float(cycle_reliability_tau)
        self.cycle_reliability_min = float(cycle_reliability_min)
        self.cycle_reliability_normalize = bool(cycle_reliability_normalize)
        self.cycle_reliability_apply_to_branches = bool(cycle_reliability_apply_to_branches)
        self.observation_target_correction_enable = bool(observation_target_correction_enable)
        self.observation_target_correction_eta = float(observation_target_correction_eta)
        self.observation_target_correction_clamp = bool(observation_target_correction_clamp)
        self.loss_func = nn.L1Loss(reduction="mean")

    def get_global_loss_weights(self):
        return None

    def get_lambda_ddl(self):
        return None

    def get_cycle_jac_lambdas(self):
        return (
            self.lambda_cycle_spatial,
            self.lambda_cycle_spectral,
            self.lambda_cycle_fused,
            self.lambda_jac_spatial,
            self.lambda_jac_spectral,
        )

    @staticmethod
    def _zero_items(reference):
        return {
            "cue_sam_mean": reference.new_tensor(0.0),
            "cue_grad_mean": reference.new_tensor(0.0),
            "gate_mean": reference.new_tensor(0.0),
            "gate_min": reference.new_tensor(0.0),
            "gate_max": reference.new_tensor(0.0),
            "reliability_weight_mean": reference.new_tensor(1.0),
            "reliability_weight_min": reference.new_tensor(1.0),
            "reliability_weight_max": reference.new_tensor(1.0),
            "spatial_reliability_weight_mean": reference.new_tensor(1.0),
            "spatial_reliability_weight_min": reference.new_tensor(1.0),
            "spatial_reliability_weight_max": reference.new_tensor(1.0),
            "spectral_reliability_weight_mean": reference.new_tensor(1.0),
            "spectral_reliability_weight_min": reference.new_tensor(1.0),
            "spectral_reliability_weight_max": reference.new_tensor(1.0),
            "obs_lr_error_mean": reference.new_tensor(0.0),
            "obs_ms_error_mean": reference.new_tensor(0.0),
            "cross_spatial_ms_error_mean": reference.new_tensor(0.0),
            "cross_spectral_lr_error_mean": reference.new_tensor(0.0),
            "target_corr_delta_spatial_mean": reference.new_tensor(0.0),
            "target_corr_delta_spectral_mean": reference.new_tensor(0.0),
            "target_corr_shift_mean": reference.new_tensor(0.0),
            "target_corr_shift_max": reference.new_tensor(0.0),
            "target_corr_eta": reference.new_tensor(0.0),
        }

    def _normalize_observation_error(self, error):
        eps = 1e-6
        if not self.cycle_reliability_normalize:
            return error
        denom = error.detach().mean(dim=(2, 3), keepdim=True) + eps
        return error / denom

    @staticmethod
    def _robust_normalize_observation_error(error):
        eps = 1e-6
        flat = error.detach().flatten(start_dim=2)
        median = flat.median(dim=2).values.view(error.size(0), error.size(1), 1, 1)
        mad = torch.abs(error.detach() - median).flatten(start_dim=2).median(dim=2).values
        mad = mad.view(error.size(0), error.size(1), 1, 1)
        scale = median + mad + eps
        return torch.clamp(error / scale, min=0.0, max=6.0)

    @staticmethod
    def _sobel_error(pred, target):
        pred_x, pred_y = sobel_gradient(pred.detach())
        target_x, target_y = sobel_gradient(target.detach())
        return (torch.abs(pred_x - target_x) + torch.abs(pred_y - target_y)).mean(dim=1, keepdim=True)

    @staticmethod
    def _haar_high_frequency(x):
        h_even = x.size(2) - (x.size(2) % 2)
        w_even = x.size(3) - (x.size(3) % 2)
        x = x[:, :, :h_even, :w_even]
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return lh, hl, hh

    @classmethod
    def _haar_error(cls, pred, target):
        pred_lh, pred_hl, pred_hh = cls._haar_high_frequency(pred.detach())
        target_lh, target_hl, target_hh = cls._haar_high_frequency(target.detach())
        error = torch.abs(pred_lh - target_lh) + torch.abs(pred_hl - target_hl) + torch.abs(pred_hh - target_hh)
        return error.mean(dim=1, keepdim=True)

    def observation_reliability_weight(self, output_lrhsi, output_hrmsi, lr_hsi, hr_msi, reference):
        """根据真实观测域残差估计 cycle target 的可靠性权重。"""
        if not self.cycle_reliability_enable:
            weight = reference.new_ones(reference.size(0), 1, reference.size(2), reference.size(3))
            return weight, {
                "reliability_weight_mean": weight.mean().detach(),
                "reliability_weight_min": weight.amin().detach(),
                "reliability_weight_max": weight.amax().detach(),
                "spatial_reliability_weight_mean": weight.mean().detach(),
                "spatial_reliability_weight_min": weight.amin().detach(),
                "spatial_reliability_weight_max": weight.amax().detach(),
                "spectral_reliability_weight_mean": weight.mean().detach(),
                "spectral_reliability_weight_min": weight.amin().detach(),
                "spectral_reliability_weight_max": weight.amax().detach(),
                "obs_lr_error_mean": reference.new_tensor(0.0),
                "obs_ms_error_mean": reference.new_tensor(0.0),
                "cross_spatial_ms_error_mean": reference.new_tensor(0.0),
                "cross_spectral_lr_error_mean": reference.new_tensor(0.0),
            }

        eps = 1e-6
        lr_error = torch.mean(torch.abs(output_lrhsi.detach() - lr_hsi), dim=1, keepdim=True)
        lr_error_hr = F.interpolate(
            lr_error,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        ms_error = torch.mean(torch.abs(output_hrmsi.detach() - hr_msi), dim=1, keepdim=True)

        lr_error_for_weight = self._normalize_observation_error(lr_error_hr)
        ms_error_for_weight = self._normalize_observation_error(ms_error)
        observation_error = 0.5 * (lr_error_for_weight + ms_error_for_weight)

        if self.cycle_reliability_mode == "robust_observation":
            observation_error = self._robust_normalize_observation_error(observation_error)
        elif self.cycle_reliability_mode == "structure_observation":
            lr_structure = self._sobel_error(output_lrhsi, lr_hsi)
            lr_structure_hr = F.interpolate(
                lr_structure,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            ms_structure = self._sobel_error(output_hrmsi, hr_msi)
            structure_error = 0.5 * (
                self._normalize_observation_error(lr_structure_hr)
                + self._normalize_observation_error(ms_structure)
            )
            observation_error = observation_error + structure_error
        elif self.cycle_reliability_mode == "frequency_observation":
            lr_hf = self._haar_error(output_lrhsi, lr_hsi)
            lr_hf_hr = F.interpolate(
                lr_hf,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            ms_hf = self._haar_error(output_hrmsi, hr_msi)
            ms_hf_hr = F.interpolate(
                ms_hf,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            frequency_error = 0.5 * (
                self._normalize_observation_error(lr_hf_hr)
                + self._normalize_observation_error(ms_hf_hr)
            )
            observation_error = observation_error + frequency_error

        reliability = torch.exp(-self.cycle_reliability_tau * observation_error)
        reliability = self.cycle_reliability_min + (1.0 - self.cycle_reliability_min) * reliability

        if self.cycle_reliability_normalize:
            reliability = reliability / (reliability.detach().mean(dim=(2, 3), keepdim=True) + eps)

        reliability = reliability.detach()
        return reliability, {
            "reliability_weight_mean": reliability.mean().detach(),
            "reliability_weight_min": reliability.amin().detach(),
            "reliability_weight_max": reliability.amax().detach(),
            "spatial_reliability_weight_mean": reliability.mean().detach(),
            "spatial_reliability_weight_min": reliability.amin().detach(),
            "spatial_reliability_weight_max": reliability.amax().detach(),
            "spectral_reliability_weight_mean": reliability.mean().detach(),
            "spectral_reliability_weight_min": reliability.amin().detach(),
            "spectral_reliability_weight_max": reliability.amax().detach(),
            "obs_lr_error_mean": lr_error_hr.mean().detach(),
            "obs_ms_error_mean": ms_error.mean().detach(),
            "cross_spatial_ms_error_mean": reference.new_tensor(0.0),
            "cross_spectral_lr_error_mean": reference.new_tensor(0.0),
        }

    def _reliability_from_error(self, error):
        eps = 1e-6
        error_for_weight = error
        if self.cycle_reliability_normalize:
            denom = error.detach().mean(dim=(2, 3), keepdim=True) + eps
            error_for_weight = error / denom

        reliability = torch.exp(-self.cycle_reliability_tau * error_for_weight)
        reliability = self.cycle_reliability_min + (1.0 - self.cycle_reliability_min) * reliability

        if self.cycle_reliability_normalize:
            reliability = reliability / (reliability.detach().mean(dim=(2, 3), keepdim=True) + eps)

        return reliability.detach()

    def cross_observation_reliability_weight(self, reconstructed_hr_spatial, reconstructed_hr_spectral,
                                             lr_hsi, hr_msi, reference):
        """Estimate each pullback branch reliability using its complementary observation."""
        if not self.cycle_reliability_enable:
            weight = reference.new_ones(reference.size(0), 1, reference.size(2), reference.size(3))
            return weight, weight, {
                "reliability_weight_mean": weight.mean().detach(),
                "reliability_weight_min": weight.amin().detach(),
                "reliability_weight_max": weight.amax().detach(),
                "spatial_reliability_weight_mean": weight.mean().detach(),
                "spatial_reliability_weight_min": weight.amin().detach(),
                "spatial_reliability_weight_max": weight.amax().detach(),
                "spectral_reliability_weight_mean": weight.mean().detach(),
                "spectral_reliability_weight_min": weight.amin().detach(),
                "spectral_reliability_weight_max": weight.amax().detach(),
                "obs_lr_error_mean": reference.new_tensor(0.0),
                "obs_ms_error_mean": reference.new_tensor(0.0),
                "cross_spatial_ms_error_mean": reference.new_tensor(0.0),
                "cross_spectral_lr_error_mean": reference.new_tensor(0.0),
            }

        spatial_hrmsi = spectral_transform(reconstructed_hr_spatial.detach(), self.R, inverse=False)
        spatial_ms_error = torch.mean(torch.abs(spatial_hrmsi - hr_msi), dim=1, keepdim=True)

        spectral_lrhsi = self.reliability_spatial_down(reconstructed_hr_spectral.detach())
        spectral_lr_error = torch.mean(torch.abs(spectral_lrhsi - lr_hsi), dim=1, keepdim=True)
        spectral_lr_error_hr = F.interpolate(
            spectral_lr_error,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        spatial_weight = self._reliability_from_error(spatial_ms_error)
        spectral_weight = self._reliability_from_error(spectral_lr_error_hr)
        combined_weight = 0.5 * (spatial_weight + spectral_weight)

        return spatial_weight, spectral_weight, {
            "reliability_weight_mean": combined_weight.mean().detach(),
            "reliability_weight_min": combined_weight.amin().detach(),
            "reliability_weight_max": combined_weight.amax().detach(),
            "spatial_reliability_weight_mean": spatial_weight.mean().detach(),
            "spatial_reliability_weight_min": spatial_weight.amin().detach(),
            "spatial_reliability_weight_max": spatial_weight.amax().detach(),
            "spectral_reliability_weight_mean": spectral_weight.mean().detach(),
            "spectral_reliability_weight_min": spectral_weight.amin().detach(),
            "spectral_reliability_weight_max": spectral_weight.amax().detach(),
            "obs_lr_error_mean": reference.new_tensor(0.0),
            "obs_ms_error_mean": reference.new_tensor(0.0),
            "cross_spatial_ms_error_mean": spatial_ms_error.mean().detach(),
            "cross_spectral_lr_error_mean": spectral_lr_error_hr.mean().detach(),
        }

    def observation_target_correction(self, target_hrhsi, output_lrhsi, output_hrmsi, lr_hsi, hr_msi):
        """用真实观测残差反投影轻微校正 cycle target。"""
        zero = target_hrhsi.new_tensor(0.0)
        eta_tensor = target_hrhsi.new_tensor(self.observation_target_correction_eta)
        if not self.observation_target_correction_enable or self.observation_target_correction_eta == 0.0:
            return target_hrhsi, {
                "target_corr_delta_spatial_mean": zero,
                "target_corr_delta_spectral_mean": zero,
                "target_corr_shift_mean": zero,
                "target_corr_shift_max": zero,
                "target_corr_eta": zero,
            }

        res_lr = lr_hsi.detach() - output_lrhsi.detach()
        res_ms = hr_msi.detach() - output_hrmsi.detach()

        delta_spatial = F.interpolate(
            res_lr,
            size=target_hrhsi.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        delta_spectral = torch.einsum("cr,brhw->bchw", self.R_pinv.to(res_ms.device), res_ms)
        correction = 0.5 * (delta_spatial + delta_spectral)
        target_corr = target_hrhsi + self.observation_target_correction_eta * correction
        if self.observation_target_correction_clamp:
            target_corr = target_corr.clamp(0.0, 1.0)

        shift = torch.abs(target_corr - target_hrhsi)
        return target_corr.detach(), {
            "target_corr_delta_spatial_mean": torch.abs(delta_spatial).mean().detach(),
            "target_corr_delta_spectral_mean": torch.abs(delta_spectral).mean().detach(),
            "target_corr_shift_mean": shift.mean().detach(),
            "target_corr_shift_max": shift.amax().detach(),
            "target_corr_eta": eta_tensor.detach(),
        }

    def forward(self, output_hrhsi, output_lrhsi, output_hrmsi, lr_hsi, hr_msi, compute_jac=True):
        # Previous min-softmax weights and DDL L1 consistency are excluded to
        # avoid duplicating the base degradation consistency loss.
        target_hrhsi = output_hrhsi.detach()
        zero = output_hrhsi.new_tensor(0.0)
        fusion_items = self._zero_items(output_hrhsi)
        reliability_items = {
            "reliability_weight_mean": output_hrhsi.new_tensor(1.0),
            "reliability_weight_min": output_hrhsi.new_tensor(1.0),
            "reliability_weight_max": output_hrhsi.new_tensor(1.0),
            "spatial_reliability_weight_mean": output_hrhsi.new_tensor(1.0),
            "spatial_reliability_weight_min": output_hrhsi.new_tensor(1.0),
            "spatial_reliability_weight_max": output_hrhsi.new_tensor(1.0),
            "spectral_reliability_weight_mean": output_hrhsi.new_tensor(1.0),
            "spectral_reliability_weight_min": output_hrhsi.new_tensor(1.0),
            "spectral_reliability_weight_max": output_hrhsi.new_tensor(1.0),
            "obs_lr_error_mean": output_hrhsi.new_tensor(0.0),
            "obs_ms_error_mean": output_hrhsi.new_tensor(0.0),
            "cross_spatial_ms_error_mean": output_hrhsi.new_tensor(0.0),
            "cross_spectral_lr_error_mean": output_hrhsi.new_tensor(0.0),
        }
        target_corr, target_corr_items = self.observation_target_correction(
            target_hrhsi,
            output_lrhsi,
            output_hrmsi,
            lr_hsi,
            hr_msi,
        )

        if self.use_dual_pullback_fusion:
            reconstructed_hr_spatial = self.upsample_blur(output_lrhsi)
            if self.spectral_pullback is None:
                reconstructed_hr_spectral = spectral_transform(output_hrmsi, self.R, inverse=True)
            else:
                reconstructed_hr_spectral = self.spectral_pullback(output_hrmsi)
            lambda_cycle_spatial, lambda_cycle_spectral, lambda_cycle_fused, lambda_jac_spatial, lambda_jac_spectral = (
                self.get_cycle_jac_lambdas()
            )

            if self.cycle_reliability_mode == "cross_observation":
                spatial_reliability_weight, spectral_reliability_weight, reliability_items = (
                    self.cross_observation_reliability_weight(
                        reconstructed_hr_spatial,
                        reconstructed_hr_spectral,
                        lr_hsi,
                        hr_msi,
                        target_hrhsi,
                    )
                )
                reliability_weight = 0.5 * (spatial_reliability_weight + spectral_reliability_weight)
            else:
                reliability_weight, reliability_items = self.observation_reliability_weight(
                    output_lrhsi,
                    output_hrmsi,
                    lr_hsi,
                    hr_msi,
                    target_hrhsi,
                )
                spatial_reliability_weight = reliability_weight
                spectral_reliability_weight = reliability_weight
            spatial_cycle_abs = torch.abs(reconstructed_hr_spatial - target_hrhsi)
            spectral_cycle_abs = torch.abs(reconstructed_hr_spectral - target_hrhsi)
            spatial_cycle_abs_for_loss = torch.abs(reconstructed_hr_spatial - target_corr)
            spectral_cycle_abs_for_loss = torch.abs(reconstructed_hr_spectral - target_corr)

            if self.cycle_reliability_apply_to_branches:
                loss_cycle_spatial_log = torch.mean(spatial_reliability_weight * spatial_cycle_abs_for_loss)
                loss_cycle_spectral_log = torch.mean(spectral_reliability_weight * spectral_cycle_abs_for_loss)
            else:
                loss_cycle_spatial_log = torch.mean(spatial_cycle_abs_for_loss)
                loss_cycle_spectral_log = torch.mean(spectral_cycle_abs_for_loss)

            # 013 消融实验：lambda_cycle_fused=0 时跳过 AGF 前向，避免 fused cycle
            # 和 AGF 参数继续影响训练。旧的 fused cycle 路径仍保留，便于恢复 011/010。
            if lambda_cycle_fused.detach().abs().item() > 0:
                fused_hr, fusion_items = self.pullback_fusion(reconstructed_hr_spatial, reconstructed_hr_spectral)
                fused_cycle_abs = torch.abs(fused_hr - target_corr)
                loss_cycle_fused = torch.mean(reliability_weight * fused_cycle_abs)
            else:
                loss_cycle_fused = zero

            loss_cycle = loss_cycle_fused + loss_cycle_spatial_log + loss_cycle_spectral_log

            if compute_jac and lambda_jac_spatial.detach().abs().item() > 0:
                jac_spatial = self.jacobian_regularizer(output_lrhsi, reconstructed_hr_spatial)
            else:
                jac_spatial = zero
            if compute_jac and lambda_jac_spectral.detach().abs().item() > 0:
                jac_spectral = self.jacobian_regularizer(output_hrmsi, reconstructed_hr_spectral)
            else:
                jac_spectral = zero
            jac_loss = jac_spatial + jac_spectral

            cycle_spatial_eff = lambda_cycle_spatial * loss_cycle_spatial_log
            cycle_spectral_eff = lambda_cycle_spectral * loss_cycle_spectral_log
            cycle_fused_eff = lambda_cycle_fused * loss_cycle_fused
            cycle_eff = cycle_fused_eff + cycle_spatial_eff + cycle_spectral_eff
            jac_spatial_eff = lambda_jac_spatial * jac_spatial
            jac_spectral_eff = lambda_jac_spectral * jac_spectral
            jac_eff = jac_spatial_eff + jac_spectral_eff
            loss_ddl = cycle_eff + jac_eff
        else:
            # Legacy modes keep the old two-branch cycle losses unchanged.
            reconstructed_hr_spatial = self.upsample_blur(output_lrhsi)
            reconstructed_hr_spectral = spectral_transform(output_hrmsi, self.R, inverse=True)
            loss_cycle_spatial_log = self.loss_func(reconstructed_hr_spatial, target_corr)
            loss_cycle_spectral_log = self.loss_func(reconstructed_hr_spectral, target_corr)
            loss_cycle_fused = zero
            loss_cycle = loss_cycle_spatial_log + loss_cycle_spectral_log

            lambda_cycle_spatial, lambda_cycle_spectral, lambda_cycle_fused, lambda_jac_spatial, lambda_jac_spectral = (
                self.get_cycle_jac_lambdas()
            )

            # Fixed degradation Jacobians are constant. In legacy modes this
            # Jacobian constrains only the learnable spatial pullback.
            if compute_jac and lambda_jac_spatial.detach().abs().item() > 0:
                jac_spatial = self.jacobian_regularizer(output_lrhsi, reconstructed_hr_spatial)
            else:
                jac_spatial = zero
            jac_spectral = zero
            jac_loss = jac_spatial + jac_spectral

            cycle_spatial_eff = lambda_cycle_spatial * loss_cycle_spatial_log
            cycle_spectral_eff = lambda_cycle_spectral * loss_cycle_spectral_log
            cycle_fused_eff = zero
            cycle_eff = cycle_spatial_eff + cycle_spectral_eff
            jac_spatial_eff = lambda_jac_spatial * jac_spatial
            jac_spectral_eff = zero
            jac_eff = jac_spatial_eff + jac_spectral_eff
            loss_ddl = cycle_eff + jac_eff

        loss_items = {
            "ddl": loss_ddl.detach(),
            "cycle_spatial": loss_cycle_spatial_log.detach(),
            "cycle_spectral": loss_cycle_spectral_log.detach(),
            "cycle": loss_cycle.detach(),
            "cycle_fused": loss_cycle_fused.detach(),
            "cycle_spatial_log": loss_cycle_spatial_log.detach(),
            "cycle_spectral_log": loss_cycle_spectral_log.detach(),
            "jac": jac_loss.detach(),
            "jac_spatial": jac_spatial.detach(),
            "jac_spectral": jac_spectral.detach(),
            "lambda_cycle_spatial": lambda_cycle_spatial.detach(),
            "lambda_cycle_spectral": lambda_cycle_spectral.detach(),
            "lambda_cycle_fused": lambda_cycle_fused.detach(),
            "lambda_jac": lambda_jac_spatial.detach(),
            "lambda_jac_spatial": lambda_jac_spatial.detach(),
            "lambda_jac_spectral": lambda_jac_spectral.detach(),
            "cycle_spatial_eff": cycle_spatial_eff.detach(),
            "cycle_spectral_eff": cycle_spectral_eff.detach(),
            "cycle_fused_eff": cycle_fused_eff.detach(),
            "cycle_eff": cycle_eff.detach(),
            "jac_spatial_eff": jac_spatial_eff.detach(),
            "jac_spectral_eff": jac_spectral_eff.detach(),
            "jac_eff": jac_eff.detach(),
            "cue_sam_mean": fusion_items["cue_sam_mean"],
            "cue_grad_mean": fusion_items["cue_grad_mean"],
            "gate_mean": fusion_items["gate_mean"],
            "gate_min": fusion_items["gate_min"],
            "gate_max": fusion_items["gate_max"],
            "reliability_weight_mean": reliability_items["reliability_weight_mean"],
            "reliability_weight_min": reliability_items["reliability_weight_min"],
            "reliability_weight_max": reliability_items["reliability_weight_max"],
            "spatial_reliability_weight_mean": reliability_items["spatial_reliability_weight_mean"],
            "spatial_reliability_weight_min": reliability_items["spatial_reliability_weight_min"],
            "spatial_reliability_weight_max": reliability_items["spatial_reliability_weight_max"],
            "spectral_reliability_weight_mean": reliability_items["spectral_reliability_weight_mean"],
            "spectral_reliability_weight_min": reliability_items["spectral_reliability_weight_min"],
            "spectral_reliability_weight_max": reliability_items["spectral_reliability_weight_max"],
            "obs_lr_error_mean": reliability_items["obs_lr_error_mean"],
            "obs_ms_error_mean": reliability_items["obs_ms_error_mean"],
            "cross_spatial_ms_error_mean": reliability_items["cross_spatial_ms_error_mean"],
            "cross_spectral_lr_error_mean": reliability_items["cross_spectral_lr_error_mean"],
            "target_corr_delta_spatial_mean": target_corr_items["target_corr_delta_spatial_mean"],
            "target_corr_delta_spectral_mean": target_corr_items["target_corr_delta_spectral_mean"],
            "target_corr_shift_mean": target_corr_items["target_corr_shift_mean"],
            "target_corr_shift_max": target_corr_items["target_corr_shift_max"],
            "target_corr_eta": target_corr_items["target_corr_eta"],
        }

        return loss_ddl, loss_items

    @staticmethod
    def jacobian_regularizer(x, y):
        # Original train_DDL.py used y.sum(), but LR->HR pullback has many more
        # output pixels, so the unnormalized Jacobian term can dominate loss.
        # grad_y_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        normalizer = float(y[0].numel()) ** 0.5
        grad_y_x = torch.autograd.grad(
            y.sum() / normalizer,
            x,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_y_x is None:
            return x.new_tensor(0.0)

        return grad_y_x.pow(2).mean()
