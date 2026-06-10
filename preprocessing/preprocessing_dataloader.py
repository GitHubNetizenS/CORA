"""
preprocessing_dataloader.py文件
　　该Python文件定义了多个数据集类，用于加载不同来源的高光谱图像数据，并通过光谱响应矩阵和Gauss模糊下采样生成低分辨率高光谱图像与高分辨率多光谱图像，
最后以滑动窗口方式裁剪成固定大小的训练样本对。
"""
import scipy.io as sio
from torch.utils.data       import Dataset
from preprocessing_utils    import *


"""
HSIDataProcess类
　　该类从高光谱数据集的.mat文件中读取高分辨率高光谱图像（HRHSI），通过光谱响应矩阵和Gauss模糊下采样生成高分辨率多光谱图像（HRMSI）和低分辨率高光
谱图像（LRHSI），然后以滑动窗口方式裁剪成固定大小的图像块，构建训练所需的样本对。
"""
class HSIDataProcess(Dataset):
    """
    __init__函数
    　　初始化数据集对象，读取指定数量的数据集场景，生成退化图像并裁剪块。

    path                包含数据集.mat文件的目录路径
    R                   光谱响应矩阵（形状为(3, 31)，用于从HRHSI下采样到HRMSI。）
    training_size       裁剪块的空间尺寸
    stride              滑动窗口的步长
    downsample_factor   空间下采样因子
    PSF                 点扩散函数（Gauss模糊核，用于从HRHSI下采样到LRHSI。）
    num                 要读取的场景数量（如CAVE共32个场景，通常取前20或全部）
    """
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(0, num, 1):
            data_path = os.path.join(path, imglist[i])
            img = sio.loadmat(data_path)
            # 读取标签为“b”的图像，也即高分辨率高光谱图像（HRHSI），并对其值归一化为[0, 1]。
            # Havard数据集的标签为“ref”，ICVL数据集的标签为“HSI”，某些CAVE数据集的标签为“b”或“hsi”。
            # 此处采用CAVE数据集，标签为“hsi”。
            img1 = img["hsi"]
            # img1 = img["ref"]
            img1 = img1 / img1.max()
            # HRHSI尺寸为（C, H, W），HSI_LR尺寸为（C, h, w），MSI_HR尺寸为（c, H, W）。
            # 以CAVE数据集为例，HRHSI(31, 512, 512)，MSI_HR(3, 512, 512)，HSI_LR(31, 64, 64)。
            HRHSI = np.transpose(img1, (2, 0, 1))
            MSI_HR = np.tensordot(R, HRHSI, axes=([1], [0]))
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)

            # print(HRHSI.shape, MSI_HR.shape, HSI_LR.shape)
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[::1, j:j+training_size:1,
                                           k:k+training_size:1]
                    temp_hrms = MSI_HR[::1, j:j+training_size:1,
                                            k:k+training_size:1]
                    temp_lrhs = HSI_LR[::1, int(j/downsample_factor):int((j+training_size)/downsample_factor):1,
                                            int(k/downsample_factor):int((k+training_size)/downsample_factor):1]
                    # 　　以CAVE数据集为例，当training_size=64、downsample_factor=8时，HRHSI(31, 64, 64)，MSI_HR(3, 64, 64),
                    # HSI_LR(31, 8, 8)。

                    # print(temp_hrhs.shape, temp_hrms.shape, temp_lrhs.shape)

                    temp_hrhs = temp_hrhs.astype(np.float32)
                    temp_hrms = temp_hrms.astype(np.float32)
                    temp_lrhs = temp_lrhs.astype(np.float32)

                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # 　　以CAVE数据集为例，当training_size=64、downsample_factor=8时，HRHSI(225, 31, 64, 64)，MSI_HR(225, 3, 64, 64),
        # HSI_LR(255, 31, 8, 8)。
        # print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)

        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    """
    __getitem__函数
    　　该函数返回指定索引的样本对。

    index   样本索引    
    """
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, ::1, ::1, ::1]
        train_hrms = self.train_hrms_all[index, ::1, ::1, ::1]
        train_lrhs = self.train_lrhs_all[index, ::1, ::1, ::1]

        # 　　以CAVE数据集为例，当training_size=64、downsample_factor=8时，HRHSI(31, 64, 64)，MSI_HR(3, 64, 64),
        # HSI_LR(31, 8, 8)。
        # print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)

        return train_hrhs, train_hrms, train_lrhs

    """
    __len__函数
    　　该函数返回数据集中样本的总数。
    """
    def __len__(self):

        return self.train_hrhs_all.shape[0]


# 测试HSIDataProcess能否正常工作。
if "__main__"==__name__:
    train_dataset = HSIDataProcess(
        "D:/all_datasets/Sample_data/CAVE_other/CAVE/Train/",
        create_F(),
        64,
        32,
        8,
        fspecial("gaussian", 8, 3),
        1
    )