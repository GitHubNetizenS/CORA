# AMSF-Net
An Asymptotic Multiscale Symmetric Fusion Network for Hyperspectral and Multispectral Image Fusion
**<p align="center">Abstract</p>**
Despite the high spectral resolution and abundant information of hyperspectral images (HSIs), their spatial resolution is relatively low due to limitations in sensor technology. Sensors often need to sacrifice some spatial resolution to ensure accurate light energy measurement when pursuing high spectral resolution. This tradeoff results in HSI’s inability to capture fine spatial details, thereby limiting its application in scenarios requiring high-precision spatial information. HSI and multispectral image (MSI) fusion is a commonly used technique for generating high-resolution HSI (HR-HSI). However, many deep learning-based HSI-MSI fusion algorithms ignore correlation and multiscale information between input images. To address this issue, we propose an asymptotic multiscale symmetric fusion network (AMSF-Net) for hyperspectral and MSI fusion. AMSF-Net consists of two parts: the multilevel feature fusion (MFF) module and the progressive cross-scale spatial perception (PCP) module. The MFF module uses multistream feature extraction branches to perform information interaction between HSI and MSI at the same scale layer by layer, compensating for the spatial details lacking in HSI and the spectral details absent in MSI. The PCP module combines the input and output features of MFF, utilizes multiscale bidirectional strip convolution and deep convolution to further refine edge features, and reconstructs HR-HSI by learning the features of different expansion roll branches by connecting across scales. Comparative experiments with several state-of-the-art HSI-MSI fusion algorithms on four publicly available datasets, CAVE, Chikusei, Houston, and WorldView 3, are conducted to validate the effectiveness and superiority of AMSF-Net. On the Chikusei dataset, improvements were 9.1%, 12.5%, and 5.1%, respectively, on the indicators root mean-square error (RMSE), error of relative global accuracy in synthesis (ERGAS), and spectral angle mapper (SAM), compared to the suboptimal method.

**Cite**: Liu Shuaiqi, Shao Tingting*, Liu Siyuan*, Li Bing and Zhang Yu-Dong. An asymptotic multiscale symmetric fusion network for hyperspectral and multispectral image fusion. IEEE Transactions on Geoscience and Remote Sensing, 2025, 63, pp. 1-16, Art no. 5503016. https://ieeexplore.ieee.org/document/10824890



# 数据集来源：
● CAVE数据集：https://gitcode.com/Universal-Tool/cgdaf

● Chikusei数据集：https://hf-mirror.com/datasets/danaroth/chikusei

● Houston数据集：https://drive.uc.cn/s/3fe4f55a213f4?public=1

● WorldView-3数据集：https://github.com/liangjiandeng/PanCollection
