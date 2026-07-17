"""
preprocessing_train_cave.py
    CAVE 数据集无监督训练、验证与实验记录脚本。
"""

import math                 as math
import os                   as os
import time                 as time
import shutil               as shutil
import argparse             as argparse
import yaml                 as yaml
import pandas               as pd
import seaborn              as sns
import torch.utils.data     as data
import matplotlib.pyplot    as plt
from tqdm                       import tqdm
from scipy.io                   import loadmat
from preprocessing_dataloader   import *
from preprocessing_utils        import BlurDownsample, DualLearningLoss
from models.AMSF                import *

# 允许重复加载 OpenMP 运行库，避免部分 Windows/Conda 环境下的库冲突报错。
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def calculate_channel_tv(tensor):
    """计算相邻光谱通道之间的平均 Total Variation，用于诊断光谱平滑程度。"""
    # tensor: (B, C, H, W)
    diff = torch.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
    tv = torch.mean(diff).item()
    return tv


def plot_channel_correlation(tensor, name="Correlation"):
    """绘制单个样本的通道相关性热力图。"""
    # 仅取 batch 中第一个样本，并转到 CPU 方便绘图。
    feat = tensor[0].detach().cpu()
    C, H, W = feat.shape
    # 将每个通道展平为一个向量，形状为 (C, H*W)。
    feat_flat = feat.view(C, -1)

    # 计算通道之间的相关系数矩阵，形状为 (C, C)。
    corr_matrix = torch.corrcoef(feat_flat).numpy()

    # 保存热力图。
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"{name} Channel Correlation")
    plt.xlabel("Channel Index")
    plt.ylabel("Channel Index")
    plt.tight_layout()
    plt.savefig(f"{name}_corr.png", dpi=300)
    plt.close()
    print(f"[*] 已生成并保存热力图: {name}_corr.png")


def calculate_shfe(tensor):
    """计算空间高频能量，使用 Sobel 梯度衡量图像细节强度。"""
    B, C, H, W = tensor.shape
    # 构造 Sobel 卷积核，并放到输入张量所在设备。
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(tensor.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(tensor.device)

    shfe_total = 0.0
    for c in range(C):
        feat_c = tensor[:, c:c + 1, :, :]
        # 分别计算 X 和 Y 方向梯度。
        grad_x = F.conv2d(feat_c, sobel_x, padding=1)
        grad_y = F.conv2d(feat_c, sobel_y, padding=1)
        # 计算梯度幅值。
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        shfe_total += grad_mag.mean().item()

    return shfe_total / C


def calculate_macs(tensor):
    """计算平均通道相似度，用于诊断不同通道特征是否过度相似。"""
    # 将每个通道展平为一个向量，形状为 (C, H*W)。
    feat = tensor[0].detach().view(tensor.shape[1], -1)
    # 对每个通道向量做 L2 归一化。
    feat_norm = F.normalize(feat, p=2, dim=1)
    # 计算通道两两余弦相似度。
    cos_sim_matrix = torch.mm(feat_norm, feat_norm.t())

    # 去掉对角线上的自相似项，只统计不同通道之间的相似度。
    C = cos_sim_matrix.shape[0]
    mask = torch.ones((C, C), dtype=torch.bool).fill_diagonal_(False)
    mean_abs_cos_sim = torch.abs(cos_sim_matrix[mask]).mean().item()

    return mean_abs_cos_sim


def make_zero_ddl_items(reference):
    """Return zero-valued DDL logs when the baseline disables DDL entirely."""
    zero = reference.new_tensor(0.0)

    return {
        "ddl": zero,
        "cycle_spatial": zero,
        "cycle_spectral": zero,
        "cycle": zero,
        "cycle_fused": zero,
        "cycle_spatial_log": zero,
        "cycle_spectral_log": zero,
        "jac": zero,
        "jac_spatial": zero,
        "jac_spectral": zero,
        "lambda_cycle_spatial": zero,
        "lambda_cycle_spectral": zero,
        "lambda_cycle_fused": zero,
        "lambda_jac": zero,
        "lambda_jac_spatial": zero,
        "lambda_jac_spectral": zero,
        "cycle_spatial_eff": zero,
        "cycle_spectral_eff": zero,
        "cycle_fused_eff": zero,
        "cycle_eff": zero,
        "jac_spatial_eff": zero,
        "jac_spectral_eff": zero,
        "jac_eff": zero,
        "cue_sam_mean": zero,
        "cue_grad_mean": zero,
        "gate_mean": zero,
        "gate_min": zero,
        "gate_max": zero,
        "reliability_weight_mean": zero,
        "reliability_weight_min": zero,
        "reliability_weight_max": zero,
        "spatial_reliability_weight_mean": zero,
        "spatial_reliability_weight_min": zero,
        "spatial_reliability_weight_max": zero,
        "spectral_reliability_weight_mean": zero,
        "spectral_reliability_weight_min": zero,
        "spectral_reliability_weight_max": zero,
        "obs_lr_error_mean": zero,
        "obs_ms_error_mean": zero,
        "cross_spatial_ms_error_mean": zero,
        "cross_spectral_lr_error_mean": zero,
        "ob_rely_pixel_lr_weight": reference.new_tensor(0.5),
        "ob_rely_structure_lr_weight": reference.new_tensor(0.5),
        "ob_rely_pixel_structure_weight": reference.new_tensor(0.5),
        "output_refine_scale_3x3_mean": zero,
        "output_refine_scale_5x5_mean": zero,
        "output_refine_scale_7x7_mean": zero,
        "output_refine_residual_abs_mean": zero,
        "ob_rely_scale_3x3_mean": zero,
        "ob_rely_scale_5x5_mean": zero,
        "ob_rely_scale_7x7_mean": zero,
        "ob_rely_band_high_mean": zero,
        "ob_rely_band_mid_mean": zero,
        "ob_rely_band_low_mean": zero,
        "ob_rely_refine_delta_abs_mean": zero,
        "ob_rely_refine_delta_cal_abs_mean": zero,
        "pullback_refine_spatial_abs_mean": zero,
        "pullback_refine_spectral_abs_mean": zero,
        "target_corr_delta_spatial_mean": zero,
        "target_corr_delta_spectral_mean": zero,
        "target_corr_shift_mean": zero,
        "target_corr_shift_max": zero,
        "target_corr_eta": zero,
        "bp_spatial_delta_mean": zero,
        "bp_spectral_delta_mean": zero,
    }


def get_cycle_schedule_scale(epoch, warmup_epoch, decay_start_epoch, end_epoch, final_scale):
    """Return the epoch-dependent scale for DDL cycle losses."""
    warmup_epoch = int(warmup_epoch)
    decay_start_epoch = int(decay_start_epoch)
    end_epoch = int(end_epoch)
    final_scale = float(final_scale)

    if warmup_epoch > 0 and epoch <= warmup_epoch:
        return float(epoch) / float(warmup_epoch)

    if decay_start_epoch < end_epoch and epoch > decay_start_epoch:
        progress = float(epoch - decay_start_epoch) / float(end_epoch - decay_start_epoch)
        if progress < 0.0:
            progress = 0.0
        elif progress > 1.0:
            progress = 1.0
        return 1.0 + (final_scale - 1.0) * progress

    return 1.0


def apply_cycle_schedule(dual_loss, epoch, end_epoch, base_cycle_weights, schedule_cfg):
    """Update DDL cycle weights in-place while keeping Jacobian weights unchanged."""
    if dual_loss is None or not schedule_cfg["enabled"]:
        return 1.0

    scale = get_cycle_schedule_scale(
        epoch,
        schedule_cfg["warmup_epoch"],
        schedule_cfg["decay_start_epoch"],
        end_epoch,
        schedule_cfg["final_scale"],
    )

    with torch.no_grad():
        dual_loss.lambda_cycle_fused.fill_(base_cycle_weights["fused"] * scale)
        dual_loss.lambda_cycle_spatial.fill_(base_cycle_weights["spatial"] * scale)
        dual_loss.lambda_cycle_spectral.fill_(base_cycle_weights["spectral"] * scale)

    return scale



if "__main__"==__name__:
    parser = argparse.ArgumentParser(description="Train AMSF-Net CAVE experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to the experiment config YAML.")
    args = parser.parse_args()
    config_file = args.config

    # ===========================================================================================
    # 读取 CAVE 数据集配置，并以配置中的 seed 作为最终随机种子。
    set_seed(42)
    with open(config_file, 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["CAVE"]
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    # Previous single cycle weight:
    # lambda_cycle = float(cfg["train"]["lambda_cycle"])
    use_ddl = bool(cfg["train"].get("use_ddl", True))
    use_base_degradation_consistency = bool(cfg["train"].get("use_base_degradation_consistency", True))
    pullback_mode = cfg["train"].get("pullback_mode", "legacy")
    lambda_cycle_spatial = float(cfg["train"].get("lambda_cycle_spatial", 0.0))
    lambda_cycle_spectral = float(cfg["train"].get("lambda_cycle_spectral", 0.0))
    lambda_cycle_fused = float(cfg["train"].get("lambda_cycle_fused", 0.0))
    lambda_jac_spatial = float(cfg["train"].get("lambda_jac_spatial", cfg["train"].get("lambda_jac", 0.0)))
    lambda_jac_spectral = float(cfg["train"].get("lambda_jac_spectral", 0.0))
    jac_interval = int(cfg["train"].get("jac_interval", 1))
    cycle_reliability_enable = bool(cfg["train"].get("cycle_reliability_enable", False))
    cycle_reliability_mode = cfg["train"].get("cycle_reliability_mode", "target_average")
    cycle_reliability_tau = float(cfg["train"].get("cycle_reliability_tau", 2.0))
    cycle_reliability_min = float(cfg["train"].get("cycle_reliability_min", 0.2))
    cycle_reliability_normalize = bool(cfg["train"].get("cycle_reliability_normalize", True))
    cycle_reliability_apply_to_branches = bool(cfg["train"].get("cycle_reliability_apply_to_branches", False))
    cycle_reliability_learnable_mix = bool(cfg["train"].get("cycle_reliability_learnable_mix", False))
    output_refinement_enable = bool(cfg["train"].get("output_refinement_enable", False))
    observation_target_correction_enable = bool(cfg["train"].get("observation_target_correction_enable", False))
    observation_target_correction_eta = float(cfg["train"].get("observation_target_correction_eta", 0.05))
    observation_target_correction_clamp = bool(cfg["train"].get("observation_target_correction_clamp", True))
    spectral_pullback_learnable = bool(cfg["train"].get("spectral_pullback_learnable", True))
    beta_spatial_bp = float(cfg["train"].get("beta_spatial_bp", 0.0))
    beta_spectral_bp = float(cfg["train"].get("beta_spectral_bp", 0.0))
    observation_backprojection_detach = bool(cfg["train"].get("observation_backprojection_detach", False))
    cycle_schedule_cfg = {
        "enabled": bool(cfg["train"].get("cycle_schedule_enable", False)),
        "warmup_epoch": int(cfg["train"].get("cycle_schedule_warmup_epoch", 100)),
        "decay_start_epoch": int(cfg["train"].get("cycle_schedule_decay_start_epoch", 800)),
        "final_scale": float(cfg["train"].get("cycle_schedule_final_scale", 0.3)),
    }
    base_cycle_weights = {
        "fused": lambda_cycle_fused,
        "spatial": lambda_cycle_spatial,
        "spectral": lambda_cycle_spectral,
    }
    experiment_name = cfg["train"].get("experiment_name")
    legacy_cycle_only_log = bool(cfg["train"].get("legacy_cycle_only_log", False))
    if experiment_name:
        cycle_experiment_name = experiment_name
    elif use_ddl and pullback_mode == "dual_pullback_agf_v1":
        cycle_experiment_name = (
            f"dual_pullback_agf_v1_f{lambda_cycle_fused:g}_js{lambda_jac_spatial:g}"
            f"_jp{lambda_jac_spectral:g}_ji{jac_interval:g}"
        ).replace(".", "p")
    elif use_ddl:
        cycle_experiment_name = (
            f"cycle_split_{pullback_mode}_s{lambda_cycle_spatial:g}_p{lambda_cycle_spectral:g}_j{lambda_jac_spatial:g}"
            f"_ji{jac_interval:g}"
        ).replace(".", "p")
    elif use_base_degradation_consistency:
        cycle_experiment_name = "base_seed42"
    else:
        cycle_experiment_name = "i1_edge_only_seed42"

    # 3. 路径设置。服务器默认目录见“目录调整.md”，也可用环境变量覆盖。
    dataset_root = cfg["train"].get("dataset_root", "/root/autodl-tmp/datasets/lrtn/CAVE")
    run_root = os.environ.get("AMSF_RUN_ROOT", cfg["train"].get("run_root", "/root/autodl-tmp/runs/AMSF-Net"))
    run_path = os.path.join(run_root, cycle_experiment_name)
    save_train_path = os.path.join(run_path, "checkpoints", "train")
    save_test_path = os.path.join(run_path, "checkpoints", "test")
    log_path = os.path.join(run_path, "logs")
    record_path = os.path.join(run_path, "records")
    config_path = os.path.join(run_path, "configs")
    prediction_path = os.path.join(run_path, "predictions")
    metrics_path = os.path.join(run_path, "metrics")
    train_path = os.environ.get("AMSF_TRAIN_PATH", os.path.join(dataset_root, "Train"))
    test_path = os.environ.get("AMSF_TEST_PATH", os.path.join(dataset_root, "Test"))
    # 4. 读取验证集文件列表，并打印当前实验路径。
    test_filename_list = get_filename_list(test_path)
    print(f"训练数据路径: {train_path}")
    print(f"测试数据路径: {test_path}")
    print(f"实验输出路径: {run_path}")
    # 5. 初始化退化矩阵和训练参数。
    # 如果需要直接使用 PyTorch 的均方误差损失，可启用下面这一行。
    # loss_func = nn.MSELoss(reduction="mean").cuda()
    # 构造 CAVE 的光谱响应矩阵和 LRTN 使用的 Gaussian PSF。
    R = create_F()
    PSF = fspecial("gaussian", 8, 3)
    downsample_factor = cfg["train"]["downsample_factor"]   # 空间下采样倍率，CAVE 默认为 8。
    training_size =     cfg["train"]["training_size"]       # 训练 patch 尺寸。
    train_stride =      cfg["train"]["train_stride"]        # 训练滑窗步长。
    lr =                cfg["train"]["lr"]                  # 初始学习率。
    end_epoch =         cfg["train"]["end_epoch"]           # 总训练轮数。
    weight_decay =      cfg["train"]["weight_decay"]        # Adam 权重衰减。
    batch_size =        cfg["train"]["batch_size"]          # 训练 batch size。
    num =               cfg["train"]["num"]                 # 参与训练的数据数量。
    epoch_gap =         cfg["train"]["epoch_gap"]           # 训练 checkpoint 保存间隔。
    val_interval =      cfg["train"]["val_interval"]        # 验证间隔。
    test_epoch =        cfg["test"]["test_epoch"]           # 开始验证的最小 epoch。
    test_stride =       cfg["test"]["test_stride"]          # 验证滑窗步长。
    psnr_optimal =      cfg["test"]["psnr_optimal"]         # PSNR 最优模型保存阈值。
    rmse_optimal =      cfg["test"]["rmse_optimal"]         # RMSE 最优模型保存阈值。
    # ===========================================================================================
    # 1. 创建输出目录。
    mkdir(record_path)
    mkdir(save_train_path)
    mkdir(save_test_path)
    mkdir(log_path)
    mkdir(config_path)
    mkdir(prediction_path)
    mkdir(metrics_path)

    config_snapshot = os.path.join(config_path, f"{cycle_experiment_name}_config.yaml")
    if not os.path.exists(config_snapshot):
        shutil.copy2(config_file, config_snapshot)

    excel_name = f"{cycle_experiment_name}_cave_record.csv"
    # excel_name = "harvard_record.csv"
    excel_path = os.path.join(record_path, excel_name)

    if not os.path.exists(excel_path):
        if legacy_cycle_only_log:
            df = pd.DataFrame(columns=["epoch", "lr", "train_loss", "val_loss",
                                       "RMSE", "PSNR", "SAM", "SSIM", "ERGAS",
                                       "lambda_cycle", "lambda_jac", "loss_cycle",
                                       "jac_loss", "cycle_eff", "jac_eff"])
        else:
            df = pd.DataFrame(columns=["epoch", "lr", "train_loss", "val_loss",
                                       "RMSE", "PSNR", "SAM", "SSIM", "ERGAS",
                                       "cycle_schedule_scale",
                                       "lambda_cycle_fused", "lambda_cycle_spatial_aux",
                                       "lambda_cycle_spectral_aux", "lambda_jac_spatial", "lambda_jac_spectral",
                                       "loss_cycle_fused", "loss_cycle_spatial_log", "loss_cycle_spectral_log",
                                       "jac_spatial", "jac_spectral", "jac_loss",
                                       "cycle_fused_eff", "cycle_spatial_aux_eff", "cycle_spectral_aux_eff",
                                       "jac_spatial_eff", "jac_spectral_eff", "jac_eff",
                                       "jac_active_count", "jac_active_ratio", "mean_jac_loss", "mean_jac_eff",
                                       "cue_sam_mean", "cue_grad_mean", "gate_mean", "gate_min", "gate_max",
                                       "reliability_weight_mean", "reliability_weight_min", "reliability_weight_max",
                                       "spatial_reliability_weight_mean", "spatial_reliability_weight_min",
                                       "spatial_reliability_weight_max", "spectral_reliability_weight_mean",
                                       "spectral_reliability_weight_min", "spectral_reliability_weight_max",
                                       "obs_lr_error_mean", "obs_ms_error_mean",
                                       "cross_spatial_ms_error_mean", "cross_spectral_lr_error_mean",
                                       "ob_rely_pixel_lr_weight", "ob_rely_structure_lr_weight",
                                       "ob_rely_pixel_structure_weight",
                                       "output_refine_scale_3x3_mean", "output_refine_scale_5x5_mean",
                                       "output_refine_scale_7x7_mean", "output_refine_residual_abs_mean",
                                       "ob_rely_scale_3x3_mean", "ob_rely_scale_5x5_mean",
                                       "ob_rely_scale_7x7_mean", "ob_rely_band_high_mean",
                                       "ob_rely_band_mid_mean", "ob_rely_band_low_mean",
                                       "ob_rely_refine_delta_abs_mean", "ob_rely_refine_delta_cal_abs_mean",
                                       "pullback_refine_spatial_abs_mean", "pullback_refine_spectral_abs_mean",
                                       "target_corr_delta_spatial_mean", "target_corr_delta_spectral_mean",
                                       "target_corr_shift_mean", "target_corr_shift_max", "target_corr_eta",
                                       "bp_spatial_delta_mean", "bp_spectral_delta_mean"])
        df.to_csv(excel_path, index=False)
    # ===========================================================================================
    # 1. 构造训练数据集和 DataLoader。
    train_dataset = HSIDataProcess(train_path, R, training_size, train_stride, downsample_factor, PSF, num)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    max_iteration = math.ceil(len(train_dataset)/batch_size) * end_epoch

    print("总迭代次数为{}。".format(max_iteration))
    # ===========================================================================================
    # 1. 初始化 AMSF 主模型。
    model = AMSF(n_select_bands=3, n_bands=31).cuda()
    # 2. 初始化卷积层、线性层和 LayerNorm 参数。
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    # 3. 初始化 DDL 损失模块，并将其可学习参数加入优化器。
    dual_loss = None
    optimizer_parameters = list(model.parameters())
    if use_ddl:
        dual_loss = DualLearningLoss(R, downsample_factor=downsample_factor,
                                     psf=PSF,
                                     lambda_cycle_spatial=lambda_cycle_spatial,
                                     lambda_cycle_spectral=lambda_cycle_spectral,
                                     lambda_jac_spatial=lambda_jac_spatial,
                                     lambda_cycle_fused=lambda_cycle_fused,
                                     lambda_jac_spectral=lambda_jac_spectral,
                                     cycle_reliability_enable=cycle_reliability_enable,
                                     cycle_reliability_mode=cycle_reliability_mode,
                                     cycle_reliability_tau=cycle_reliability_tau,
                                     cycle_reliability_min=cycle_reliability_min,
                                     cycle_reliability_normalize=cycle_reliability_normalize,
                                     cycle_reliability_apply_to_branches=cycle_reliability_apply_to_branches,
                                     cycle_reliability_learnable_mix=cycle_reliability_learnable_mix,
                                     output_refinement_enable=output_refinement_enable,
                                     observation_target_correction_enable=observation_target_correction_enable,
                                     observation_target_correction_eta=observation_target_correction_eta,
                                     observation_target_correction_clamp=observation_target_correction_clamp,
                                     spectral_pullback_learnable=spectral_pullback_learnable,
                                     beta_spatial_bp=beta_spatial_bp,
                                     beta_spectral_bp=beta_spectral_bp,
                                     observation_backprojection_detach=observation_backprojection_detach,
                                     pullback_mode=pullback_mode).cuda()
        optimizer_parameters += list(dual_loss.parameters())
    optimizer = torch.optim.Adam(optimizer_parameters,
                                 lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
    # ===========================================================================================
    # 1. 如果存在历史 checkpoint，则从最近一轮恢复训练。
    start_epoch = findLastCheckpoint(save_train_path)
    if start_epoch>=1:
        print(f"从第{start_epoch}轮开始恢复训练。")

        checkpoint = torch.load(os.path.join(save_train_path, f"model_{start_epoch:04d}.pth"), weights_only=True)

        model.load_state_dict(checkpoint["net_parameter"], strict=False)
        reset_optimizer_state = False
        if dual_loss is not None and "dual_parameter" in checkpoint:
            dual_state = checkpoint["dual_parameter"]
            dual_loss.load_state_dict(dual_state, strict=False)
            if "raw_lambda_cycle" in dual_state or "raw_lambda_jac" in dual_state:
                reset_optimizer_state = True
                print("DDL参数集合发生变化，将重新初始化优化器和调度器状态。")
        if not reset_optimizer_state:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_parameter"])
                scheduler.load_state_dict(checkpoint["scheduler_parameter"])
            except ValueError as exc:
                print(f"优化器或调度器状态与当前DDL参数不匹配，将仅恢复模型参数。原因：{exc}")
        # checkpoint 中记录的是已完成的 epoch。
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
    # 2. 初始化训练阶段使用的空间退化、blur 退化和光谱退化算子。
    spatial_down = SpatialDownsample(PSF, downsample_factor).cuda()
    blur_down = BlurDownsample(scale_factor=downsample_factor, channels=31).cuda()
    spectral_down = SpectralDownsample(R).cuda()

    # 旧版特征诊断代码曾用于查看 SHFE、MACS、TV 和通道相关性。
    # 这些诊断逻辑不参与当前训练流程；如需重新启用，建议单独写成独立分析脚本。

    # ===========================================================================================
    # 开始训练主循环。
    for epoch in range(start_epoch+1, end_epoch+1, 1):
        model.train()
        if dual_loss is not None:
            dual_loss.train()
        cycle_schedule_scale = apply_cycle_schedule(
            dual_loss,
            epoch,
            end_epoch,
            base_cycle_weights,
            cycle_schedule_cfg,
        )

        loss_all = []
        epoch_loss = 0
        jac_active_count = 0
        jac_loss_epoch_sum = 0.0
        jac_eff_epoch_sum = 0.0
        batch_count = 0
        loop = tqdm(train_loader, total=len(train_loader))
        start_time = time.time()

        for batch_idx, (hr_hsi, hr_msi, lr_hsi) in enumerate(loop):
            lr_hsi = lr_hsi.cuda()
            hr_msi = hr_msi.cuda()
            # ===========================================================================================
            # 前向传播：AMSF 输出 HRHSI 预测结果以及边缘分支特征。
            output_hrhsi, spat_edge1, spat_edge2, spec_edge = model(lr_hsi, hr_msi)
            if dual_loss is not None:
                output_hrhsi = dual_loss.refine_hrhsi(output_hrhsi)

            # 只有基础退化一致性或 DDL 启用时，才需要计算空间退化结果。
            need_output_lrhsi = use_base_degradation_consistency or (dual_loss is not None)
            if need_output_lrhsi:
                output_lrhsi = spatial_down(output_hrhsi)
            else:
                output_lrhsi = None
            # output_lrhsi = blur_down(output_hrhsi)
            # 光谱退化结果仍用于空间边缘损失，因此这里始终保留。
            output_hrmsi = spectral_down(output_hrhsi)

            B, C, H, W = output_hrhsi.shape

            # L2 退化一致性损失：空间一致性 + 光谱一致性，可用于消融实验中关闭。
            if use_base_degradation_consistency:
                _, C_lr, H_lr, W_lr = lr_hsi.shape
                loss_spatial = torch.sum((output_lrhsi - lr_hsi) ** 2) / (2 * W_lr * H_lr * C_lr)
                _, C_ms, H_ms, W_ms = hr_msi.shape
                loss_spectral = torch.sum((output_hrmsi - hr_msi) ** 2) / (2 * W_ms * H_ms * C_ms)
                loss_L2 = loss_spatial + loss_spectral
            else:
                loss_spatial = output_hrhsi.new_tensor(0.0)
                loss_spectral = output_hrhsi.new_tensor(0.0)
                loss_L2 = output_hrhsi.new_tensor(0.0)

            # 空间边缘损失：比较预测 HRMSI 与观测 HRMSI 的水平/垂直空间梯度。
            _, C_ms, H_ms, W_ms = hr_msi.shape
            pred_spat_e1 = output_hrmsi[:, :, 0:H_ms - 1, :] - output_hrmsi[:, :, 1:H_ms, :]
            pred_spat_e2 = output_hrmsi[:, :, :, 0:W_ms - 1] - output_hrmsi[:, :, :, 1:W_ms]
            gt_spat_e1 = hr_msi[:, :, 0:H_ms - 1, :] - hr_msi[:, :, 1:H_ms, :]
            gt_spat_e2 = hr_msi[:, :, :, 0:W_ms - 1] - hr_msi[:, :, :, 1:W_ms]
            # 对应公式(19)，分母为 2*W*C*(H-1)。
            loss_spat1 = torch.sum((pred_spat_e1 - gt_spat_e1) ** 2) / (2 * W_ms * C_ms * (H_ms - 1))
            # 对应公式(20)，分母为 2*H*C*(W-1)。
            loss_spat2 = torch.sum((pred_spat_e2 - gt_spat_e2) ** 2) / (2 * H_ms * C_ms * (W_ms - 1))
            # 对应公式(21)，两个方向等权相加。
            loss_spat = 0.5 * loss_spat1 + 0.5 * loss_spat2

            # 光谱边缘损失：将 LRHSI 上采样后构造代理光谱边缘，与模型光谱边缘分支对齐。
            lr_hsi_up = F.interpolate(lr_hsi, scale_factor=downsample_factor, mode="bilinear")
            proxy_spec_edge = lr_hsi_up[:, 0:lr_hsi_up.size(1) - 1, :, :] - lr_hsi_up[:, 1:lr_hsi_up.size(1), :, :]
            # 对应公式(22)，分母为 2*H*W*(C-1)。
            loss_spec = torch.sum((spec_edge - proxy_spec_edge) ** 2) / (2 * H * W * (C - 1))

            loss_L1 = loss_spec + loss_spat
            # 基础损失：L2退化一致性 + 0.2倍边缘损失。
            # loss_base = 0.2 * loss_L1 + loss_L2
            loss_edge = 0.2 * loss_L1
            # 原始基础损失写法：
            # loss_base = 0.2 * loss_L1 + loss_L2
            if use_base_degradation_consistency:
                loss_base = loss_edge + loss_L2
            else:
                loss_base = loss_edge
            if dual_loss is not None:
                compute_jac = jac_interval > 0 and (batch_idx % jac_interval == 0)
                loss_ddl, loss_ddl_items = dual_loss(
                    output_hrhsi, output_lrhsi, output_hrmsi, lr_hsi, hr_msi,
                    compute_jac=compute_jac,
                )
            else:
                compute_jac = False
                loss_ddl = output_hrhsi.new_tensor(0.0)
                loss_ddl_items = make_zero_ddl_items(output_hrhsi)
            batch_count += 1
            if compute_jac:
                jac_active_count += 1
            jac_loss_epoch_sum += loss_ddl_items["jac"].item()
            jac_eff_epoch_sum += loss_ddl_items["jac_eff"].item()
            # 历史尝试包括固定 lambda_ddl、全局可学习 base/DDL 权重和单个可学习 DDL 权重。
            # 当前版本统一在 DualLearningLoss 内部用固定权重组合 DDL 子项。
            loss = loss_base + loss_ddl
            # ===========================================================================================
            # 旧版消融中曾测试 L1/Charbonnier/动态边缘权重等替代损失。
            # 当前收敛版本不再启用这些分支，保留说明即可。
            # ===========================================================================================
            # 反向传播与参数更新。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_all.append(loss_value)
            epoch_loss += loss_value
            # 当前学习率用于进度条日志。
            lr_now = optimizer.param_groups[0]["lr"]

            loop.set_description(f"[第{epoch}轮/共{end_epoch}轮]")
            current_jac_active_ratio = jac_active_count / batch_count if batch_count > 0 else 0.0
            # 终端进度条只保留关键状态，完整诊断项仍写入 CSV。
            loop.set_postfix({"loss":   f"{loss_value:.8f}",
                              "base":   f"{loss_base.item():.8f}",
                              "ddl":    f"{loss_ddl_items['ddl'].item():.8f}",
                              "cyc":    f"{loss_ddl_items['cycle_eff'].item():.8f}",
                              "jac":    f"{loss_ddl_items['jac_eff'].item():.8f}",
                              "tshift": f"{loss_ddl_items['target_corr_shift_mean'].item():.6f}",
                              "lr":     f"{lr_now:.8f}"})
        scheduler.step()

        elapsed_time = time.time() - start_time

        print(f"第{epoch}轮训练完毕，总共用时为{elapsed_time:.2f}秒。")
        # print(f"loss_spatial:  {loss_spatial.item():.6f}")
        # print(f"loss_spectral: {loss_spectral.item():.6f}")
        # print(f"loss_spat:     {loss_spat.item():.6f}")
        # print(f"loss_spec:     {loss_spec.item():.6f}")
        # print(f"loss_L1:       {loss_L1.item():.6f}")
        # print(f"loss_L2:       {loss_L2.item():.6f}")
        # print(f"loss_ss:       {loss_ss.item():.6f}")
        # print(f"loss:          {loss.item():.6f}")
        # 按 epoch_gap 间隔保存训练 checkpoint。
        if 0==epoch%epoch_gap:
            # 同时保存主模型、优化器、调度器和 DDL 参数，便于继续训练。
            checkpoint = {
                "net_parameter":        model.state_dict(),
                "optimizer_parameter":  optimizer.state_dict(),
                "scheduler_parameter":  scheduler.state_dict(),
                "epoch":                epoch,
                "loss":                 float(np.mean(loss_all)),
            }
            if dual_loss is not None:
                checkpoint["dual_parameter"] = dual_loss.state_dict()

            mkdir(save_train_path)
            torch.save(checkpoint, os.path.join(save_train_path, f"model_{epoch:04d}.pth"))
        # 按 val_interval 间隔执行验证；第 1 轮也会验证一次用于观察初始状态。
        if ((0==epoch%val_interval) and (epoch>=test_epoch)) or 1==epoch:
            model.eval()
            if dual_loss is not None:
                dual_loss.eval()

            val_loss_meter = AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE().cuda()
            PSNR = Loss_PSNR().cuda()
            SSIM = Loss_SSIM().cuda()
            ERGAS = Loss_ERGAS().cuda()
            sam_meter = AverageMeter()
            rmse_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()
            ergas_meter = AverageMeter()

            with torch.no_grad():
                for i, filename in enumerate(test_filename_list):
                    file_path = os.path.join(test_path, filename)
                    img = loadmat(file_path)
                    # CAVE 数据归一化到 [0, 1]。
                    img1 = img["hsi"] / img["hsi"].max()
                    # img1 = img["ref"] / img["ref"].max()
                    HRHSI_np = np.transpose(img1, (2, 0, 1)).astype(np.float32)
                    HRHSI = torch.Tensor(HRHSI_np)
                    # 构造 GT HRHSI，并使用 LRTN 的验证退化方式生成 LRHSI 和 HRMSI。
                    HRHSI_gt = torch.unsqueeze(HRHSI, 0).cuda()
                    # HSI_LR = spatial_down(HRHSI_gt)
                    # MSI_HR = spectral_down(HRHSI_gt)
                    HSI_LR_np = Gaussian_downsample(HRHSI_np, PSF, downsample_factor).astype(np.float32)
                    MSI_HR_np = np.tensordot(R, HRHSI_np, axes=([1], [0])).astype(np.float32)
                    HSI_LR = torch.Tensor(HSI_LR_np).unsqueeze(0).cuda()
                    MSI_HR = torch.Tensor(MSI_HR_np).unsqueeze(0).cuda()
                    prediction, val_loss_meter = reconstruction(model, R, HSI_LR, MSI_HR, HRHSI_gt,
                                                                downsample_factor, training_size, test_stride,
                                                                val_loss_meter,
                                                                output_refiner=(
                                                                    dual_loss.refine_hrhsi if dual_loss is not None else None
                                                                ))
                    # 转为 HWC 格式，用于 SAM 等 numpy 指标。
                    pred_np = prediction.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    gt_np = HRHSI.cpu().numpy().transpose(1, 2, 0)
                    prediction = prediction.unsqueeze(0)

                    sam_meter.update(SAM(gt_np, pred_np))
                    rmse_meter.update(RMSE(HRHSI_gt, prediction))
                    psnr_meter.update(PSNR(HRHSI_gt, prediction))
                    ssim_meter.update(SSIM(HRHSI_gt, prediction))
                    ergas_meter.update(ERGAS(HRHSI_gt, prediction))
            # 打印当前验证指标。
            print(f"RMSE:\t\t{rmse_meter.avg:.4f}\n"
                  f"PSNR:\t\t{psnr_meter.avg:.4f}\n"
                  f"SAM:\t\t{sam_meter.avg:.4f}\n"
                  f"SSIM:\t\t{ssim_meter.avg:.4f}\n"
                  f"ERGAS:\t\t{ergas_meter.avg:.4f}\n"
                  f"验证损失:\t\t{val_loss_meter.avg:.6f}")
            # 按 PSNR/RMSE 阈值保存候选最优模型。
            if 1==epoch:
                torch.save(model.state_dict(), os.path.join(save_test_path, f"{cycle_experiment_name}_1EPOCH_PSNR_best.pkl"))
            if abs(psnr_optimal-psnr_meter.avg)<0.15:
                torch.save(model.state_dict(), os.path.join(save_test_path, f"{cycle_experiment_name}_{epoch}EPOCH_PSNR_best.pkl"))
            if psnr_meter.avg>psnr_optimal:
                psnr_optimal = psnr_meter.avg
            if abs(rmse_optimal-rmse_meter.avg)<0.15:
                torch.save(model.state_dict(), os.path.join(save_test_path, f"{cycle_experiment_name}_{epoch}EPOCH_RMSE_best.pkl"))
            if rmse_meter.avg<rmse_optimal:
                rmse_optimal = rmse_meter.avg

            # 统计 Jacobian 的启用比例和 epoch 平均有效权重值，并写入 CSV。
            jac_stat_count = batch_count if batch_count > 0 else 1
            jac_active_ratio = jac_active_count / jac_stat_count
            mean_jac_loss = jac_loss_epoch_sum / jac_stat_count
            mean_jac_eff = jac_eff_epoch_sum / jac_stat_count
            if legacy_cycle_only_log:
                val_list = [epoch, optimizer.param_groups[0]["lr"], np.mean(loss_all), val_loss_meter.avg,
                            rmse_meter.avg, psnr_meter.avg, sam_meter.avg, ssim_meter.avg, ergas_meter.avg,
                            loss_ddl_items["lambda_cycle_spatial"].item(),
                            loss_ddl_items["lambda_jac"].item(),
                            loss_ddl_items["cycle"].item(), loss_ddl_items["jac"].item(),
                            loss_ddl_items["cycle_eff"].item(), loss_ddl_items["jac_eff"].item()]
            else:
                val_list = [epoch, optimizer.param_groups[0]["lr"], np.mean(loss_all), val_loss_meter.avg,
                            rmse_meter.avg, psnr_meter.avg, sam_meter.avg, ssim_meter.avg, ergas_meter.avg,
                            cycle_schedule_scale,
                            loss_ddl_items["lambda_cycle_fused"].item(),
                            loss_ddl_items["lambda_cycle_spatial"].item(),
                            loss_ddl_items["lambda_cycle_spectral"].item(),
                            loss_ddl_items["lambda_jac_spatial"].item(),
                            loss_ddl_items["lambda_jac_spectral"].item(),
                            loss_ddl_items["cycle_fused"].item(),
                            loss_ddl_items["cycle_spatial_log"].item(),
                            loss_ddl_items["cycle_spectral_log"].item(),
                            loss_ddl_items["jac_spatial"].item(),
                            loss_ddl_items["jac_spectral"].item(),
                            loss_ddl_items["jac"].item(),
                            loss_ddl_items["cycle_fused_eff"].item(),
                            loss_ddl_items["cycle_spatial_eff"].item(),
                            loss_ddl_items["cycle_spectral_eff"].item(),
                            loss_ddl_items["jac_spatial_eff"].item(),
                            loss_ddl_items["jac_spectral_eff"].item(),
                            loss_ddl_items["jac_eff"].item(),
                            jac_active_count, jac_active_ratio,
                            mean_jac_loss, mean_jac_eff,
                            loss_ddl_items["cue_sam_mean"].item(),
                            loss_ddl_items["cue_grad_mean"].item(),
                            loss_ddl_items["gate_mean"].item(),
                            loss_ddl_items["gate_min"].item(),
                            loss_ddl_items["gate_max"].item(),
                            loss_ddl_items["reliability_weight_mean"].item(),
                            loss_ddl_items["reliability_weight_min"].item(),
                            loss_ddl_items["reliability_weight_max"].item(),
                            loss_ddl_items["spatial_reliability_weight_mean"].item(),
                            loss_ddl_items["spatial_reliability_weight_min"].item(),
                            loss_ddl_items["spatial_reliability_weight_max"].item(),
                            loss_ddl_items["spectral_reliability_weight_mean"].item(),
                            loss_ddl_items["spectral_reliability_weight_min"].item(),
                            loss_ddl_items["spectral_reliability_weight_max"].item(),
                            loss_ddl_items["obs_lr_error_mean"].item(),
                            loss_ddl_items["obs_ms_error_mean"].item(),
                            loss_ddl_items["cross_spatial_ms_error_mean"].item(),
                            loss_ddl_items["cross_spectral_lr_error_mean"].item(),
                            loss_ddl_items["ob_rely_pixel_lr_weight"].item(),
                            loss_ddl_items["ob_rely_structure_lr_weight"].item(),
                            loss_ddl_items["ob_rely_pixel_structure_weight"].item(),
                            loss_ddl_items["output_refine_scale_3x3_mean"].item(),
                            loss_ddl_items["output_refine_scale_5x5_mean"].item(),
                            loss_ddl_items["output_refine_scale_7x7_mean"].item(),
                            loss_ddl_items["output_refine_residual_abs_mean"].item(),
                            loss_ddl_items["ob_rely_scale_3x3_mean"].item(),
                            loss_ddl_items["ob_rely_scale_5x5_mean"].item(),
                            loss_ddl_items["ob_rely_scale_7x7_mean"].item(),
                            loss_ddl_items["ob_rely_band_high_mean"].item(),
                            loss_ddl_items["ob_rely_band_mid_mean"].item(),
                            loss_ddl_items["ob_rely_band_low_mean"].item(),
                            loss_ddl_items["ob_rely_refine_delta_abs_mean"].item(),
                            loss_ddl_items["ob_rely_refine_delta_cal_abs_mean"].item(),
                            loss_ddl_items["pullback_refine_spatial_abs_mean"].item(),
                            loss_ddl_items["pullback_refine_spectral_abs_mean"].item(),
                            loss_ddl_items["target_corr_delta_spatial_mean"].item(),
                            loss_ddl_items["target_corr_delta_spectral_mean"].item(),
                            loss_ddl_items["target_corr_shift_mean"].item(),
                            loss_ddl_items["target_corr_shift_max"].item(),
                            loss_ddl_items["target_corr_eta"].item(),
                            loss_ddl_items["bp_spatial_delta_mean"].item(),
                            loss_ddl_items["bp_spectral_delta_mean"].item()]
            val_data = pd.DataFrame([val_list])

            val_data.to_csv(excel_path, mode='a', header=False, index=False)
            time.sleep(0.1)

