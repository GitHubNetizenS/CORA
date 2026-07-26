#!/usr/bin/env python3
"""Average two AMSF checkpoints and evaluate the averaged model on CAVE."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.io import loadmat

from models.AMSF import AMSF
from preprocessing_utils import (
    AverageMeter,
    Gaussian_downsample,
    Loss_ERGAS,
    Loss_PSNR,
    Loss_RMSE,
    Loss_SAM,
    Loss_SSIM,
    create_F,
    fspecial,
    get_filename_list,
    reconstruction,
    set_seed,
)


DEFAULT_RUN_DIR = Path(
    "/root/autodl-tmp/runs/AMSF-Net/061_seed42_058_main_spectral_anchor"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="平均两个训练checkpoint的主模型参数，并使用LRTN验证流程计算指标。"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint-a", type=Path, default=None)
    parser.add_argument("--checkpoint-b", type=Path, default=None)
    parser.add_argument(
        "--weight-a",
        type=float,
        default=0.5,
        help="checkpoint A的平均权重；checkpoint B权重自动设为1-weight-a。",
    )
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--result-csv", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch设备，例如cuda、cuda:0或cpu；本项目通常应使用cuda。",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    run_dir = args.run_dir.resolve()
    experiment_name = run_dir.name

    if args.config is None:
        args.config = (
            run_dir / "configs" / f"{experiment_name}_config.yaml"
        )
    if args.checkpoint_a is None:
        args.checkpoint_a = (
            run_dir / "checkpoints" / "train" / "model_0950.pth"
        )
    if args.checkpoint_b is None:
        args.checkpoint_b = (
            run_dir / "checkpoints" / "train" / "model_1000.pth"
        )
    if args.output_checkpoint is None:
        args.output_checkpoint = (
            run_dir
            / "checkpoints"
            / "averaged"
            / "model_avg_0950_1000.pth"
        )
    if args.result_csv is None:
        args.result_csv = (
            run_dir / "metrics" / "checkpoint_average_0950_1000.csv"
        )

    args.run_dir = run_dir
    args.config = args.config.resolve()
    args.checkpoint_a = args.checkpoint_a.resolve()
    args.checkpoint_b = args.checkpoint_b.resolve()
    args.output_checkpoint = args.output_checkpoint.resolve()
    args.result_csv = args.result_csv.resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    for path_name in ("config", "checkpoint_a", "checkpoint_b"):
        path = getattr(args, path_name)
        if not path.is_file():
            raise FileNotFoundError(f"{path_name}不存在: {path}")

    if not 0.0 <= args.weight_a <= 1.0:
        raise ValueError("--weight-a必须位于[0, 1]。")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用CUDA GPU。")


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "net_parameter" not in checkpoint:
        raise KeyError(f"checkpoint缺少net_parameter字段: {path}")
    return checkpoint


def average_state_dicts(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    weight_a: float,
) -> dict[str, torch.Tensor]:
    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise KeyError(
            "两份checkpoint的模型参数键不一致。"
            f" 仅A存在: {only_a[:5]}; 仅B存在: {only_b[:5]}"
        )

    weight_b = 1.0 - weight_a
    averaged: dict[str, torch.Tensor] = {}
    for key in state_a:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"参数{key}形状不一致: {tensor_a.shape} vs {tensor_b.shape}"
            )
        if tensor_a.dtype != tensor_b.dtype:
            raise TypeError(
                f"参数{key}类型不一致: {tensor_a.dtype} vs {tensor_b.dtype}"
            )

        if tensor_a.is_floating_point() or tensor_a.is_complex():
            averaged[key] = tensor_a.mul(weight_a).add(
                tensor_b, alpha=weight_b
            )
        else:
            # 整数计数器等非浮点buffer不适合平均，采用较新checkpoint的值。
            averaged[key] = tensor_b.clone()
    return averaged


def save_average_checkpoint(
    args: argparse.Namespace,
    checkpoint_a: dict[str, Any],
    checkpoint_b: dict[str, Any],
    averaged_state: dict[str, torch.Tensor],
) -> None:
    output = {
        "net_parameter": averaged_state,
        "checkpoint_type": "parameter_average_for_evaluation_only",
        "source_checkpoints": [
            str(args.checkpoint_a),
            str(args.checkpoint_b),
        ],
        "source_epochs": [
            checkpoint_a.get("epoch"),
            checkpoint_b.get("epoch"),
        ],
        "average_weights": [args.weight_a, 1.0 - args.weight_a],
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_checkpoint)


def evaluate(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    train_cfg = cfg["train"]
    test_cfg = cfg["test"]
    if bool(train_cfg.get("output_refinement_enable", False)):
        raise ValueError(
            "该脚本面向实验061；检测到output_refinement_enable=true，"
            "无法仅凭主模型参数复现输出精炼。"
        )

    dataset_root = train_cfg.get(
        "dataset_root", "/root/autodl-tmp/datasets/lrtn/CAVE"
    )
    test_path = Path(
        os.environ.get("AMSF_TEST_PATH", os.path.join(dataset_root, "Test"))
    )
    if not test_path.is_dir():
        raise FileNotFoundError(f"测试数据目录不存在: {test_path}")

    test_filenames = get_filename_list(str(test_path))
    if not test_filenames:
        raise RuntimeError(f"测试目录中没有可用数据: {test_path}")

    response = create_F()
    psf = fspecial("gaussian", 8, 3)
    downsample_factor = int(train_cfg["downsample_factor"])
    training_size = int(train_cfg["training_size"])
    test_stride = int(test_cfg["test_stride"])

    val_loss_meter = AverageMeter()
    sam_metric = Loss_SAM()
    rmse_metric = Loss_RMSE().to(device)
    psnr_metric = Loss_PSNR().to(device)
    ssim_metric = Loss_SSIM().to(device)
    ergas_metric = Loss_ERGAS().to(device)
    sam_meter = AverageMeter()
    rmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    ergas_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for index, filename in enumerate(test_filenames, start=1):
            file_path = test_path / filename
            image = loadmat(file_path)
            if "hsi" not in image:
                raise KeyError(f"CAVE文件缺少hsi字段: {file_path}")

            normalized = image["hsi"] / image["hsi"].max()
            hrhsi_np = np.transpose(normalized, (2, 0, 1)).astype(
                np.float32
            )
            hrhsi_cpu = torch.from_numpy(hrhsi_np)
            hrhsi_gt = hrhsi_cpu.unsqueeze(0).to(device)

            lrhsi_np = Gaussian_downsample(
                hrhsi_np, psf, downsample_factor
            ).astype(np.float32)
            hrmsi_np = np.tensordot(
                response, hrhsi_np, axes=([1], [0])
            ).astype(np.float32)
            lrhsi = torch.from_numpy(lrhsi_np).unsqueeze(0).to(device)
            hrmsi = torch.from_numpy(hrmsi_np).unsqueeze(0).to(device)

            prediction, val_loss_meter = reconstruction(
                model,
                response,
                lrhsi,
                hrmsi,
                hrhsi_gt,
                downsample_factor,
                training_size,
                test_stride,
                val_loss_meter,
                output_refiner=None,
            )

            prediction_hwc = prediction.cpu().numpy().transpose(1, 2, 0)
            target_hwc = hrhsi_cpu.numpy().transpose(1, 2, 0)
            prediction_batch = prediction.unsqueeze(0)

            sam_meter.update(sam_metric(target_hwc, prediction_hwc))
            rmse_meter.update(rmse_metric(hrhsi_gt, prediction_batch))
            psnr_meter.update(psnr_metric(hrhsi_gt, prediction_batch))
            ssim_meter.update(ssim_metric(hrhsi_gt, prediction_batch))
            ergas_meter.update(ergas_metric(hrhsi_gt, prediction_batch))
            print(
                f"[{index:02d}/{len(test_filenames):02d}] "
                f"{filename}  PSNR={psnr_meter.avg:.4f} "
                f"SAM={sam_meter.avg:.4f}",
                flush=True,
            )

    return {
        "val_loss": float(val_loss_meter.avg),
        "RMSE": float(rmse_meter.avg),
        "PSNR": float(psnr_meter.avg),
        "SAM": float(sam_meter.avg),
        "SSIM": float(ssim_meter.avg),
        "ERGAS": float(ergas_meter.avg),
    }


def write_result(
    args: argparse.Namespace,
    checkpoint_a: dict[str, Any],
    checkpoint_b: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    row: dict[str, Any] = {
        "checkpoint_a": str(args.checkpoint_a),
        "epoch_a": checkpoint_a.get("epoch"),
        "weight_a": args.weight_a,
        "checkpoint_b": str(args.checkpoint_b),
        "epoch_b": checkpoint_b.get("epoch"),
        "weight_b": 1.0 - args.weight_a,
        "averaged_checkpoint": str(args.output_checkpoint),
        **metrics,
    }
    args.result_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.result_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = resolve_paths(parse_args())
    validate_args(args)

    print(f"实验目录: {args.run_dir}")
    print(f"配置文件: {args.config}")
    print(f"checkpoint A: {args.checkpoint_a}")
    print(f"checkpoint B: {args.checkpoint_b}")
    print(
        f"平均权重: A={args.weight_a:.4f}, "
        f"B={1.0 - args.weight_a:.4f}"
    )

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cave_cfg = config["CAVE"]
    set_seed(int(cave_cfg["train"].get("seed", 42)))

    checkpoint_a = load_checkpoint(args.checkpoint_a)
    checkpoint_b = load_checkpoint(args.checkpoint_b)
    averaged_state = average_state_dicts(
        checkpoint_a["net_parameter"],
        checkpoint_b["net_parameter"],
        args.weight_a,
    )
    save_average_checkpoint(
        args, checkpoint_a, checkpoint_b, averaged_state
    )

    device = torch.device(args.device)
    model = AMSF(n_select_bands=3, n_bands=31).to(device)
    model.load_state_dict(averaged_state, strict=True)
    metrics = evaluate(model, cave_cfg, device)
    write_result(args, checkpoint_a, checkpoint_b, metrics)

    print("\n平均模型验证完成")
    for name in ("RMSE", "PSNR", "SAM", "SSIM", "ERGAS", "val_loss"):
        print(f"{name}: {metrics[name]:.6f}")
    print(f"平均checkpoint: {args.output_checkpoint}")
    print(f"指标CSV: {args.result_csv}")


if __name__ == "__main__":
    main()
