"""Chikusei H5 adapter for the unified LRTN observation protocol.

Only the HR-HSI dataset stored under ``GT`` is used. LR-HSI and HR-MSI
observations shipped in the source H5 files belong to a different protocol and
are deliberately regenerated with the configured PSF and SRF.
"""

from dataclasses import dataclass
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing_utils import Gaussian_downsample


H5_EXTENSIONS = (".h5", ".hdf5")


@dataclass(frozen=True)
class ChikuseiH5Sample:
    """A logical HSI scene stored inside a batched H5 dataset."""

    file_path: str
    index: int
    key: str
    layout: str
    height: int
    width: int

    @property
    def name(self):
        filename = os.path.basename(self.file_path)
        return f"{filename}:{self.key}[{self.index}]"


def _list_h5_files(path):
    if os.path.isfile(path):
        if not str(path).lower().endswith(H5_EXTENSIONS):
            raise ValueError(f"文件 {path} 不是 .h5 或 .hdf5 文件。")
        return [os.path.abspath(path)]
    files = sorted(
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if str(filename).lower().endswith(H5_EXTENSIONS)
    )
    if not files:
        raise RuntimeError(f"目录 {path} 中没有找到 .h5 或 .hdf5 文件。")
    return files


def discover_chikusei_h5_samples(path, key="GT", expected_bands=128):
    """Enumerate all logical scenes from one or more batched Chikusei H5 files."""
    expected_bands = int(expected_bands)
    samples = []

    for file_path in _list_h5_files(path):
        with h5py.File(file_path, "r") as h5_file:
            if key not in h5_file:
                raise KeyError(
                    f"{file_path} 中不存在数据键 {key!r}，可用键为 {sorted(h5_file.keys())}。"
                )
            dataset = h5_file[key]
            if dataset.ndim != 4:
                raise ValueError(
                    f"{file_path} 中 {key!r} 的形状为 {dataset.shape}，"
                    "期望 NCHW 或 NHWC 四维数组。"
                )

            if dataset.shape[1] == expected_bands:
                layout = "NCHW"
                height, width = int(dataset.shape[2]), int(dataset.shape[3])
            elif dataset.shape[-1] == expected_bands:
                layout = "NHWC"
                height, width = int(dataset.shape[1]), int(dataset.shape[2])
            else:
                raise ValueError(
                    f"{file_path} 中 {key!r} 的形状为 {dataset.shape}，"
                    f"无法找到 {expected_bands} 波段维。"
                )

            for index in range(int(dataset.shape[0])):
                samples.append(
                    ChikuseiH5Sample(
                        file_path=file_path,
                        index=index,
                        key=key,
                        layout=layout,
                        height=height,
                        width=width,
                    )
                )

    return samples


def _normalize_hsi(image, mode="none", value=None):
    image = image.astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)):
        raise ValueError("Chikusei HSI 包含 NaN 或 Inf。")

    image_max = float(np.max(image))
    if image_max <= 0.0:
        raise ValueError(f"Chikusei HSI 最大值为 {image_max}，无法使用。")

    mode = str(mode).lower()
    if mode == "scene_max":
        image = image / image_max
    elif mode == "fixed_max":
        if value is None or float(value) <= 0.0:
            raise ValueError("normalization='fixed_max' requires a positive value.")
        if image_max > 1.0 + 1e-6:
            image = image / float(value)
    elif mode not in {"none", "identity"}:
        raise ValueError(
            f"不支持归一化模式 {mode!r}，可选值为 none、scene_max 或 fixed_max。"
        )

    return image.astype(np.float32, copy=False)


def _read_sample_from_handle(h5_file, sample, normalization, normalization_value):
    dataset = h5_file[sample.key]
    image = np.empty(dataset.shape[1:], dtype=np.float32)
    dataset.read_direct(image, source_sel=np.s_[sample.index, ...])
    if sample.layout == "NCHW":
        image = np.transpose(image, (1, 2, 0))
    image = _normalize_hsi(image, normalization, normalization_value)
    return image


def load_chikusei_h5_sample(
        sample,
        normalization="none",
        normalization_value=None,
):
    """Load one logical Chikusei scene as an HWC float32 array."""
    with h5py.File(sample.file_path, "r") as h5_file:
        image = _read_sample_from_handle(
            h5_file,
            sample,
            normalization,
            normalization_value,
        )
    return image


class ChikuseiH5Dataset(Dataset):
    """Lazily read Chikusei GT patches and generate unified LRTN observations."""

    def __init__(
            self,
            path,
            R,
            training_size,
            stride,
            downsample_factor,
            PSF,
            num=None,
            mat_key="GT",
            normalization="none",
            normalization_value=None,
            cache_observations=True,
    ):
        self.R = np.asarray(R, dtype=np.float64)
        self.training_size = int(training_size)
        self.stride = int(stride)
        self.downsample_factor = int(downsample_factor)
        self.PSF = np.asarray(PSF)
        self.normalization = normalization
        self.normalization_value = normalization_value
        self.cache_observations = bool(cache_observations)
        self._h5_handles = {}
        self._observation_cache = {}

        if self.training_size % self.downsample_factor != 0:
            raise ValueError(
                "training_size 必须能被 downsample_factor 整除，"
                f"当前为 {self.training_size} 和 {self.downsample_factor}。"
            )

        source_samples = discover_chikusei_h5_samples(
            path,
            key=mat_key,
            expected_bands=self.R.shape[1],
        )
        if num is not None:
            num = int(num)
            if num <= 0:
                raise ValueError(f"num 必须为正数，当前为 {num}。")
            if num > len(source_samples):
                raise ValueError(
                    f"训练H5仅包含 {len(source_samples)} 个GT样本，配置要求 {num} 个。"
                )
            source_samples = source_samples[:num]

        self.patch_index = []
        for sample in source_samples:
            if sample.height < self.training_size or sample.width < self.training_size:
                raise ValueError(
                    f"{sample.name} 的尺寸为 {sample.height}x{sample.width}，"
                    f"小于训练patch {self.training_size}。"
                )
            row_starts = self._window_starts(sample.height)
            col_starts = self._window_starts(sample.width)
            for top in row_starts:
                for left in col_starts:
                    self.patch_index.append((sample, top, left))

    def _window_starts(self, length):
        starts = list(range(0, length - self.training_size + 1, self.stride))
        last = length - self.training_size
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    def _get_handle(self, file_path):
        handle = self._h5_handles.get(file_path)
        if handle is None:
            handle = h5py.File(file_path, "r")
            self._h5_handles[file_path] = handle
        return handle

    def _load_hrhsi(self, sample, top, left):
        handle = self._get_handle(sample.file_path)
        image = _read_sample_from_handle(
            handle,
            sample,
            self.normalization,
            self.normalization_value,
        )
        patch = image[
            top:top + self.training_size,
            left:left + self.training_size,
            :,
        ]
        return np.transpose(patch, (2, 0, 1)).astype(np.float32, copy=False)

    def __getitem__(self, index):
        sample, top, left = self.patch_index[index]
        hr_hsi = np.ascontiguousarray(self._load_hrhsi(sample, top, left))

        cached = self._observation_cache.get(index)
        if cached is None:
            hr_msi = np.tensordot(self.R, hr_hsi, axes=([1], [0])).astype(np.float32)
            lr_hsi = Gaussian_downsample(
                hr_hsi,
                self.PSF,
                self.downsample_factor,
            ).astype(np.float32)
            hr_msi = np.ascontiguousarray(hr_msi)
            lr_hsi = np.ascontiguousarray(lr_hsi)
            if self.cache_observations:
                self._observation_cache[index] = (hr_msi, lr_hsi)
        else:
            hr_msi, lr_hsi = cached

        return (
            torch.from_numpy(hr_hsi),
            torch.from_numpy(hr_msi),
            torch.from_numpy(lr_hsi),
        )

    def __len__(self):
        return len(self.patch_index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def close(self):
        for handle in self._h5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles = {}

    def __del__(self):
        self.close()
