import os
import time
import csv
import argparse
import hashlib
import json
import random
import resource
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import Subset
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from tqdm import tqdm

from utils.misc_utils import *
from utils.dataloader_CMRx4DFlow import (
    CMRx4DFlowDataSet,
    DEFAULT_OPTS as DATA_DEFAULT_OPTS,
)
from utils.utils_datasl import save_coo_npz
from utils.utils_metrics import nRMSE, SSIM, RelErr, AngErr, VelocityVectorRMSE
from utils.utils_flow import complex2magflow
from utils.utils_bgc import execute_MSAC
from utils.flowvn_preprocess import finalize_flowvn_deferred_adjoint
from utils.flowvn_distillation import (
    DistillationLossHelper,
    FlowVNTeacher,
    initialize_student_from_teacher_checkpoint,
)
from utils.flowvn_checkpoint import remap_state_dict_for_expected_keys
from utils.flowvn_run_naming import next_numeric_run_id
from utils.flowvn_validation import (
    StreamingValidationAccumulator,
    build_qualitative_source,
    select_validation_filename_indices,
)
from networks.flowvn_lowmem import FlowVNLowMem
from networks.flowvn import FlowVN

try:
    import wandb
except Exception:
    wandb = None

try:
    import yaml
except Exception:
    yaml = None


def _set_torchdynamo_lru_cache(enabled: bool):
    eval_frame = getattr(
        getattr(getattr(torch, "_C", None), "_dynamo", None),
        "eval_frame",
        None,
    )
    setter = getattr(eval_frame, "_set_lru_cache", None)
    if callable(setter):
        setter(enabled)
        return "private_lru"

    # PyTorch 2.5 predates the private LRU-cache switch in newer releases.
    # For enabled=False, expand the public recompilation limits to cover the
    # 43 variable-shape test cases instead of failing at startup or falling
    # back to eager execution after the default eight shapes.
    if enabled:
        raise RuntimeError(
            "This PyTorch build cannot explicitly enable TorchDynamo's LRU cache"
        )
    config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if config is None:
        raise RuntimeError("This PyTorch build exposes no TorchDynamo cache controls")
    if hasattr(config, "cache_size_limit"):
        config.cache_size_limit = max(int(config.cache_size_limit), 64)
    if hasattr(config, "accumulated_cache_size_limit"):
        config.accumulated_cache_size_limit = max(
            int(config.accumulated_cache_size_limit), 1024
        )
    return "expanded_limits"


def _apply_torch_runtime_options(args: dict):
    if bool(args.get("torchdynamo_disable_lru_cache", False)):
        cache_mode = _set_torchdynamo_lru_cache(False)
        if cache_mode == "private_lru":
            print(
                "[INFO] TorchDynamo LRU cache disabled for variable-shape "
                "recompilation"
            )
        else:
            print(
                "[INFO] TorchDynamo private LRU control unavailable; expanded "
                "variable-shape cache limits"
            )

    torch.backends.cudnn.benchmark = bool(args.get("cudnn_benchmark", True))
    deterministic = bool(args.get("deterministic_algorithms", False))
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)

    allow_tf32 = bool(args.get("allow_tf32", False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _apply_seed_options(args: dict):
    seed = args.get("seed", None)
    if seed is None:
        return
    seed = int(seed)
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lightning_precision(args: dict):
    mode = str(args.get("precision_mode", "fp32"))
    if mode == "amp_fp16":
        return "16-mixed"
    if mode == "amp_bf16":
        return "bf16-mixed"
    return "32-true"


def _state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _dataloader_kwargs(args: dict, batch_size: int, num_workers: int, shuffle: bool):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "shuffle": shuffle,
    }
    if num_workers > 0:
        if bool(args.get("dataloader_persistent_workers", False)):
            kwargs["persistent_workers"] = True
        prefetch_factor = args.get("dataloader_prefetch_factor", None)
        if prefetch_factor is not None:
            prefetch_factor = int(prefetch_factor)
            if prefetch_factor > 0:
                kwargs["prefetch_factor"] = prefetch_factor
    if args.get("seed", None) is not None:
        seed = int(args["seed"]) + (0 if shuffle else 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
        kwargs["worker_init_fn"] = _seed_worker
    return kwargs


def _select_validation_subset(dataset, args: dict):
    requested = args.get("val_sample_indices", None)
    case_id = args.get("val_case_id", None)
    usrate_filter = args.get("val_usrate_filter", None)
    semantic_filter = case_id not in (None, "", "None") or usrate_filter is not None
    if requested not in (None, [], ()) and semantic_filter:
        raise ValueError(
            "--val_sample_indices cannot be combined with --val_case_id or "
            "--val_usrate_filter"
        )
    if requested not in (None, [], ()):
        indices = [int(index) for index in requested]
        if len(indices) != len(set(indices)):
            raise ValueError("--val_sample_indices must not contain duplicates")
        invalid = [index for index in indices if index < 0 or index >= len(dataset)]
        if invalid:
            raise IndexError(
                f"Validation indices out of range for dataset of size {len(dataset)}: "
                f"{invalid}"
            )
        print(f"[INFO] val_sample_indices enabled: {indices}")
        return Subset(dataset, indices)

    if semantic_filter:
        filenames = getattr(dataset, "filename", None)
        if filenames is None:
            raise ValueError("Validation dataset does not expose filename metadata")
        indices = select_validation_filename_indices(
            filenames,
            case_id=case_id,
            usrate=usrate_filter,
        )
        print(
            "[INFO] semantic validation filter enabled: "
            f"case_id={case_id} usrate={usrate_filter} indices={indices}"
        )
        return Subset(dataset, indices)

    sample_limit = int(args.get("val_sample_limit", 0) or 0)
    if sample_limit > 0 and sample_limit < len(dataset):
        print(f"[INFO] val_sample_limit enabled: using {sample_limit} samples")
        return Subset(dataset, list(range(sample_limit)))
    return dataset


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _scalar_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return _scalar_bool(value[0])
    return bool(value)


def _finalize_deferred_flowvn_test_preprocess(batch: dict):
    if "kdata_p1_unnorm" not in batch:
        return batch
    imdata_p1, kdata_p1, norm = finalize_flowvn_deferred_adjoint(
        batch["kdata_p1_unnorm"],
        batch["coil_sens"],
        batch["mask"],
        fe_ifft=_scalar_bool(batch.get("deferred_gpu_preprocess_fe_ifft", False)),
    )
    batch["imdata_p1"] = imdata_p1
    batch["kdata_p1"] = kdata_p1
    batch["norm"] = norm
    del batch["kdata_p1_unnorm"]
    del batch["mask"]
    return batch


_FLOWVN_ACTIVATION_NAMES = (
    "activation1",
    "activation2",
    "activation3",
    "activation4",
    "activation5",
)

DEFAULT_FLOWVN_TEACHER_CKPT = "checkpoints/flowvn_basline_16-epochepoch=019.ckpt"


def _compile_flowvn_activation_modules(flowvn: nn.Module, compile_mode: str):
    if not hasattr(flowvn, "cell_list"):
        raise ValueError("--compile_flowvn_activations requires canonical FlowVN")
    for cell in flowvn.cell_list:
        for activation_name in _FLOWVN_ACTIVATION_NAMES:
            module = getattr(cell, activation_name)
            setattr(cell, activation_name, torch.compile(module, mode=compile_mode))
    return flowvn


def _compile_flowvn_regularizer_modules(flowvn: nn.Module, compile_mode: str):
    if not hasattr(flowvn, "cell_list"):
        raise ValueError("--compile_flowvn_regularizer requires canonical FlowVN")
    for cell in flowvn.cell_list:
        cell.compile_regularizer(compile_mode=compile_mode)
    return flowvn


def _maybe_compile_flowvn_module(flowvn: nn.Module, args: dict):
    compile_model = bool(args.get("compile_model", False))
    compile_flowvn_activations = bool(args.get("compile_flowvn_activations", False))
    compile_flowvn_regularizer = bool(args.get("compile_flowvn_regularizer", False))
    compile_mode = str(args.get("compile_mode", "default"))
    if compile_flowvn_activations:
        flowvn = _compile_flowvn_activation_modules(flowvn, compile_mode)
    if compile_flowvn_regularizer:
        flowvn = _compile_flowvn_regularizer_modules(flowvn, compile_mode)
    if compile_model:
        flowvn = torch.compile(flowvn, mode=compile_mode)
    return flowvn


def _maybe_compile_flowvn(model: "UnrolledNetwork", args: dict):
    compile_model = bool(args.get("compile_model", False))
    compile_flowvn_activations = bool(args.get("compile_flowvn_activations", False))
    compile_flowvn_regularizer = bool(args.get("compile_flowvn_regularizer", False))
    if (
        not compile_model
        and not compile_flowvn_activations
        and not compile_flowvn_regularizer
    ):
        return model
    if args.get("network") != "FlowVN":
        raise ValueError("FlowVN compile options are only supported for network=FlowVN")
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")
    model.network = _maybe_compile_flowvn_module(model.network, args)
    teacher_network = model.teacher_network
    if teacher_network is not None:
        teacher_network.model = _maybe_compile_flowvn_module(
            teacher_network.model, args
        )
    return model


class ParamGradTensorBoardCallback(pl.Callback):
    """
    Log:
    - Histograms of each parameter (optional)
    - Histograms of each parameter gradient (optional)
    - Scalar stats such as norm, mean, std, max for parameters/gradients
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_hist: bool = False,
        log_stats: bool = True,
        max_params: Optional[int] = None,
        grad_none_as_zero: bool = False,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_hist = log_hist
        self.log_stats = log_stats
        self.max_params = max_params
        self.grad_none_as_zero = grad_none_as_zero

    def _should_log(self, trainer: "pl.Trainer"):
        return (
            trainer.global_step % self.log_every_n_steps
        ) == 0 and trainer.global_step > 0

    @torch.no_grad()
    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if not trainer.training:
            return
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        if not self._should_log(trainer):
            return

        writer = trainer.logger.experiment
        step = trainer.global_step

        n = 0
        for name, p in pl_module.named_parameters():
            if (self.max_params is not None) and (n >= self.max_params):
                break
            n += 1

            if self.log_hist:
                writer.add_histogram(f"params/{name}", p.detach().float().cpu(), step)

            if self.log_stats:
                pdata = p.detach().float()
                writer.add_scalar(f"params_norm/{name}", pdata.norm().item(), step)
                writer.add_scalar(
                    f"params_absmax/{name}", pdata.abs().max().item(), step
                )
                writer.add_scalar(f"params_mean/{name}", pdata.mean().item(), step)
                writer.add_scalar(
                    f"params_std/{name}", pdata.std(unbiased=False).item(), step
                )

            g = p.grad
            if g is None:
                if not self.grad_none_as_zero:
                    writer.add_scalar(f"grads_is_none/{name}", 1.0, step)
                    continue
                g = torch.zeros_like(p)

            gdata = g.detach().float()
            if self.log_hist:
                writer.add_histogram(f"grads/{name}", gdata.cpu(), step)

            if self.log_stats:
                writer.add_scalar(f"grads_norm/{name}", gdata.norm().item(), step)
                writer.add_scalar(
                    f"grads_absmax/{name}", gdata.abs().max().item(), step
                )
                writer.add_scalar(f"grads_mean/{name}", gdata.mean().item(), step)
                writer.add_scalar(
                    f"grads_std/{name}", gdata.std(unbiased=False).item(), step
                )

        writer.flush()


class RuntimeStatsCallback(pl.Callback):
    """Log phase wall time and process/GPU peak memory to every configured logger."""

    _GIB = float(1024**3)
    _PHASE_METRIC_KEYS = {
        "train": (
            "system/train_epoch_seconds",
            "system/train_peak_cuda_allocated_gib",
            "system/train_peak_cuda_reserved_gib",
        ),
        "validation": (
            "system/validation_epoch_seconds",
            "system/validation_peak_cuda_allocated_gib",
            "system/validation_peak_cuda_reserved_gib",
        ),
    }

    def __init__(self):
        super().__init__()
        self._phase_started_at = {}
        self._phase_cuda_peaks = {}

    @staticmethod
    def _process_peak_rss_gib() -> float:
        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss *= 1024.0
        return peak_rss / RuntimeStatsCallback._GIB

    def _start_phase(self, phase: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._phase_started_at[phase] = time.perf_counter()
        self._phase_cuda_peaks[phase] = (0.0, 0.0)

    def _capture_cuda_peaks(self, phase: str):
        if not torch.cuda.is_available() or phase not in self._phase_started_at:
            return
        allocated = float(torch.cuda.max_memory_allocated()) / self._GIB
        reserved = float(torch.cuda.max_memory_reserved()) / self._GIB
        previous = self._phase_cuda_peaks.get(phase, (0.0, 0.0))
        self._phase_cuda_peaks[phase] = (
            max(previous[0], allocated),
            max(previous[1], reserved),
        )

    def _finish_phase(self, trainer: "pl.Trainer", phase: str):
        started_at = self._phase_started_at.get(phase)
        if started_at is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._capture_cuda_peaks(phase)
        self._phase_started_at.pop(phase, None)
        allocated, reserved = self._phase_cuda_peaks.pop(phase, (0.0, 0.0))
        seconds_key, allocated_key, reserved_key = self._PHASE_METRIC_KEYS[phase]
        metrics = {
            seconds_key: float(time.perf_counter() - started_at),
            allocated_key: allocated,
            reserved_key: reserved,
            "system/process_peak_rss_gib": self._process_peak_rss_gib(),
        }
        if torch.cuda.is_available():
            metrics["system/gpu_total_memory_gib"] = (
                float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
                / self._GIB
            )
        if getattr(trainer, "global_rank", 0) == 0:
            for logger in getattr(trainer, "loggers", []):
                logger.log_metrics(metrics, step=int(trainer.global_step))
            values = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"[RUNTIME] phase={phase} {values}")

    def on_train_epoch_start(self, trainer, pl_module):
        self._start_phase("train")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._capture_cuda_peaks("train")
        try:
            is_last_batch = int(batch_idx) + 1 >= int(trainer.num_training_batches)
        except (TypeError, ValueError, OverflowError):
            is_last_batch = False
        if is_last_batch:
            self._finish_phase(trainer, "train")

    def on_train_epoch_end(self, trainer, pl_module):
        self._finish_phase(trainer, "train")

    def on_validation_epoch_start(self, trainer, pl_module):
        self._start_phase("validation")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._capture_cuda_peaks("validation")
        pl_module._maybe_empty_inference_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._finish_phase(trainer, "validation")


class CMRSaveCallback(pl.Callback):
    def __init__(self):
        self._cache = {}

    def _key(self, meta: dict):
        out_dir = str(meta.get("out_dir", ""))
        case_dir = str(meta.get("case_dir", ""))
        R = int(meta.get("usrate", 0))
        base = out_dir if out_dir not in ("", "None") else case_dir
        return (base, R)

    def _flush_one(self, base_out: str, R: int):
        expected_segs = [0, 1, 2, 3]
        pack = self._cache.get((base_out, R), None)
        if pack is None:
            return

        recon_map = pack["recon"]
        seg_map = pack["seg"]
        missing = [i for i in expected_segs if i not in recon_map]
        if missing:
            return

        img = np.stack([recon_map[i] for i in expected_segs], axis=0)
        img = np.transpose(img, (0, 1, 4, 3, 2))

        s = seg_map[0]
        s = np.transpose(s, (2, 1, 0))
        s = s[None, None, :, :, :]

        out_dir = Path(base_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("SAVE", out_dir)
        save_coo_npz(str(out_dir / f"img_ktGaussian{R}.npz"), img * s)

        csv_path = out_dir / f"recontime_ktGaussian{R}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["recontime"])
            w.writerow([pack["recon_ms_sum"] / 1000.0])

        del self._cache[(base_out, R)]

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        meta = {k: outputs.get(k) for k in outputs.keys() if k != "recon"}
        seg_idx = int(meta.get("seg_idx", 0))
        recon_ms = meta.get("recon_ms", 0.0)

        k = self._key(meta)
        if k not in self._cache:
            self._cache[k] = {"recon": {}, "seg": {}, "meta": meta, "recon_ms_sum": 0.0}

        if recon_ms is not None:
            self._cache[k]["recon_ms_sum"] += float(recon_ms)

        recon = outputs["recon"]
        x = recon[0] if recon.ndim == 5 else recon
        x_np = x.numpy()

        seg = batch["segmentation"]
        if hasattr(seg, "detach"):
            seg = seg.detach().cpu().numpy()
        seg = seg.astype(bool)[0]

        self._cache[k]["recon"][seg_idx] = x_np
        self._cache[k]["seg"][seg_idx] = seg

        base_out, R = k
        self._flush_one(base_out, R)

    def on_test_epoch_end(self, trainer, pl_module):
        for base_out, R in list(self._cache.keys()):
            self._flush_one(base_out, R)
        self._cache.clear()


class UnrolledNetwork(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        self.log_img_count = 0

        if self.options["network"] == "FlowVN":
            use_lowmem = bool(self.options.get("lowmem", False)) and (
                self.options.get("mode") == "test"
            )
            if use_lowmem:
                self.network = FlowVNLowMem(**self.options)
            else:
                self.network = FlowVN(**self.options)
        else:
            raise ValueError("Only network=FlowVN is supported")

        self.L1_loss = nn.L1Loss()
        self.teacher_initialization_report = None
        if bool(self.options.get("teacher_initialize_student", False)) and (
            self.options.get("mode") == "train"
        ):
            if self.options.get("network") != "FlowVN":
                raise ValueError(
                    "teacher_initialize_student is only supported for FlowVN"
                )
            self.teacher_initialization_report = (
                initialize_student_from_teacher_checkpoint(
                    self.network,
                    self.options.get("teacher_ckpt_path"),
                    teacher_num_stages=int(
                        self.options.get("teacher_num_stages", 16)
                    ),
                )
            )
        object.__setattr__(self, "teacher_network", None)
        if bool(self.options.get("distill_enabled", False)) and (
            self.options.get("mode") == "train"
        ):
            # Keep the teacher out of Lightning state_dict/optimizer ownership.
            object.__setattr__(
                self,
                "teacher_network",
                FlowVNTeacher(
                    self.options, self.options.get("teacher_ckpt_path"), self.device
                ),
            )
            object.__setattr__(
                self,
                "distillation_loss_helper",
                DistillationLossHelper(self.options, self.device),
            )

        self._vis_done_per_R = {}
        self._fixed_vis_case = None
        self.vis_nv_idx = 0
        self.vis_nt_idx = 0
        self._flow_metric_warned = False
        self._val_accumulator = None
        self._train_vis_case = None
        self._train_image_logged_epoch = -1
        self._test_log_count = 0

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected_keys = super().state_dict().keys()
        remapped = remap_state_dict_for_expected_keys(state_dict, expected_keys)
        return super().load_state_dict(remapped, strict=strict, assign=assign)

    def on_fit_start(self):  # This hook does not run during testing.
        pass

    def on_train_epoch_start(self):
        # Reset fixed train visualization target each epoch so image logging
        # is not locked to only the very first epoch's sample.
        self._train_vis_case = None

    def _to_metrics_predgt(self, x: torch.Tensor) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu()
        x_np = x.numpy()
        if x_np.ndim != 5:
            raise RuntimeError(
                f"Expected 5D tensor for metrics (Nv,Nt,FE,PE,SPE), got {x_np.shape}"
            )
        return np.transpose(x_np, (0, 1, 4, 3, 2))

    def _to_metrics_seg(self, seg: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if torch.is_tensor(seg):
            seg = seg.detach().cpu().numpy()[0]
        seg = seg.astype(bool)
        return np.transpose(seg, (2, 1, 0))

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - x.min()
        d = x.max() - x.min()
        return x / (d if d > 0 else 1.0)

    @staticmethod
    def _to_venc_np(
        venc: Union[torch.Tensor, np.ndarray, list, tuple, None], n_flow: int
    ) -> Optional[np.ndarray]:
        if venc is None or n_flow <= 0:
            return None
        if torch.is_tensor(venc):
            venc = venc.detach().cpu().numpy()
        venc_np = np.asarray(venc, dtype=np.float32)
        if venc_np.ndim > 1:
            venc_np = venc_np[0]
        venc_np = venc_np.reshape(-1)
        if venc_np.size < n_flow:
            return None
        return venc_np[:n_flow]

    def _to_metrics_single_nv(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert one-encoding tensor to (Nt,SPE,PE,FE) for epoch-level grouping."""
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)

        # Accept common shapes produced by Lightning/DataLoader with batch=1.
        if x.ndim == 6 and x.shape[0] == 1:
            x = x[0]
        if x.ndim == 5 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 4:
            raise RuntimeError(
                f"Expected single-encoding tensor with 4 dims (Nt,FE,PE,SPE), got {x.shape}"
            )

        return np.transpose(x, (0, 3, 2, 1))

    def _to_metrics_all_nv(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert multi-encoding tensor to (Nv,Nt,SPE,PE,FE) for grouped validation metrics."""
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)

        # Accept [B,Nv,Nt,FE,PE,SPE] or [Nv,Nt,FE,PE,SPE].
        if x.ndim == 6 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 5:
            raise RuntimeError(
                f"Expected multi-encoding tensor with 5 dims (Nv,Nt,FE,PE,SPE), got {x.shape}"
            )

        return np.transpose(x, (0, 1, 4, 3, 2))

    def _val_group_key(self, batch: dict):
        case_dir = (
            batch["case_dir"][0]
            if isinstance(batch["case_dir"], (list, tuple))
            else batch["case_dir"]
        )
        slice_start = (
            int(batch["slice_start"][0])
            if hasattr(batch["slice_start"], "__len__")
            else int(batch["slice_start"])
        )
        usrate = (
            int(batch["usrate"][0])
            if hasattr(batch["usrate"], "__len__")
            else int(batch["usrate"])
        )
        return (str(case_dir), slice_start, usrate)

    def _evaluate_val_group(
        self,
        key,
        pred_np: np.ndarray,
        gt_np: np.ndarray,
        seg_np: np.ndarray,
        venc: Optional[np.ndarray],
        want_visualization: bool,
    ):
        pred_np, gt_np = self._maybe_apply_phase_correction(pred_np, gt_np)

        # Preserve the established challenge metric convention. RelErr and AngErr
        # are evaluated on phase differences; VENC remains available in the CSV
        # provenance but is not applied here so old and new validation agree.
        mag_pred, flow_pred = complex2magflow(pred_np)
        mag_gt, flow_gt = complex2magflow(gt_np)

        metrics = {
            "nrmse": None,
            "ssim": None,
            "relerr": None,
            "angerr": None,
            "velocity_vector_rmse_cm_s": None,
        }
        try:
            metrics["nrmse"] = float(nRMSE(mag_pred, mag_gt, seg_np))
        except Exception:
            pass
        try:
            metrics["ssim"] = float(SSIM(mag_pred, mag_gt, seg_np))
        except Exception:
            pass
        if flow_pred.shape[0] > 0 and flow_gt.shape[0] > 0:
            try:
                metrics["relerr"] = float(RelErr(flow_pred, flow_gt, seg_np))
                metrics["angerr"] = float(AngErr(flow_pred, flow_gt, seg_np))
            except Exception:
                pass
            if venc is not None:
                try:
                    metrics["velocity_vector_rmse_cm_s"] = float(
                        VelocityVectorRMSE(
                            flow_pred,
                            flow_gt,
                            venc_cm_s=venc,
                            segmask=seg_np,
                        )
                    )
                except Exception:
                    pass

        visualization = None
        if want_visualization:
            source = build_qualitative_source(
                mag_pred=mag_pred,
                mag_gt=mag_gt,
                flow_pred=flow_pred,
                flow_gt=flow_gt,
                segmentation=seg_np,
                time_index=None,
            )
            if venc is not None:
                source["venc"] = np.asarray(venc, dtype=np.float32)
            panel = self._make_val_album_panel(
                {
                    "mag_pred": mag_pred,
                    "mag_gt": mag_gt,
                    "flow_pred": flow_pred,
                    "flow_gt": flow_gt,
                    "seg": seg_np,
                    "time_index": int(source["time_index"]),
                }
            )
            visualization = {"panel": panel, "source": source}
        return metrics, visualization

    def _update_val_group_cache(
        self,
        recon_eval: torch.Tensor,
        gt_eval: torch.Tensor,
        batch: dict,
        normalized_l1: Optional[float] = None,
    ):
        if self._val_accumulator is None:
            self._val_accumulator = StreamingValidationAccumulator(
                self._evaluate_val_group
            )
        key = self._val_group_key(batch)
        seg_idx = (
            int(batch["seg_idx"][0])
            if hasattr(batch["seg_idx"], "__len__")
            else int(batch["seg_idx"])
        )

        seg_np = self._to_metrics_seg(batch["segmentation"])

        venc = self._to_venc_np(batch.get("VENC", None), n_flow=3)

        if seg_idx >= 0:
            pred_nv = self._to_metrics_single_nv(recon_eval)
            gt_nv = self._to_metrics_single_nv(gt_eval)
            self._val_accumulator.add_encoding(
                key=key,
                encoding=seg_idx,
                pred=pred_nv,
                gt=gt_nv,
                segmentation=seg_np,
                venc=venc,
                normalized_l1=normalized_l1,
            )
            return

        pred_all = self._to_metrics_all_nv(recon_eval)
        gt_all = self._to_metrics_all_nv(gt_eval)
        n_enc = min(pred_all.shape[0], gt_all.shape[0])
        for enc_idx in range(n_enc):
            self._val_accumulator.add_encoding(
                key=key,
                encoding=int(enc_idx),
                pred=pred_all[enc_idx],
                gt=gt_all[enc_idx],
                segmentation=seg_np,
                venc=venc,
                normalized_l1=normalized_l1,
            )

    def _compute_full_val_epoch_metrics(self):
        if self._val_accumulator is None:
            return {
                "nrmse": None,
                "ssim": None,
                "relerr": None,
                "angerr": None,
                "velocity_vector_rmse_cm_s": None,
                "normalized_l1": None,
                "n_groups": 0,
                "n_complete": 0,
                "n_incomplete": 0,
                "max_pending_groups": 0,
                "vis_payload": None,
                "vis_payloads": [],
            }
        return {
            **self._val_accumulator.summary(),
            "vis_payload": None,
            "vis_payloads": self._val_accumulator.visualizations,
        }

    def _validation_metrics_output_path(self) -> Path:
        configured = self.options.get("val_metrics_output", None)
        if configured not in (None, "", "None"):
            output_path = Path(str(configured))
        else:
            output_path = Path(self.options.get("save_dir", "outputs/exp")) / "validation_metrics.csv"

        if self.options.get("mode") == "train":
            output_path = output_path.with_name(
                f"{output_path.stem}_epoch{int(self.current_epoch):03d}{output_path.suffix}"
            )
        return output_path

    def _maybe_empty_inference_cache(self):
        if (
            bool(self.options.get("inference_empty_cache", False))
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

    def _maybe_apply_phase_correction(
        self, pred_np: np.ndarray, gt_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Match demo evaluation: estimate background phase from GT and apply to both GT and prediction.
        Skip safely if the sample does not contain enough velocity encodings or correction fails.
        """
        if gt_np.shape[0] < 2:
            return pred_np, gt_np

        try:
            corr_maps = execute_MSAC(gt_np, corr_fit_order=3, th=0.1)
            if corr_maps.shape == gt_np[1:].shape:
                phase_corr = np.exp(-1j * corr_maps)
                pred_np = pred_np.copy()
                gt_np = gt_np.copy()
                pred_np[1:] *= phase_corr
                gt_np[1:] *= phase_corr
        except Exception:
            # Keep training robust; metrics still computed without phase correction.
            pass

        return pred_np, gt_np

    def _log_val_images(self, recon, gt, batch, usrate):
        if self.global_rank != 0:
            return

        subj = (
            batch["subj"][0]
            if isinstance(batch["subj"], (list, tuple))
            else batch["subj"]
        )
        slice_start = (
            int(batch["slice_start"][0])
            if hasattr(batch["slice_start"], "__len__")
            else int(batch["slice_start"])
        )
        seg_idx = (
            int(batch["seg_idx"][0])
            if hasattr(batch["seg_idx"], "__len__")
            else int(batch["seg_idx"])
        )
        case_id = (subj, slice_start, seg_idx)

        if self._fixed_vis_case is None:
            self._fixed_vis_case = case_id

        if case_id != self._fixed_vis_case:
            return
        if self._vis_done_per_R.get(int(usrate), False):
            return

        self._vis_done_per_R[int(usrate)] = True

        pred = recon.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        if pred.ndim == 6:
            pred = pred[0]
            gt_np = gt_np[0]
        if pred.ndim == 5:
            Nv, Nt, FE, PE, SPE = pred.shape
        else:
            raise RuntimeError(f"Unexpected pred shape: {pred.shape}")

        nv = int(np.clip(self.vis_nv_idx, 0, Nv - 1))
        nt = int(np.clip(self.vis_nt_idx, 0, Nt - 1))

        # Use the FlowVN visualization convention.
        spe = SPE // 2
        pred_mag = np.abs(pred[nv, nt, :, :, spe])
        gt_mag = np.abs(gt_np[nv, nt, :, :, spe])
        base = f"val_images/usrate{int(usrate)}/{subj}_slice{slice_start}_seg{seg_idx}_nv{nv}_nt{nt}_spe{spe}"

        pred_vis = self._norm01(pred_mag)
        gt_vis = self._norm01(gt_mag)

        step = self.global_step

        self._log_image_to_all_loggers(f"{base}/gt", gt_vis, step)
        self._log_image_to_all_loggers(f"{base}/pred", pred_vis, step)
        self._log_image_to_all_loggers(
            f"{base}/absdiff", self._norm01(np.abs(pred_mag - gt_mag)), step
        )

        input_vol = batch["imdata_p1"]
        if torch.is_tensor(input_vol):
            input_vol = (input_vol * batch["norm"]).detach().cpu().numpy()
        while np.asarray(input_vol).ndim > 4:
            input_vol = np.asarray(input_vol)[0]
        in_mag = np.abs(input_vol[nt, :, :, spe])
        self._log_image_to_all_loggers(f"{base}/input", self._norm01(in_mag), step)

    def _should_log_train_images(self) -> bool:
        every = int(self.options.get("log_images_every_n_steps", 500))
        if every <= 0:
            return False
        # Emit one train image panel per epoch for stable monitoring.
        return self._train_image_logged_epoch != int(self.current_epoch)

    def _should_compute_train_metrics(self) -> bool:
        metric_interval = int(self.options.get("train_metric_interval", 1))
        if metric_interval <= 0:
            return False
        return self.global_step % metric_interval == 0

    def _log_train_images(self, recon, gt, batch) -> bool:
        if self.global_rank != 0:
            return False

        subj = (
            batch["subj"][0]
            if isinstance(batch["subj"], (list, tuple))
            else batch["subj"]
        )
        slice_start = (
            int(batch["slice_start"][0])
            if hasattr(batch["slice_start"], "__len__")
            else int(batch["slice_start"])
        )
        case_id = (subj, slice_start)
        if self._train_vis_case is None:
            self._train_vis_case = case_id
        if case_id != self._train_vis_case:
            return False

        def _to4d(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            while x.ndim > 4:
                x = x[0]
            return x

        recon4 = _to4d(recon * batch["norm"])
        gt4 = _to4d(gt * batch["norm"])
        in4 = _to4d(batch["imdata_p1"] * batch["norm"])

        nt = int(np.clip(self.vis_nt_idx, 0, recon4.shape[0] - 1))
        # Use the FlowVN visualization convention.
        spe = recon4.shape[-1] // 2
        pred_mag = np.abs(recon4[nt, :, :, spe])
        gt_mag = np.abs(gt4[nt, :, :, spe])
        in_mag = np.abs(in4[nt, :, :, spe])
        base = f"train_images/{subj}_slice{slice_start}_nt{nt}_spe{spe}"
        diff = np.abs(pred_mag - gt_mag)

        step = self.global_step
        self._log_image_to_all_loggers(f"{base}/input", self._norm01(in_mag), step)
        self._log_image_to_all_loggers(f"{base}/pred", self._norm01(pred_mag), step)
        self._log_image_to_all_loggers(f"{base}/gt", self._norm01(gt_mag), step)
        self._log_image_to_all_loggers(f"{base}/absdiff", self._norm01(diff), step)
        return True

    def _log_image_to_all_loggers(self, tag: str, img2d: np.ndarray, step: int):
        if self.global_rank != 0:
            return
        img2d = np.asarray(img2d, dtype=np.float32)

        all_loggers = []
        if hasattr(self, "trainer") and getattr(self.trainer, "loggers", None):
            all_loggers = list(self.trainer.loggers)
        elif self.logger is not None:
            all_loggers = [self.logger]

        for lg in all_loggers:
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_image(tag, torch.from_numpy(img2d[None, :, :]), step)
            elif isinstance(lg, WandbLogger) and wandb is not None:
                lg.experiment.log({tag: wandb.Image(img2d), "global_step": step})

    def _log_val_flow_images(self, vis_payload: Optional[dict], step: int):
        if self.global_rank != 0 or vis_payload is None:
            return

        mag_pred = vis_payload["mag_pred"]
        mag_gt = vis_payload["mag_gt"]
        flow_pred = vis_payload["flow_pred"]
        flow_gt = vis_payload["flow_gt"]
        seg = vis_payload["seg"].astype(np.float32)

        nt = int(
            np.clip(
                vis_payload.get("time_index", self.vis_nt_idx),
                0,
                mag_pred.shape[1] - 1,
            )
        )
        spe = mag_pred.shape[2] // 2
        roi2d = seg[spe].T

        # Log magnitude for all available encodings (typically Nv=4).
        n_mag = min(mag_pred.shape[0], mag_gt.shape[0])
        for nv in range(n_mag):
            mag_pred2d = np.abs(mag_pred[nv, nt, spe]).T * roi2d
            mag_gt2d = np.abs(mag_gt[nv, nt, spe]).T * roi2d
            self._log_image_to_all_loggers(
                f"val_full_images/mag_nv{nv}/pred", self._norm01(mag_pred2d), step
            )
            self._log_image_to_all_loggers(
                f"val_full_images/mag_nv{nv}/gt", self._norm01(mag_gt2d), step
            )
            self._log_image_to_all_loggers(
                f"val_full_images/mag_nv{nv}/absdiff",
                self._norm01(np.abs(mag_pred2d - mag_gt2d)),
                step,
            )

            # Backward-compatible aliases for dashboards expecting the old single-mag keys.
            if nv == 0:
                self._log_image_to_all_loggers(
                    "val_full_images/mag/pred", self._norm01(mag_pred2d), step
                )
                self._log_image_to_all_loggers(
                    "val_full_images/mag/gt", self._norm01(mag_gt2d), step
                )
                self._log_image_to_all_loggers(
                    "val_full_images/mag/absdiff",
                    self._norm01(np.abs(mag_pred2d - mag_gt2d)),
                    step,
                )

        n_dirs = min(flow_pred.shape[0], 3)
        for d in range(n_dirs):
            pred2d = flow_pred[d, nt, spe].T * roi2d
            gt2d = flow_gt[d, nt, spe].T * roi2d
            err2d = (pred2d - gt2d) * roi2d

            vmax = float(
                np.max(np.abs(np.concatenate([pred2d.reshape(-1), gt2d.reshape(-1)])))
            )
            vmax = max(vmax, 1e-6)
            self._log_image_to_all_loggers(
                f"val_full_images/flow_dir{d + 1}/pred",
                np.clip(pred2d / vmax, -1.0, 1.0),
                step,
            )
            self._log_image_to_all_loggers(
                f"val_full_images/flow_dir{d + 1}/gt",
                np.clip(gt2d / vmax, -1.0, 1.0),
                step,
            )
            self._log_image_to_all_loggers(
                f"val_full_images/flow_dir{d + 1}/err",
                np.clip(err2d / vmax, -1.0, 1.0),
                step,
            )

    def _make_val_album_panel(self, payload: dict) -> np.ndarray:
        """Build a compact montage for one full sample: magnitude + flow rows with pred/gt/error columns."""
        mag_pred = payload["mag_pred"]
        mag_gt = payload["mag_gt"]
        flow_pred = payload["flow_pred"]
        flow_gt = payload["flow_gt"]
        seg = payload["seg"].astype(np.float32)

        nt = int(
            np.clip(
                payload.get("time_index", self.vis_nt_idx),
                0,
                mag_pred.shape[1] - 1,
            )
        )
        spe = mag_pred.shape[2] // 2
        roi2d = seg[spe].T

        rows = []

        # Magnitude rows for each encoding.
        for nv in range(mag_pred.shape[0]):
            pred2d = np.abs(mag_pred[nv, nt, spe]).T * roi2d
            gt2d = np.abs(mag_gt[nv, nt, spe]).T * roi2d
            err2d = np.abs(pred2d - gt2d)
            row = np.concatenate(
                [
                    (self._norm01(pred2d) * 255).astype(np.uint8),
                    (self._norm01(gt2d) * 255).astype(np.uint8),
                    (self._norm01(err2d) * 255).astype(np.uint8),
                ],
                axis=1,
            )
            rows.append(row)

        # Flow rows for each available velocity direction.
        n_dirs = min(flow_pred.shape[0], flow_gt.shape[0], 3)
        for d in range(n_dirs):
            pred2d = flow_pred[d, nt, spe].T * roi2d
            gt2d = flow_gt[d, nt, spe].T * roi2d
            err2d = (pred2d - gt2d) * roi2d
            vmax = float(
                np.max(np.abs(np.concatenate([pred2d.reshape(-1), gt2d.reshape(-1)])))
            )
            vmax = max(vmax, 1e-6)

            pred_u8 = ((np.clip(pred2d / vmax, -1.0, 1.0) + 1.0) * 127.5).astype(
                np.uint8
            )
            gt_u8 = ((np.clip(gt2d / vmax, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
            err_u8 = ((np.clip(err2d / vmax, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
            row = np.concatenate([pred_u8, gt_u8, err_u8], axis=1)
            rows.append(row)

        return np.concatenate(rows, axis=0)

    def _log_val_wandb_album(self, vis_payloads: list[dict], step: int):
        if self.global_rank != 0 or len(vis_payloads) == 0:
            return

        all_loggers = []
        if hasattr(self, "trainer") and getattr(self.trainer, "loggers", None):
            all_loggers = list(self.trainer.loggers)
        elif self.logger is not None:
            all_loggers = [self.logger]

        for lg in all_loggers:
            if isinstance(lg, WandbLogger) and wandb is not None:
                images = []
                for payload in vis_payloads[:5]:
                    panel = payload.get("panel", None)
                    if panel is None:
                        panel = self._make_val_album_panel(payload)
                    meta = payload.get("meta", {})
                    cap = (
                        f"case={meta.get('case_id', Path(str(meta.get('case_dir', 'unknown'))).name)} "
                        f"slice={int(meta.get('slice_start', -1))} "
                        f"R={int(meta.get('usrate', -1))} "
                        "rows: mag(v0..v3), then flow(dir1..dir3); cols: pred|gt|err"
                    )
                    images.append(wandb.Image(panel, caption=cap))

                if len(images) > 0:
                    lg.experiment.log({"val_full/album": images, "global_step": step})

            elif isinstance(lg, TensorBoardLogger):
                # Keep TensorBoard usable too, one panel per sample.
                for i, payload in enumerate(vis_payloads[:5]):
                    panel = payload.get("panel", None)
                    if panel is None:
                        panel = self._make_val_album_panel(payload)
                    panel = panel.astype(np.float32) / 255.0
                    lg.experiment.add_image(
                        f"val_full_album/sample_{i}",
                        torch.from_numpy(panel[None, :, :]),
                        step,
                    )

    def _log_test_images(self, batch: dict, recon_img_complex: torch.Tensor, step: int):
        if self.global_rank != 0:
            return
        if not bool(self.options.get("test_log_images", False)):
            return
        max_images = int(self.options.get("test_log_max_images", 3))
        if self._test_log_count >= max_images:
            return

        subj = (
            batch["subj"][0]
            if isinstance(batch["subj"], (list, tuple))
            else batch["subj"]
        )
        usrate = (
            int(batch["usrate"][0])
            if hasattr(batch["usrate"], "__len__")
            else int(batch["usrate"])
        )

        # Input is the zero-filled adjoint reconstruction.
        inp = batch["imdata_p1"]
        if torch.is_tensor(inp):
            inp = inp.detach().cpu().numpy()
        if inp.ndim >= 5:
            inp = inp[0]

        rec = recon_img_complex.detach().cpu().numpy()

        # Both inputs are (Nv, Nt, FE, PE, SPE) for FlowVN.
        if inp.ndim == 5:
            nt = inp.shape[1] // 2
            spe = inp.shape[-1] // 2
            in2d = np.abs(inp[0, nt, :, :, spe])
        else:
            in2d = np.abs(inp.squeeze())

        if rec.ndim == 5:
            nt = rec.shape[1] // 2
            spe = rec.shape[-1] // 2
            rec2d = np.abs(rec[0, nt, :, :, spe])
        else:
            rec2d = np.abs(rec.squeeze())

        base = f"test_images/{subj}/R{usrate}"
        self._log_image_to_all_loggers(f"{base}/input", self._norm01(in2d), step)
        self._log_image_to_all_loggers(f"{base}/recon", self._norm01(rec2d), step)
        self._test_log_count += 1

    @torch.no_grad()
    def _compute_challenge_metrics(
        self,
        recon: torch.Tensor,
        gt: torch.Tensor,
        batch: dict,
        compute_flow: bool = True,
    ) -> dict:
        """Compute challenge metrics on one batch using magnitude/flow decomposition."""
        if (
            recon.ndim == 6
            and gt.ndim == 6
            and recon.shape[0] > 1
            and recon.shape[1] == gt.shape[0]
        ):
            # exp_loss=True in training returns all stages; evaluate final stage for metrics.
            recon = recon[-1]

        # Convert to per-sample tensors in (Nv, Nt, FE, PE, SPE).
        if recon.ndim == 6:
            recon = recon[0]
        if gt.ndim == 6:
            gt = gt[0]
        if recon.ndim != 5 or gt.ndim != 5:
            raise RuntimeError(
                f"Unexpected recon/gt dims for metrics: recon={tuple(recon.shape)}, gt={tuple(gt.shape)}"
            )

        recon_eval = recon * batch["norm"]
        gt_eval = gt * batch["norm"]

        pred_np = self._to_metrics_predgt(recon_eval)
        gt_np = self._to_metrics_predgt(gt_eval)
        seg_np = self._to_metrics_seg(batch["segmentation"])

        # Some sampled FE blocks can have empty segmask. Fallback to full-mask for stable monitoring.
        if np.sum(seg_np.astype(np.int64)) == 0:
            seg_np = np.ones_like(seg_np, dtype=bool)

        pred_np, gt_np = self._maybe_apply_phase_correction(pred_np, gt_np)

        n_flow = max(pred_np.shape[0] - 1, 0)
        venc_use = self._to_venc_np(batch.get("VENC", None), n_flow=n_flow)

        mag_pred, flow_pred = complex2magflow(pred_np, venc=venc_use)
        mag_gt, flow_gt = complex2magflow(gt_np, venc=venc_use)

        out = {
            "nrmse": float(nRMSE(mag_pred, mag_gt, seg_np)),
            "ssim": float(SSIM(mag_pred, mag_gt, seg_np)),
            "relerr": None,
            "angerr": None,
            "flow_available": False,
        }

        if not np.isfinite(out["nrmse"]):
            out["nrmse"] = None
        if not np.isfinite(out["ssim"]):
            out["ssim"] = None

        # Flow metrics require at least one flow-encoded channel beyond reference.
        if compute_flow and flow_pred.shape[0] > 0 and flow_gt.shape[0] > 0:
            rel = float(RelErr(flow_pred, flow_gt, seg_np))
            ang = float(AngErr(flow_pred, flow_gt, seg_np))
            if np.isfinite(rel) and np.isfinite(ang):
                out["relerr"] = rel
                out["angerr"] = ang
                out["flow_available"] = True

        return out

    def _flowvn_supervised_gt_loss(self, recon_img_p1, batch):
        if bool(self.options.get("exp_loss", False)):
            tau = self.current_epoch / 10
            weights = torch.exp(
                torch.tensor(
                    [
                        -tau * (self.options["num_stages"] - k + 1)
                        for k in range(self.options["num_stages"])
                    ],
                    device=recon_img_p1.device,
                    dtype=recon_img_p1.real.dtype,
                )
            )
            weights = weights / torch.sum(weights)
            return (
                torch.sum(
                    weights
                    * torch.norm(
                        recon_img_p1 - batch["gt"],
                        p=1,
                        dim=[1, 2, 3, 4, 5, 6],
                    )
                )
                / 40000
            )

        gt = batch["gt"]
        if torch.is_tensor(gt) and gt.ndim == 6 and gt.shape[1] == 1:
            gt = gt[:, 0]
        if torch.is_tensor(recon_img_p1) and recon_img_p1.ndim == 6 and recon_img_p1.shape[1] == 1:
            recon_img_p1 = recon_img_p1[:, 0]
        return self.L1_loss(recon_img_p1 - gt, torch.zeros_like(recon_img_p1))

    def _flowvn_distillation_loss(
        self,
        recon_img_p1,
        batch,
        student_intermediates=None,
    ):
        teacher_scale = self.distillation_loss_helper.teacher_schedule_scale(
            self.current_epoch
        )
        if teacher_scale <= 0.0:
            return self.distillation_loss_helper.compute_loss(
                batch,
                recon_img_p1,
                None,
                self.current_epoch,
            )

        teacher = self.teacher_network
        if teacher is None:
            raise RuntimeError("distill_enabled=True requires a loaded teacher_network")

        use_intermediates = student_intermediates is not None
        teacher_result = teacher.run_once(
            batch,
            precision_mode=self.options.get("distill_teacher_precision", "fp32"),
            return_intermediates=use_intermediates,
        )
        if use_intermediates:
            teacher_recon, teacher_intermediates = teacher_result
        else:
            teacher_recon = teacher_result
            teacher_intermediates = None

        total, losses = self.distillation_loss_helper.compute_loss(
            batch,
            recon_img_p1,
            teacher_recon,
            self.current_epoch,
            student_intermediates=student_intermediates,
            teacher_intermediates=teacher_intermediates,
        )
        return total, losses

    def training_step(self, batch):
        metric_vals = None
        if self.options["loss"] == "ssdu":
            kdata_p2 = batch["kdata_p2"]
            loss_mask = abs(kdata_p2[:, :, 0, :, 0, :, :]) != 0

            recon_img_p1 = self.network(
                (batch["imdata_p1"]),
                batch["kdata_p1"],
                batch["coil_sens"],
                batch["usrate_true"],
            )
            kdata_p1 = mri_forward_op(
                recon_img_p1, batch["coil_sens"], loss_mask.float()
            )
            loss = 0.5 * torch.norm(
                torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=2
            ) / torch.norm(torch.view_as_real(kdata_p2), p=2) + 0.5 * torch.norm(
                torch.view_as_real(kdata_p2) - torch.view_as_real(kdata_p1), p=1
            ) / torch.norm(torch.view_as_real(kdata_p2), p=1)

        elif self.options["loss"] == "supervised":
            distill_enabled = bool(
                self.options.get("distill_enabled", False)
            )
            distill_teacher_active = distill_enabled and (
                self.distillation_loss_helper.teacher_schedule_scale(
                    self.current_epoch
                )
                > 0.0
            )
            use_distill_intermediates = distill_teacher_active and (
                float(self.options.get("distill_trajectory_weight", 0.0))
                != 0.0
                or float(self.options.get("distill_update_weight", 0.0))
                != 0.0
            )
            if use_distill_intermediates:
                recon_img_p1, student_intermediates = self.network(
                    batch["imdata_p1"],
                    batch["kdata_p1"],
                    batch["coil_sens"],
                    batch["usrate_true"],
                    return_intermediates=True,
                )
            else:
                recon_img_p1 = self.network(
                    batch["imdata_p1"],
                    batch["kdata_p1"],
                    batch["coil_sens"],
                    batch["usrate_true"],
                )
                student_intermediates = None

            if distill_enabled:
                loss, distill_losses = self._flowvn_distillation_loss(
                    recon_img_p1,
                    batch,
                    student_intermediates=student_intermediates,
                )
                self.log(
                    "train/loss_gt",
                    distill_losses["student_gt_loss"],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train/loss_teacher",
                    distill_losses["image_teacher_loss"],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train/loss_kspace",
                    distill_losses["kspace_teacher_consistency_loss"],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train/loss_trajectory",
                    distill_losses["trajectory_teacher_loss"],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train/loss_update",
                    distill_losses["update_teacher_loss"],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train/distill_teacher_scale",
                    distill_losses["teacher_schedule_scale"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                self.log(
                    "train/loss_total",
                    loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
            else:
                loss = self._flowvn_supervised_gt_loss(recon_img_p1, batch)

            if self._should_compute_train_metrics():
                try:
                    metric_vals = self._compute_challenge_metrics(
                        recon_img_p1, batch["gt"], batch, compute_flow=False
                    )
                except Exception as e:
                    if self.global_rank == 0:
                        print(f"[WARN] train metrics skipped: {e}")


        self.log(
            "train/skipped_batch",
            0.0,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        self.log_dict(
            {"train_loss_epoch": loss, "step": self.current_epoch * 1.0},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if metric_vals is not None:
            if metric_vals.get("nrmse", None) is not None:
                self.log(
                    "train/nRMSE",
                    metric_vals["nrmse"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
            if metric_vals.get("ssim", None) is not None:
                self.log(
                    "train/SSIM",
                    metric_vals["ssim"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

        if self._should_log_train_images():
            recon_for_vis = (
                recon_img_p1[-1]
                if (
                    torch.is_tensor(recon_img_p1)
                    and recon_img_p1.ndim == 6
                    and recon_img_p1.shape[0] > 1
                )
                else recon_img_p1
            )
            gt_for_vis = batch["gt"][:, 0]
            did_log = self._log_train_images(recon_for_vis, gt_for_vis, batch)
            if did_log:
                self._train_image_logged_epoch = int(self.current_epoch)

        return {"loss": loss}

    def on_validation_epoch_start(self):
        self._vis_done_per_R = {}
        self._val_accumulator = StreamingValidationAccumulator(
            self._evaluate_val_group
        )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        usrate = (
            int(batch["usrate"][0])
            if hasattr(batch["usrate"], "__len__")
            else int(batch["usrate"])
        )

        recon = self.network(
            batch["imdata_p1"],
            batch["kdata_p1"],
            batch["coil_sens"],
            batch["usrate_true"],
        )
        gt = batch["gt"]

        if torch.is_tensor(gt) and gt.ndim == 6 and gt.shape[1] == 1:
            gt = gt[:, 0]
        if recon.ndim == 6 and recon.shape[1] == 1:
            recon = recon[:, 0]

        loss = self.L1_loss(recon - gt, torch.zeros_like(recon))

        self.log(
            f"val/loss_usrate{usrate}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if bool(self.options.get("val_loss_only", False)):
            return {"val_loss": loss, "usrate": usrate}

        recon_vis = recon * batch["norm"]
        gt_vis = gt * batch["norm"]
        try:
            self._update_val_group_cache(
                recon_vis,
                gt_vis,
                batch,
                normalized_l1=float(loss.detach().cpu().item()),
            )
        except Exception as e:
            if self.global_rank == 0:
                print(f"[WARN] val group cache update skipped: {e}")
        self._log_val_images(recon_vis, gt_vis, batch, usrate)

        return {"val_loss": loss, "usrate": usrate}

    def on_validation_epoch_end(self):
        if bool(self.options.get("val_loss_only", False)):
            return
        try:
            full_vals = self._compute_full_val_epoch_metrics()
            if full_vals["nrmse"] is not None:
                self.log(
                    "val/nRMSE",
                    full_vals["nrmse"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    "val_full/nRMSE",
                    full_vals["nrmse"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
            if full_vals["ssim"] is not None:
                self.log(
                    "val/SSIM",
                    full_vals["ssim"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    "val_full/SSIM",
                    full_vals["ssim"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
            if full_vals["relerr"] is not None:
                self.log(
                    "val/RelErr",
                    full_vals["relerr"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    "val_full/RelErr",
                    full_vals["relerr"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
            if full_vals["angerr"] is not None:
                self.log(
                    "val/AngErr",
                    full_vals["angerr"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    "val_full/AngErr",
                    full_vals["angerr"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            if full_vals["velocity_vector_rmse_cm_s"] is not None:
                self.log(
                    "val/VelocityVectorRMSE_cm_s",
                    full_vals["velocity_vector_rmse_cm_s"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    "val_full/VelocityVectorRMSE_cm_s",
                    full_vals["velocity_vector_rmse_cm_s"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            for usrate, rate_vals in self._val_accumulator.summary_by_usrate().items():
                for metric_key, metric_name in (
                    ("nrmse", "nRMSE"),
                    ("ssim", "SSIM"),
                    ("relerr", "RelErr"),
                    ("angerr", "AngErr"),
                    ("normalized_l1", "normalized_l1"),
                    ("velocity_vector_rmse_cm_s", "VelocityVectorRMSE_cm_s"),
                ):
                    metric_value = rate_vals.get(metric_key)
                    if metric_value is not None:
                        self.log(
                            f"val/{metric_name}_usrate{int(usrate)}",
                            metric_value,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                            add_dataloader_idx=False,
                        )

            if self.global_rank == 0:
                output_path = self._validation_metrics_output_path()
                csv_path, json_path = self._val_accumulator.write_outputs(output_path)
                print(f"[VAL FULL] metrics_csv={csv_path} summary_json={json_path}")
                visualization_path = output_path.with_name(
                    f"{output_path.stem}_visualizations.npz"
                )
                vis_path, vis_manifest_path = self._val_accumulator.write_visualizations(
                    visualization_path
                )
                print(
                    f"[VAL FULL] visualizations_npz={vis_path} "
                    f"manifest={vis_manifest_path}"
                )

            self._log_val_flow_images(
                full_vals.get("vis_payload", None), self.global_step
            )
            self._log_val_wandb_album(
                full_vals.get("vis_payloads", []), self.global_step
            )

            if self.global_rank == 0:
                print(
                    f"[VAL FULL] groups={full_vals['n_groups']} "
                    f"complete={full_vals.get('n_complete', 0)} "
                    f"incomplete={full_vals.get('n_incomplete', 0)} "
                    f"max_pending_groups={full_vals.get('max_pending_groups', 0)} "
                    f"nRMSE={full_vals['nrmse']} SSIM={full_vals['ssim']} "
                    f"RelErr={full_vals['relerr']} AngErr={full_vals['angerr']} "
                    f"VelocityVectorRMSE_cm_s="
                    f"{full_vals['velocity_vector_rmse_cm_s']}"
                )
        except Exception as e:
            if self.global_rank == 0:
                print(f"[WARN] full validation metrics skipped: {e}")
        finally:
            self._val_accumulator = None

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        recon_ms = None
        batch = _finalize_deferred_flowvn_test_preprocess(batch)
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        recon_img = self.network(
            batch["imdata_p1"],
            batch["kdata_p1"],
            batch["coil_sens"],
            batch["usrate_true"],
        )

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            recon_ms = float(start_event.elapsed_time(end_event))

        norm = batch["norm"].to(recon_img[0].device, non_blocking=True)
        recon_img_complex = (recon_img[0] * norm).detach().cpu()
        del recon_img
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        usrate = (
            int(batch["usrate"][0])
            if hasattr(batch["usrate"], "__len__")
            else int(batch["usrate"])
        )
        seg_idx = (
            int(batch["seg_idx"][0])
            if hasattr(batch["seg_idx"], "__len__")
            else int(batch["seg_idx"])
        )
        subj = (
            batch["subj"][0]
            if isinstance(batch["subj"], (list, tuple))
            else batch["subj"]
        )

        case_dir = (
            batch["case_dir"][0]
            if isinstance(batch["case_dir"], (list, tuple))
            else batch["case_dir"]
        )
        out_dir = (
            batch["out_dir"][0]
            if isinstance(batch["out_dir"], (list, tuple))
            else batch["out_dir"]
        )
        try:
            self._log_test_images(batch, recon_img_complex, step=int(batch_idx))
        except Exception as e:
            if self.global_rank == 0:
                print(f"[WARN] test image logging skipped: {e}")
        self._maybe_empty_inference_cache()
        return {
            "recon": recon_img_complex,
            "subj": subj,
            "case_dir": case_dir,
            "out_dir": out_dir,
            "seg_idx": seg_idx,
            "usrate": usrate,
            "recon_ms": recon_ms,
        }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.options["lr"])
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.options["epoch"]
                )
            },
        }


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Network arguments")

    parser.add_argument(
        "--config", type=str, default=None, help="path to YAML config file"
    )

    parser.add_argument(
        "--D_size",
        type=int,
        default=7,
        help="number of slices per volume (FlowVN only)",
    )
    parser.add_argument(
        "--T_size", type=int, default=5, help="number of cardiac bins per volume"
    )
    parser.add_argument(
        "--V_size",
        type=int,
        default=1,
        help="number of velocity encodings per sample; -1 uses all",
    )
    parser.add_argument(
        "--root_dir", type=str, default="data/own_card3d", help="directory of the data"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/exp",
        help="directory of the experiment",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpoint to test or restart training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="full Lightning checkpoint used to resume optimizer, scheduler, epoch, and global step",
    )
    parser.add_argument(
        "--input", type=str, default="", help="name of network input file"
    )

    parser.add_argument(
        "--grad_check", type=bool, default=False, help="use gradient checkpointing"
    )
    parser.add_argument(
        "--flowvn_grad_checkpoint_policy",
        "--flowvn-grad-checkpoint-policy",
        type=str,
        default="all",
        help="FlowVN train-only gradient checkpoint policy when --grad_check is enabled: all, none, first:N, last:N, every:K",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="FlowVN",
        choices=["FlowVN"],
        help="reconstruction model",
    )
    parser.add_argument(
        "--num_stages", type=int, default=10, help="number of stages in the network"
    )
    parser.add_argument(
        "--features_in", type=int, default=1, help="number of input dimensions"
    )
    parser.add_argument(
        "--features_out",
        type=int,
        default=24,
        help="number of filters for convolutional kernel",
    )

    parser.add_argument("--kernel_size", type=int, default=7, help="xyz kernel size")
    parser.add_argument(
        "--act",
        type=str,
        default="linear",
        help="what activation to use, rbf or linear",
    )
    parser.add_argument(
        "--num_act_weights",
        type=int,
        default=71,
        help="number of basis functions for activation",
    )
    parser.add_argument(
        "--grid", type=float, default=0.25, help="grid size for linear act"
    )
    parser.add_argument(
        "--weight", type=float, default=0.025, help="scale weights for RBF kernel"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-3.5,
        help="min value of filter response for rbf activation",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=3.5,
        help="max value of filter response for rbf activation",
    )
    parser.add_argument(
        "--sgd_momentum", type=bool, default=True, help="use sgd momentum"
    )
    parser.add_argument(
        "--exp_loss", type=bool, default=False, help="use exponentially weighted loss"
    )
    parser.add_argument(
        "--distill_enabled",
        "--distill-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FlowVN only: enable teacher-student distillation during supervised training",
    )
    parser.add_argument(
        "--teacher_ckpt_path",
        "--teacher-ckpt-path",
        type=str,
        default=DEFAULT_FLOWVN_TEACHER_CKPT,
        help="FlowVN distillation teacher checkpoint path; defaults to the current 16-stage pretrained checkpoint",
    )
    parser.add_argument(
        "--teacher_num_stages",
        "--teacher-num-stages",
        type=int,
        default=16,
        help="FlowVN distillation teacher cascade count",
    )
    parser.add_argument(
        "--teacher_initialize_student",
        "--teacher-initialize-student",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="initialize student stages from uniformly paired teacher stages",
    )
    parser.add_argument(
        "--distill_gt_weight",
        "--distill-gt-weight",
        type=float,
        default=1.0,
        help="FlowVN distillation GT supervised loss weight",
    )
    parser.add_argument(
        "--distill_teacher_weight",
        "--distill-teacher-weight",
        type=float,
        default=0.5,
        help="FlowVN distillation teacher image-output matching loss weight",
    )
    parser.add_argument(
        "--distill_kspace_weight",
        "--distill-kspace-weight",
        type=float,
        default=0.1,
        help="FlowVN distillation k-space consistency loss weight",
    )
    parser.add_argument(
        "--distill_trajectory_weight",
        "--distill-trajectory-weight",
        type=float,
        default=0.0,
        help="teacher-aligned intermediate reconstruction loss weight",
    )
    parser.add_argument(
        "--distill_update_weight",
        "--distill-update-weight",
        type=float,
        default=0.0,
        help="teacher-aligned stage-update loss weight",
    )
    parser.add_argument(
        "--distill_kspace_region",
        "--distill-kspace-region",
        choices=("acquired", "unacquired", "all"),
        default="acquired",
        help="k-space region used for teacher consistency",
    )
    parser.add_argument(
        "--distill_normalize_teacher_losses",
        "--distill-normalize-teacher-losses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="normalize teacher-derived image losses by target magnitude",
    )
    parser.add_argument(
        "--distill_normalization_epsilon",
        "--distill-normalization-epsilon",
        type=float,
        default=1e-6,
        help="minimum denominator for normalized teacher losses",
    )
    parser.add_argument(
        "--distill_teacher_schedule",
        "--distill-teacher-schedule",
        choices=("constant", "warmup_cosine"),
        default="constant",
        help="epoch schedule applied to all teacher-derived loss weights",
    )
    parser.add_argument(
        "--distill_teacher_warmup_epochs",
        "--distill-teacher-warmup-epochs",
        type=int,
        default=0,
        help="linear warmup epochs for teacher-derived loss weights",
    )
    parser.add_argument(
        "--distill_teacher_active_epochs",
        "--distill-teacher-active-epochs",
        type=int,
        default=0,
        help=(
            "initial epochs with teacher losses; <=0 keeps distillation active "
            "for the full run"
        ),
    )
    parser.add_argument(
        "--distill_teacher_final_scale",
        "--distill-teacher-final-scale",
        type=float,
        default=0.1,
        help="final multiplier for warmup-cosine teacher-derived losses",
    )
    parser.add_argument(
        "--log_images_every_n_steps",
        type=int,
        default=500,
        help="log training image panels every N steps; <=0 disables",
    )
    parser.add_argument(
        "--train_metric_interval",
        type=int,
        default=1,
        help="train mode: compute expensive challenge metrics every N training steps; <=0 disables",
    )
    parser.add_argument(
        "--param_grad_log_every_n_steps",
        type=int,
        default=50,
        help="train mode: log parameter/gradient stats every N steps; <=0 disables callback",
    )
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="flowvn-distillation", help="wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="wandb entity/team"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="wandb run name"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode",
    )
    parser.add_argument(
        "--test_log_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="test mode: log a few input/recon images",
    )
    parser.add_argument(
        "--test_log_max_images",
        type=int,
        default=3,
        help="test mode: max images to log",
    )
    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FlowVN only: compile model with torch.compile",
    )
    parser.add_argument(
        "--compile_flowvn_activations",
        "--compile-flowvn-activations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FlowVN only: compile activation modules with torch.compile while leaving complex MRI ops eager",
    )
    parser.add_argument(
        "--compile_flowvn_regularizer",
        "--compile-flowvn-regularizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FlowVN only: compile each real-valued regularizer branch wrapper with torch.compile while leaving complex MRI ops eager",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        help="torch.compile mode when --compile_model is enabled",
    )
    parser.add_argument(
        "--torchdynamo_disable_lru_cache",
        "--torchdynamo-disable-lru-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable TorchDynamo's graph LRU cache for variable-shape checkpoint recomputation",
    )
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable TF32 matmul/convolution on supported NVIDIA GPUs",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable cuDNN autotune benchmarking for stable input shapes",
    )
    parser.add_argument(
        "--deterministic_algorithms",
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="force PyTorch deterministic algorithms where available",
    )
    parser.add_argument(
        "--precision_mode",
        type=str,
        default="fp32",
        choices=["fp32", "amp_fp16", "amp_bf16"],
        help="trainer precision mode",
    )
    parser.add_argument(
        "--inference_empty_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="validation/test mode: clear unused CUDA allocator blocks after each batch",
    )
    parser.add_argument(
        "--test_skip_gt_precompute",
        "--test-skip-gt-precompute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="test mode: skip unused GT image precomputation in the dataset",
    )
    parser.add_argument(
        "--test_cache_case_assets",
        "--test-cache-case-assets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="test mode: cache shared per-case CPU assets while iterating velocity segments",
    )
    parser.add_argument(
        "--test_gpu_preprocess_adjoint",
        "--test-gpu-preprocess-adjoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="test mode: defer FlowVN adjoint preprocessing to the active torch device",
    )
    parser.add_argument(
        "--test_gpu_preprocess_fe_ifft",
        "--test-gpu-preprocess-fe-ifft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="test mode: also defer FlowVN fully sampled FE inverse FFT to the active torch device",
    )
    parser.add_argument(
        "--dataloader_persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep DataLoader workers alive across epochs when num_workers > 0",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=None,
        help="DataLoader prefetch_factor when num_workers > 0; unset keeps PyTorch default",
    )
    parser.add_argument(
        "--flowvn_activation_lowmem",
        "--flowvn-activation-lowmem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FlowVN linear_flowvn activations: loop per filter to reduce memory; use --no-flowvn-activation-lowmem for vectorized activation benchmarking",
    )
    parser.add_argument(
        "--flowvn_transpose_conv_as_conv",
        "--flowvn-transpose-conv-as-conv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FlowVN only: replace regularizer conv_transpose3d with equivalent flipped conv3d for benchmarking",
    )
    parser.add_argument(
        "--flowvn_dc_centered_fft",
        "--flowvn-dc-centered-fft",
        choices=["original", "checkerboard"],
        default="original",
        help="FlowVN only: data-consistency centered FFT implementation; checkerboard is opt-in and not bitwise identical",
    )


    parser.add_argument(
        "--checkpoint_monitor",
        type=str,
        default="val/nRMSE",
        help="metric name to select best checkpoint",
    )
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="min or max for best checkpoint",
    )
    parser.add_argument(
        "--checkpoint_save_top_k",
        type=int,
        default=1,
        help="number of best checkpoints to keep",
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=1,
        help="checkpoint cadence in training epochs",
    )
    parser.add_argument(
        "--checkpoint_save_last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also keep last.ckpt when validation-based checkpointing is enabled",
    )
    parser.add_argument(
        "--checkpoint_save_weights_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save only model weights; disable for resumable workshop training",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker count for training and fallback count for validation",
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=None,
        help="validation DataLoader worker count; unset inherits --num_workers",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "validate", "test"],
        default="train",
        help="train, validate a checkpoint, or run challenge test inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional seed for Python, NumPy, PyTorch, Lightning, and DataLoader workers",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--epoch", type=int, default=100, help="number of training epoch"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--train_sample_limit",
        type=int,
        default=50,
        help="max number of training samples to use; <=0 uses all samples",
    )
    parser.add_argument(
        "--val_sample_limit",
        type=int,
        default=50,
        help="max number of validation encoding samples to use; <=0 uses all",
    )
    parser.add_argument(
        "--val_sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="explicit validation dataset indices, useful for complete four-encoding memory smokes",
    )
    parser.add_argument(
        "--val_case_id",
        type=str,
        default=None,
        help="stable pseudonymous case ID for semantic validation subsetting",
    )
    parser.add_argument(
        "--val_usrate_filter",
        type=int,
        default=None,
        help="acceleration factor for semantic validation subsetting",
    )
    parser.add_argument(
        "--val_on_the_fly_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="validation: generate ktGaussian masks on-the-fly instead of loading usmask_ktGaussian*.mat",
    )
    parser.add_argument(
        "--val_mask_seed",
        type=int,
        default=12345,
        help="base seed for deterministic on-the-fly validation masks",
    )
    parser.add_argument(
        "--val_loss_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="validation: only compute loss and skip heavy metrics/logging",
    )
    parser.add_argument(
        "--val_metrics_output",
        type=str,
        default=None,
        help="validation per-group CSV path; default is save_dir/validation_metrics.csv",
    )
    parser.add_argument(
        "--fit_with_validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run validation during fit; disable for patch training on GPUs that cannot hold full validation volumes",
    )
    parser.add_argument(
        "--loss", type=str, default="", help="type of loss (ssdu or supervised)"
    )
    parser.add_argument(
        "--usrate",
        type=int,
        nargs="+",
        default=None,
        help="test only: one or more ktGaussian undersampling rates, e.g. --usrate 10 20 30",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ids, e.g. --devices 0 or --devices 0 1 2 3",
    )
    parser.add_argument(
        "--test_roots",
        type=str,
        nargs="*",
        default=None,
        help="test mode: one or more roots to scan for cases (recursive)",
    )
    parser.add_argument(
        "--train_roots",
        type=str,
        nargs="*",
        default=None,
        help="train mode: one or more roots to scan for cases (recursive)",
    )
    parser.add_argument(
        "--train_split_json",
        type=str,
        default=None,
        help="train mode: JSON list of relative case paths (requires --split_base_dir)",
    )
    parser.add_argument(
        "--val_roots",
        type=str,
        nargs="*",
        default=None,
        help="val mode: one or more roots to scan for cases (recursive)",
    )
    parser.add_argument(
        "--val_split_json",
        type=str,
        default=None,
        help="val mode: JSON list of relative case paths (requires --split_base_dir)",
    )
    parser.add_argument(
        "--split_base_dir",
        type=str,
        default=None,
        help="base directory for split JSON relative paths (e.g. .../TrainSet/Aorta)",
    )
    parser.add_argument(
        "--in_base_dir",
        type=str,
        default=None,
        help="base input dir used to compute relative path, e.g. .../ChallengeData/TaskR1&R2",
    )
    parser.add_argument(
        "--out_base_dir",
        type=str,
        default=None,
        help="base output dir used to mirror directory structure, e.g. .../ChallengeData_FlowVN/TaskR1&R2",
    )
    parser.add_argument(
        "--lowmem",
        action="store_true",
        help="test-only: use FlowVNLowMem for inference to reduce memory",
    )
    return parser


def _train(args: dict, parser: argparse.ArgumentParser):
    _apply_torch_runtime_options(args)
    _apply_seed_options(args)

    using_split = bool(args.get("train_split_json")) or bool(args.get("val_split_json"))

    # Explicitly use dataloader DEFAULT_OPTS when CLI roots are not provided.
    if not using_split and args.get("train_roots", None) in (None, [], ()):
        args["train_roots"] = list(DATA_DEFAULT_OPTS.get("train_roots", []))
    if not using_split and args.get("val_roots", None) in (None, [], ()):
        args["val_roots"] = list(DATA_DEFAULT_OPTS.get("val_roots", []))

    print(f"[INFO] train_roots resolved to: {args.get('train_roots', [])}")
    print(f"[INFO] val_roots resolved to: {args.get('val_roots', [])}")
    if using_split:
        print(f"[INFO] train_split_json: {args.get('train_split_json', None)}")
        print(f"[INFO] val_split_json: {args.get('val_split_json', None)}")
        print(f"[INFO] split_base_dir: {args.get('split_base_dir', None)}")

    save_dir = Path(args["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    loggers = [TensorBoardLogger("outputs/lightning_logs", name="")]

    if args.get("use_wandb", False):
        if wandb is None:
            raise ImportError(
                "WandB is enabled but not installed. Run: uv sync --extra wandb"
            )

        # Ensure WandB is not force-disabled by environment.
        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
            os.environ.pop("WANDB_DISABLED", None)

        mode = args.get("wandb_mode", "online")
        if mode in ("offline", "disabled"):
            os.environ["WANDB_MODE"] = mode
        else:
            os.environ.pop("WANDB_MODE", None)

        wb_logger = WandbLogger(
            project=args.get("wandb_project", "flowvn-distillation"),
            entity=args.get("wandb_entity", None),
            name=args.get("wandb_run_name", None),
            save_dir=str(save_dir),
            log_model=False,
        )
        wb_logger.log_hyperparams(args)
        loggers.append(wb_logger)

        # Force run initialization early so failures are immediate and visible.
        _ = wb_logger.experiment
        run = wandb.run
        if run is not None:
            print(f"[INFO] WandB run initialized: {run.name} ({run.id})")
            if getattr(run, "url", None):
                print(f"[INFO] WandB URL: {run.url}")

    configured_run_name = args.get("wandb_run_name")
    if configured_run_name:
        run_prefix = str(configured_run_name)
    else:
        run_prefix = next_numeric_run_id(Path("outputs/lightning_logs").glob("*"))
    fit_with_validation = bool(args.get("fit_with_validation", True))
    checkpoint_every_n_epochs = max(
        int(args.get("checkpoint_every_n_epochs", 1)), 1
    )
    if fit_with_validation:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename=run_prefix + "-epoch{epoch:03d}",
            monitor=str(args.get("checkpoint_monitor", "val/nRMSE")),
            mode=str(args.get("checkpoint_mode", "min")),
            save_top_k=int(args.get("checkpoint_save_top_k", 1)),
            every_n_epochs=checkpoint_every_n_epochs,
            save_last=bool(args.get("checkpoint_save_last", False)),
            save_weights_only=bool(args.get("checkpoint_save_weights_only", True)),
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename=run_prefix + "-epoch{epoch:03d}",
            monitor=None,
            save_top_k=-1,
            every_n_epochs=checkpoint_every_n_epochs,
            save_last=True,
            save_weights_only=bool(args.get("checkpoint_save_weights_only", True)),
        )

    callbacks = [checkpoint_callback]
    callbacks.append(RuntimeStatsCallback())
    param_grad_log_every_n_steps = int(args.get("param_grad_log_every_n_steps", 50))
    if param_grad_log_every_n_steps > 0:
        paramgrad_cb = ParamGradTensorBoardCallback(
            log_every_n_steps=param_grad_log_every_n_steps,
            log_hist=False,
            log_stats=True,
            max_params=None,
        )
        callbacks.append(paramgrad_cb)

    dataset = CMRx4DFlowDataSet(**args)
    train_sample_limit = int(args.get("train_sample_limit", 0) or 0)
    if args.get("devices", [0]) and args["devices"][0] == 0:
        print(
            f"[INFO] requested limits: train_sample_limit={train_sample_limit}, val_sample_limit={int(args.get('val_sample_limit', 0) or 0)}"
        )
    if train_sample_limit > 0 and train_sample_limit < len(dataset):
        dataset = Subset(dataset, list(range(train_sample_limit)))
        if args.get("devices", [0]) and args["devices"][0] == 0:
            print(
                f"[INFO] train_sample_limit enabled: using {train_sample_limit} samples"
            )
    num_workers = int(args.get("num_workers", 2))
    val_num_workers_option = args.get("val_num_workers", None)
    if val_num_workers_option is None:
        val_num_workers = num_workers
    else:
        val_num_workers = int(val_num_workers_option)
    batch_size = int(args.get("batch_size", 1))
    dataloader = DataLoader(
        dataset, **_dataloader_kwargs(args, batch_size, num_workers, shuffle=True)
    )

    val_dataloader = None
    if fit_with_validation:
        val_sample_limit = int(args.get("val_sample_limit", 0) or 0)
        same_train_val_sanity = (
            train_sample_limit > 0
            and val_sample_limit > 0
            and train_sample_limit == val_sample_limit
            and args.get("val_sample_indices", None) in (None, [], ())
        )
        if same_train_val_sanity:
            val_dataset = dataset
            if args.get("devices", [0]) and args["devices"][0] == 0:
                print(
                    f"[SANITY] train and validation are using the same dataset subset (limit={train_sample_limit})."
                )
        else:
            args_val = args.copy()
            args_val["mode"] = "val"
            if not using_split and args_val.get("val_roots", None) in (None, [], ()):
                args_val["val_roots"] = list(DATA_DEFAULT_OPTS.get("val_roots", []))
            val_dataset = CMRx4DFlowDataSet(**args_val)
            val_dataset = _select_validation_subset(val_dataset, args)
        if len(val_dataset) == 0:
            raise RuntimeError(
                "Validation dataset is empty. "
                f"val_roots={args.get('val_roots', [])}, "
                f"val_split_json={args.get('val_split_json', None)}. "
                "Please set a valid --val_roots or --val_split_json path."
            )
        val_dataloader = DataLoader(
            val_dataset,
            **_dataloader_kwargs(args, batch_size, val_num_workers, shuffle=False),
        )
    else:
        print("[INFO] fit_with_validation=False: skipping validation during fit")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args["devices"],
        strategy="ddp_find_unused_parameters_true"
        if len(args["devices"]) > 1
        else "auto",
        max_epochs=args["epoch"],
        logger=loggers,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        precision=_lightning_precision(args),
    )

    resume_from_checkpoint = args.get("resume_from_checkpoint", None)
    if resume_from_checkpoint in ("", "None"):
        resume_from_checkpoint = None
    if resume_from_checkpoint is not None and args["ckpt_path"] is not None:
        raise ValueError("Use only one of --ckpt_path or --resume_from_checkpoint")

    if args["ckpt_path"] is not None:
        model = UnrolledNetwork.load_from_checkpoint(args["ckpt_path"], **args)
    else:
        model = UnrolledNetwork(**args)

    if resume_from_checkpoint is None:
        if model.teacher_initialization_report is not None:
            teacher_initialization_path = (
                save_dir / "teacher_initialization.json"
            )
            teacher_initialization_path.write_text(
                json.dumps(
                    model.teacher_initialization_report,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                "[PROVENANCE] teacher_initialization="
                f"{teacher_initialization_path}"
            )
            print(
                "[PROVENANCE] teacher_stage_map="
                f"{model.teacher_initialization_report['teacher_stage_map']}"
            )
            for logger in loggers:
                logger.log_hyperparams(
                    {
                        "teacher_initialization_checkpoint_sha256": (
                            model.teacher_initialization_report[
                                "teacher_checkpoint_sha256"
                            ]
                        ),
                        "teacher_initialization_stage_map": (
                            model.teacher_initialization_report[
                                "teacher_stage_map"
                            ]
                        ),
                    }
                )
        student_initialization_sha256 = _state_dict_sha256(model.network)
        fingerprint_path = save_dir / "student_initialization.sha256"
        fingerprint_path.write_text(student_initialization_sha256 + "\n")
        print(
            "[PROVENANCE] student_initialization_sha256="
            f"{student_initialization_sha256}"
        )
        for logger in loggers:
            logger.log_hyperparams(
                {"student_initialization_sha256": student_initialization_sha256}
            )
    else:
        print(
            "[PROVENANCE] student initialization restored by Lightning from "
            f"{resume_from_checkpoint}"
        )
    model = _maybe_compile_flowvn(model, args)

    if args.get("use_wandb", False):
        print(f"[INFO] Active loggers: {[type(lg).__name__ for lg in loggers]}")

    trainer.fit(
        model,
        train_dataloaders=dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_from_checkpoint,
    )


def _validate(args: dict):
    if args.get("ckpt_path", None) in (None, "", "None"):
        raise ValueError("validate mode requires --ckpt_path")

    _apply_torch_runtime_options(args)
    _apply_seed_options(args)

    model_args = args.copy()
    model_args["mode"] = "validate"
    model_args["distill_enabled"] = False
    model_args["val_loss_only"] = False

    dataset_args = args.copy()
    dataset_args["mode"] = "val"
    using_split = bool(dataset_args.get("val_split_json"))
    if not using_split and dataset_args.get("val_roots", None) in (None, [], ()):
        dataset_args["val_roots"] = list(DATA_DEFAULT_OPTS.get("val_roots", []))

    val_dataset = CMRx4DFlowDataSet(**dataset_args)
    val_dataset = _select_validation_subset(val_dataset, args)
    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation dataset is empty. "
            f"val_roots={dataset_args.get('val_roots', [])}, "
            f"val_split_json={dataset_args.get('val_split_json', None)}."
        )

    val_num_workers_option = args.get("val_num_workers", None)
    if val_num_workers_option is None:
        val_num_workers = int(args.get("num_workers", 2))
    else:
        val_num_workers = int(val_num_workers_option)
    batch_size = int(args.get("batch_size", 1))
    val_dataloader = DataLoader(
        val_dataset,
        **_dataloader_kwargs(args, batch_size, val_num_workers, shuffle=False),
    )

    save_dir = Path(args.get("save_dir", "outputs/validation"))
    save_dir.mkdir(parents=True, exist_ok=True)
    loggers = [TensorBoardLogger("outputs/lightning_logs", name="validation")]
    if args.get("use_wandb", False):
        if wandb is None:
            raise ImportError(
                "WandB is enabled but not installed. Run: uv sync --extra wandb"
            )
        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
            os.environ.pop("WANDB_DISABLED", None)
        wandb_mode = args.get("wandb_mode", "online")
        if wandb_mode in ("offline", "disabled"):
            os.environ["WANDB_MODE"] = wandb_mode
        else:
            os.environ.pop("WANDB_MODE", None)
        wb_logger = WandbLogger(
            project=args.get("wandb_project", "flowvn-distillation"),
            entity=args.get("wandb_entity", None),
            name=args.get("wandb_run_name", None),
            save_dir=str(save_dir),
            log_model=False,
        )
        wb_logger.log_hyperparams(model_args)
        loggers.append(wb_logger)
        _ = wb_logger.experiment
        run = wandb.run
        if run is not None:
            print(f"[INFO] WandB validation run initialized: {run.name} ({run.id})")
            if getattr(run, "url", None):
                print(f"[INFO] WandB URL: {run.url}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args["devices"],
        strategy="ddp_find_unused_parameters_true"
        if len(args["devices"]) > 1
        else "auto",
        logger=loggers,
        num_sanity_val_steps=0,
        callbacks=[RuntimeStatsCallback()],
        precision=_lightning_precision(args),
    )

    model = UnrolledNetwork.load_from_checkpoint(
        args["ckpt_path"], map_location="cpu", **model_args
    )
    model.eval()
    model = _maybe_compile_flowvn(model, model_args)
    trainer.validate(model, dataloaders=val_dataloader)


def _test(args: dict):
    _apply_torch_runtime_options(args)
    _apply_seed_options(args)

    if args.get("usrate", None) is None:
        raise ValueError(
            "test mode requires --usrate, e.g. --usrate 10 or --usrate 10 20 30"
        )
    if bool(args.get("test_gpu_preprocess_fe_ifft", False)) and not bool(
        args.get("test_gpu_preprocess_adjoint", False)
    ):
        raise ValueError(
            "--test_gpu_preprocess_fe_ifft requires --test_gpu_preprocess_adjoint"
        )
    if (
        bool(args.get("test_gpu_preprocess_adjoint", False))
        and args.get("network") != "FlowVN"
    ):
        raise ValueError(
            "--test_gpu_preprocess_adjoint is only supported for network=FlowVN"
        )

    if isinstance(args["usrate"], int):
        args["usrate"] = [int(args["usrate"])]
    else:
        args["usrate"] = [int(u) for u in args["usrate"]]

    if args.get("test_roots", None) is not None:
        args["test_roots"] = args["test_roots"]

    if args.get("out_base_dir", None) in (None, "", "None"):
        args["out_base_dir"] = args["save_dir"]

    if args.get("in_base_dir", None) in (None, "", "None"):
        tr = args.get("test_roots", None)
        if tr and len(tr) > 0:
            p = Path(tr[0]).resolve()
            args["in_base_dir"] = (
                str(p.parent) if p.name in ("ValidationSet", "TrainSet") else str(p)
            )

    dataset = CMRx4DFlowDataSet(**args)
    num_workers = int(args.get("num_workers", 1))
    batch_size = int(args.get("batch_size", 1))
    dataloader = DataLoader(
        dataset, **_dataloader_kwargs(args, batch_size, num_workers, shuffle=False)
    )
    save_callback = CMRSaveCallback()
    loggers = []
    if args.get("use_wandb", False):
        if wandb is None:
            raise ImportError(
                "WandB is enabled but not installed. Run: uv sync --extra wandb"
            )

        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
            os.environ.pop("WANDB_DISABLED", None)

        mode = args.get("wandb_mode", "online")
        if mode in ("offline", "disabled"):
            os.environ["WANDB_MODE"] = mode
        else:
            os.environ.pop("WANDB_MODE", None)

        wb_logger = WandbLogger(
            project=args.get("wandb_project", "flowvn-distillation"),
            entity=args.get("wandb_entity", None),
            name=args.get("wandb_run_name", None),
            save_dir=str(args.get("save_dir", "outputs/exp")),
            log_model=False,
        )
        wb_logger.log_hyperparams(args)
        loggers.append(wb_logger)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args["devices"][0]]
        if isinstance(args["devices"], (list, tuple))
        else [args["devices"]],
        logger=loggers if len(loggers) > 0 else False,
        callbacks=[save_callback],
        precision=_lightning_precision(args),
    )

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = "cpu"

    model = UnrolledNetwork.load_from_checkpoint(
        args["ckpt_path"],
        map_location="cpu",  # Load on the CPU before Lightning moves the model.
        **args,
    )
    model.eval()
    model = _maybe_compile_flowvn(model, args)

    trainer.test(model, dataloaders=dataloader)


def main():
    parser = _build_arg_parser()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config:
        if yaml is None:
            raise ImportError(
                "PyYAML is required for --config. Run: uv sync"
            )
        with open(pre_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(
                "YAML config must be a mapping of argument names to values"
            )
        parser.set_defaults(**cfg)

    args_ns = parser.parse_args()
    args = vars(args_ns)
    from utils.flowvn_mask_backend import require_mask_backend

    try:
        require_mask_backend(args)
    except ValueError as error:
        parser.error(str(error))
    print_options(parser, args_ns)

    if args["mode"] == "train":
        _train(args, parser)
    elif args["mode"] == "validate":
        _validate(args)
    elif args["mode"] == "test":
        _test(args)
    else:
        raise ValueError("mode must be 'train', 'validate', or 'test'")


if __name__ == "__main__":
    main()
