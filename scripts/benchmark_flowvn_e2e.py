#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowVN challenge-style end-to-end inference benchmark")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "inference.yaml"))
    parser.add_argument("--case-dir", type=str, required=True, help="real case directory or dataset root")
    parser.add_argument("--real-mode", choices=["test", "val", "train"], default="test")
    parser.add_argument("--real-index-start", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0, help="max dataset items to run; <=0 runs all")
    parser.add_argument(
        "--repeat-index-cycles",
        type=int,
        default=1,
        help="benchmark-only: repeat the selected dataset index list this many times in one Python process",
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--usrate", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out-base-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-algorithms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-flowvn-activations", action="store_true")
    parser.add_argument("--compile-flowvn-regularizer", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument(
        "--cudagraph-mark-step-begin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="benchmark-only: call torch.compiler.cudagraph_mark_step_begin() before each model forward",
    )
    parser.add_argument(
        "--cuda-graph-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="benchmark-only: capture fixed-shape FlowVN forward with torch.cuda.CUDAGraph and replay it for subsequent items",
    )
    parser.add_argument(
        "--cuda-graph-warmup-iters",
        type=int,
        default=1,
        help="benchmark-only warmup forwards on a side stream before each CUDA Graph capture",
    )
    parser.add_argument(
        "--cuda-graph-verify-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="benchmark-only: compare each CUDA Graph replay output against an eager forward; disables clean timing",
    )
    parser.add_argument("--flowvn-activation-lowmem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flowvn-transpose-conv-as-conv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flowvn-dc-centered-fft", choices=["original", "checkerboard"], default="original")
    parser.add_argument("--test-skip-gt-precompute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-cache-case-assets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-gpu-preprocess-adjoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-gpu-preprocess-fe-ifft", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-gpu-preprocess-inputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-preprocess", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-next-item", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loader-mode", choices=["direct", "dataloader"], default="direct")
    parser.add_argument("--num-workers", type=int, default=0, help="benchmark-only DataLoader worker count")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="benchmark-only DataLoader prefetch_factor when num_workers > 0")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False, help="benchmark-only DataLoader pin_memory")
    parser.add_argument("--segment-batch-size", type=int, default=1, help="benchmark-only number of velocity-segment items per model forward")
    parser.add_argument("--precision-mode", choices=["fp32", "amp_fp16", "amp_bf16"], default="fp32")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--time-frames", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--features-out", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def synchronize(torch_module, device) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def cuda_peak_memory(torch_module, device) -> int:
    return int(torch_module.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


class IndexedDataset:
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> dict[str, Any]:
        dataset_index = int(self.indices[pos])
        item = dict(self.dataset[dataset_index])
        item["_dataset_index"] = dataset_index
        return item


def scalar_int(torch_module, value: Any) -> int:
    if torch_module.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return scalar_int(torch_module, value[0])
    return int(value)


def scalar_bool(torch_module, value: Any) -> bool:
    if torch_module.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return scalar_bool(torch_module, value[0])
    return bool(value)


def scalar_str(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return scalar_str(value[0])
    return str(value)


def meta_sequence(value: Any, batch_size: int) -> str | list[str]:
    if batch_size <= 1:
        return scalar_str(value)
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return [scalar_str(v) for v in value]
        if len(value) == 1:
            return [scalar_str(value[0]) for _ in range(batch_size)]
    return [str(value) for _ in range(batch_size)]


def meta_at(value: Any, index: int) -> str:
    if isinstance(value, (list, tuple)):
        return scalar_str(value[index])
    return scalar_str(value)


def batched_tensor(torch_module, value: Any, raw_ndim: int, device=None, dtype=None):
    tensor = torch_module.as_tensor(value, dtype=dtype) if dtype is not None else torch_module.as_tensor(value)
    if tensor.ndim == raw_ndim:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != raw_ndim + 1:
        raise RuntimeError(f"Expected tensor ndim {raw_ndim} or {raw_ndim + 1}, got {tensor.ndim}")
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    return tensor


def json_safe(torch_module, value: Any):
    if torch_module.is_tensor(value):
        if value.numel() == 1:
            return value.reshape(-1)[0].item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(torch_module, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(torch_module, v) for v in value]
    return value


def optional_float(value: Any):
    return None if value is None else float(value)


def sum_optional(items: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(item[key]) for item in items if item.get(key) is not None))


def tensor_to_device(torch_module, item: dict[str, Any], device) -> dict[str, Any]:
    deferred = "kdata_p1_unnorm" in item
    seg_idx = torch_module.as_tensor(item["seg_idx"], dtype=torch_module.int64).view(-1)
    usrate = torch_module.as_tensor(item["usrate"], dtype=torch_module.int64).view(-1)
    batch_size = int(seg_idx.numel())
    batch = {
        "coil_sens": batched_tensor(torch_module, item["coil_sens"], raw_ndim=4, device=device),
        "usrate_true": torch_module.as_tensor(item["usrate_true"], dtype=torch_module.float32).view(-1).to(device, non_blocking=True),
        "segmentation": batched_tensor(torch_module, item["segmentation"], raw_ndim=3),
        "usrate": usrate,
        "seg_idx": seg_idx,
        "case_dir": meta_sequence(item["case_dir"], batch_size),
        "out_dir": meta_sequence(item["out_dir"], batch_size),
        "subj": meta_sequence(item["subj"], batch_size),
    }
    if "slice_start" in item:
        batch["slice_start"] = torch_module.as_tensor(item["slice_start"], dtype=torch_module.int64).view(-1)
    if "gt" in item:
        batch["gt"] = batched_tensor(torch_module, item["gt"], raw_ndim=5, device=device)
    if "VENC" in item:
        batch["VENC"] = torch_module.as_tensor(item["VENC"], dtype=torch_module.float32).view(-1)
    if deferred:
        batch["kdata_p1_unnorm"] = batched_tensor(torch_module, item["kdata_p1_unnorm"], raw_ndim=6, device=device)
        batch["mask"] = batched_tensor(torch_module, item["mask"], raw_ndim=6, device=device)
        batch["deferred_gpu_preprocess_fe_ifft"] = scalar_bool(
            torch_module,
            item.get("deferred_gpu_preprocess_fe_ifft", False),
        )
    else:
        batch["imdata_p1"] = batched_tensor(torch_module, item["imdata_p1"], raw_ndim=5, device=device)
        batch["kdata_p1"] = batched_tensor(torch_module, item["kdata_p1"], raw_ndim=6, device=device)
        batch["norm"] = torch_module.as_tensor(item["norm"], dtype=torch_module.float32).view(-1).to(device, non_blocking=True)
    return batch


def metrics_predgt_array(torch_module, value: Any):
    if torch_module.is_tensor(value):
        value = value.detach().cpu().numpy()
    if value.ndim != 5:
        raise RuntimeError(f"Expected metrics tensor (Nv,Nt,FE,PE,SPE), got {tuple(value.shape)}")
    return value.transpose(0, 1, 4, 3, 2)


def metrics_single_nv_array(torch_module, value: Any):
    if torch_module.is_tensor(value):
        value = value.detach().cpu().numpy()
    if value.ndim == 6 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4:
        raise RuntimeError(f"Expected single encoding tensor (Nt,FE,PE,SPE), got {tuple(value.shape)}")
    return value.transpose(0, 3, 2, 1)


def metrics_seg_array(torch_module, value: Any):
    if torch_module.is_tensor(value):
        value = value.detach().cpu().numpy()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise RuntimeError(f"Expected segmentation tensor (FE,PE,SPE), got {tuple(value.shape)}")
    return value.astype(bool).transpose(2, 1, 0)


def maybe_apply_phase_correction(pred_np, gt_np):
    from utils.utils_bgc import execute_MSAC

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
        pass
    return pred_np, gt_np


class ValMetricAccumulator:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.groups: dict[tuple[str, int, int], dict[str, Any]] = {}

    def update(self, batch: dict[str, Any], recon_cpu):
        if "gt" not in batch:
            return
        case_dir = scalar_str(batch["case_dir"])
        slice_start = scalar_int(self.torch, batch.get("slice_start", 0))
        usrate = scalar_int(self.torch, batch["usrate"])
        seg_idx = scalar_int(self.torch, batch["seg_idx"])
        key = (case_dir, slice_start, usrate)

        norm_cpu = batch["norm"].detach().cpu().view(-1)[0]
        gt_eval = (batch["gt"][0].detach().cpu() * norm_cpu)
        pred_nv = metrics_single_nv_array(self.torch, recon_cpu)
        gt_nv = metrics_single_nv_array(self.torch, gt_eval)
        seg_np = metrics_seg_array(self.torch, batch["segmentation"])

        if key not in self.groups:
            self.groups[key] = {"pred": {}, "gt": {}, "seg": seg_np}
        self.groups[key]["pred"][seg_idx] = pred_nv
        self.groups[key]["gt"][seg_idx] = gt_nv

    def compute(self) -> dict[str, Any]:
        from utils.utils_flow import complex2magflow
        from utils.utils_metrics import AngErr, RelErr, SSIM, nRMSE

        expected_segs = (0, 1, 2, 3)
        nrmse_vals, ssim_vals, relerr_vals, angerr_vals = [], [], [], []
        groups_out = []
        n_complete = 0

        for key, pack in self.groups.items():
            complete = all(i in pack["pred"] and i in pack["gt"] for i in expected_segs)
            group_record = {
                "case_dir": str(key[0]),
                "slice_start": int(key[1]),
                "usrate": int(key[2]),
                "complete": bool(complete),
            }
            if not complete:
                groups_out.append(group_record)
                continue

            n_complete += 1
            pred_np = np.stack([pack["pred"][i] for i in expected_segs], axis=0)
            gt_np = np.stack([pack["gt"][i] for i in expected_segs], axis=0)
            seg_np = pack["seg"]
            if np.sum(seg_np.astype(np.int64)) == 0:
                seg_np = np.ones_like(seg_np, dtype=bool)

            pred_np, gt_np = maybe_apply_phase_correction(pred_np, gt_np)
            mag_pred, flow_pred = complex2magflow(pred_np)
            mag_gt, flow_gt = complex2magflow(gt_np)

            vals = {"nrmse": None, "ssim": None, "relerr": None, "angerr": None}
            try:
                val = float(nRMSE(mag_pred, mag_gt, seg_np))
                if np.isfinite(val):
                    vals["nrmse"] = val
                    nrmse_vals.append(val)
            except Exception:
                pass
            try:
                val = float(SSIM(mag_pred, mag_gt, seg_np))
                if np.isfinite(val):
                    vals["ssim"] = val
                    ssim_vals.append(val)
            except Exception:
                pass
            if flow_pred.shape[0] > 0 and flow_gt.shape[0] > 0:
                try:
                    val = float(RelErr(flow_pred, flow_gt, seg_np))
                    if np.isfinite(val):
                        vals["relerr"] = val
                        relerr_vals.append(val)
                except Exception:
                    pass
                try:
                    val = float(AngErr(flow_pred, flow_gt, seg_np))
                    if np.isfinite(val):
                        vals["angerr"] = val
                        angerr_vals.append(val)
                except Exception:
                    pass
            group_record.update(vals)
            groups_out.append(group_record)

        return {
            "n_groups": int(len(self.groups)),
            "n_complete": int(n_complete),
            "nrmse": float(np.mean(nrmse_vals)) if nrmse_vals else None,
            "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
            "relerr": float(np.mean(relerr_vals)) if relerr_vals else None,
            "angerr": float(np.mean(angerr_vals)) if angerr_vals else None,
            "groups": groups_out,
        }


def finalize_deferred_preprocess(torch_module, batch: dict[str, Any], device):
    if "kdata_p1_unnorm" not in batch:
        return 0.0, None
    from utils.flowvn_preprocess import finalize_flowvn_deferred_adjoint

    use_cuda = device.type == "cuda"
    if use_cuda:
        start_event = torch_module.cuda.Event(enable_timing=True)
        end_event = torch_module.cuda.Event(enable_timing=True)
        torch_module.cuda.synchronize(device)
        start_event.record()
    wall_start = time.perf_counter()
    with torch_module.inference_mode():
        imdata_p1, kdata_p1, norm = finalize_flowvn_deferred_adjoint(
            batch["kdata_p1_unnorm"],
            batch["coil_sens"],
            batch["mask"],
            fe_ifft=bool(batch.get("deferred_gpu_preprocess_fe_ifft", False)),
        )
    if use_cuda:
        end_event.record()
        torch_module.cuda.synchronize(device)
        cuda_ms = float(start_event.elapsed_time(end_event))
    else:
        cuda_ms = None
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    batch["imdata_p1"] = imdata_p1
    batch["kdata_p1"] = kdata_p1
    batch["norm"] = norm
    if "gt" in batch:
        batch["gt"] = batch["gt"] / norm.view(norm.shape[0], 1, 1, 1, 1, 1)
    del batch["kdata_p1_unnorm"]
    del batch["mask"]
    return float(wall_ms), cuda_ms


def tensor_diff(torch_module, reference, candidate):
    ref = reference.detach()
    cand = candidate.detach()
    diff = torch_module.abs(cand - ref)
    ref_norm = torch_module.linalg.vector_norm(ref.reshape(-1))
    diff_norm = torch_module.linalg.vector_norm((cand - ref).reshape(-1))
    denom = float(ref_norm.item()) if float(ref_norm.item()) != 0.0 else 1.0
    return {
        "max_abs": float(torch_module.max(diff).item()) if diff.numel() else 0.0,
        "mean_abs": float(torch_module.mean(diff).item()) if diff.numel() else 0.0,
        "rel_l2": float(diff_norm.item()) / denom,
    }


def compare_gpu_preprocess_inputs(torch_module, dataset, options, indices, device):
    if not indices:
        return None
    from utils.dataloader_CMRx4DFlow import CMRx4DFlowDataSet

    cpu_options = dict(options)
    cpu_options["test_gpu_preprocess_adjoint"] = False
    cpu_options["test_gpu_preprocess_fe_ifft"] = False
    cpu_dataset = CMRx4DFlowDataSet(**cpu_options)
    item_idx = int(indices[0])
    cpu_batch = tensor_to_device(torch_module, cpu_dataset[item_idx], device)
    gpu_batch = tensor_to_device(torch_module, dataset[item_idx], device)
    gpu_wall_ms, gpu_cuda_ms = finalize_deferred_preprocess(torch_module, gpu_batch, device)
    synchronize(torch_module, device)
    return {
        "dataset_index": item_idx,
        "gpu_preprocess_wall_ms": float(gpu_wall_ms),
        "gpu_preprocess_cuda_ms": gpu_cuda_ms,
        "imdata_p1": tensor_diff(torch_module, cpu_batch["imdata_p1"], gpu_batch["imdata_p1"]),
        "kdata_p1": tensor_diff(torch_module, cpu_batch["kdata_p1"], gpu_batch["kdata_p1"]),
        "norm": tensor_diff(torch_module, cpu_batch["norm"], gpu_batch["norm"]),
    }


def maybe_cudagraph_mark_step_begin(torch_module, enabled: bool) -> None:
    if not enabled:
        return
    compiler = getattr(torch_module, "compiler", None)
    marker = None if compiler is None else getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is None:
        raise RuntimeError("This PyTorch build does not provide torch.compiler.cudagraph_mark_step_begin")
    marker()


class CudaGraphForwardRunner:
    def __init__(
        self,
        torch_module,
        model,
        precision_mode: str,
        device,
        warmup_iters: int = 1,
        verify_eager: bool = False,
    ):
        if device.type != "cuda":
            raise ValueError("--cuda-graph-replay requires a CUDA device")
        if getattr(torch_module.cuda, "CUDAGraph", None) is None:
            raise RuntimeError("This PyTorch build does not provide torch.cuda.CUDAGraph")
        self.torch = torch_module
        self.model = model
        self.precision_mode = precision_mode
        self.device = device
        self.warmup_iters = max(0, int(warmup_iters))
        self.verify_eager = bool(verify_eager)
        self.graph = None
        self.static_batch: dict[str, Any] | None = None
        self.static_output = None
        self.signature = None
        self.capture_count = 0
        self.total_capture_wall_ms = 0.0

    def _signature(self, batch: dict[str, Any]) -> tuple:
        parts = []
        for key in ("imdata_p1", "kdata_p1", "coil_sens", "usrate_true"):
            tensor = batch[key]
            parts.append((key, tuple(tensor.shape), str(tensor.dtype), str(tensor.device)))
        parts.append(("precision_mode", self.precision_mode))
        return tuple(parts)

    def _make_static_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: batch[key].detach().clone()
            for key in ("imdata_p1", "kdata_p1", "coil_sens", "usrate_true")
        }

    def _copy_inputs(self, batch: dict[str, Any]) -> None:
        assert self.static_batch is not None
        for key, static_tensor in self.static_batch.items():
            static_tensor.copy_(batch[key], non_blocking=True)

    def ensure_captured(self, batch: dict[str, Any]) -> dict[str, Any]:
        signature = self._signature(batch)
        if self.graph is not None and self.signature == signature:
            return {
                "cuda_graph_recaptured": False,
                "cuda_graph_capture_wall_ms": 0.0,
                "cuda_graph_capture_count": int(self.capture_count),
            }

        capture_start = time.perf_counter()
        self.static_batch = self._make_static_batch(batch)
        self.torch.cuda.synchronize(self.device)

        current_stream = self.torch.cuda.current_stream(self.device)
        side_stream = self.torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(current_stream)
        with self.torch.cuda.stream(side_stream):
            with self.torch.inference_mode():
                for _ in range(self.warmup_iters):
                    self.static_output = run_network(
                        self.torch,
                        self.model,
                        self.static_batch,
                        self.precision_mode,
                    )
        current_stream.wait_stream(side_stream)
        self.torch.cuda.synchronize(self.device)

        graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(graph):
            self.static_output = run_network(
                self.torch,
                self.model,
                self.static_batch,
                self.precision_mode,
            )
        self.torch.cuda.synchronize(self.device)

        self.graph = graph
        self.signature = signature
        self.capture_count += 1
        capture_wall_ms = (time.perf_counter() - capture_start) * 1000.0
        self.total_capture_wall_ms += capture_wall_ms
        return {
            "cuda_graph_recaptured": True,
            "cuda_graph_capture_wall_ms": float(capture_wall_ms),
            "cuda_graph_capture_count": int(self.capture_count),
        }

    def benchmark(self, batch: dict[str, Any]):
        capture_stats = self.ensure_captured(batch)
        assert self.graph is not None
        assert self.static_output is not None

        start_event = self.torch.cuda.Event(enable_timing=True)
        copy_done_event = self.torch.cuda.Event(enable_timing=True)
        end_event = self.torch.cuda.Event(enable_timing=True)

        self.torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event.record()
        self._copy_inputs(batch)
        copy_done_event.record()
        self.graph.replay()
        end_event.record()
        self.torch.cuda.synchronize(self.device)
        forward_wall_ms = (time.perf_counter() - wall_start) * 1000.0

        graph_diff = None
        if self.verify_eager:
            eager = run_network(
                self.torch,
                self.model,
                batch,
                self.precision_mode,
            )
            self.torch.cuda.synchronize(self.device)
            graph_diff = tensor_diff(self.torch, eager, self.static_output)
            del eager

        forward_cuda_ms = float(start_event.elapsed_time(end_event))
        graph_stats = dict(capture_stats)
        graph_stats.update(
            {
                "cuda_graph_copy_cuda_ms": float(start_event.elapsed_time(copy_done_event)),
                "cuda_graph_replay_cuda_ms": float(copy_done_event.elapsed_time(end_event)),
                "cuda_graph_forward_cuda_ms": forward_cuda_ms,
                "cuda_graph_eager_diff": graph_diff,
            }
        )
        return self.static_output, float(forward_wall_ms), forward_cuda_ms, graph_stats


def run_network(
    torch_module,
    model,
    batch: dict[str, Any],
    precision_mode: str,
    cudagraph_mark_step_begin: bool = False,
):
    enabled = precision_mode != "fp32" and batch["imdata_p1"].device.type == "cuda"
    dtype = torch_module.float16 if precision_mode == "amp_fp16" else torch_module.bfloat16
    with torch_module.inference_mode(), torch_module.autocast("cuda", dtype=dtype, enabled=enabled):
        maybe_cudagraph_mark_step_begin(torch_module, cudagraph_mark_step_begin)
        return model(
            batch["imdata_p1"],
            batch["kdata_p1"],
            batch["coil_sens"],
            batch["usrate_true"],
        )


def benchmark_forward(
    torch_module,
    model,
    batch: dict[str, Any],
    precision_mode: str,
    cudagraph_mark_step_begin: bool = False,
    cuda_graph_runner: CudaGraphForwardRunner | None = None,
):
    if cuda_graph_runner is not None:
        return cuda_graph_runner.benchmark(batch)
    device = batch["imdata_p1"].device
    use_cuda = device.type == "cuda"
    if use_cuda:
        start_event = torch_module.cuda.Event(enable_timing=True)
        end_event = torch_module.cuda.Event(enable_timing=True)
        torch_module.cuda.synchronize(device)
        start_event.record()
    wall_start = time.perf_counter()
    recon = run_network(
        torch_module,
        model,
        batch,
        precision_mode,
        cudagraph_mark_step_begin=cudagraph_mark_step_begin,
    )
    if use_cuda:
        end_event.record()
        torch_module.cuda.synchronize(device)
        forward_cuda_ms = float(start_event.elapsed_time(end_event))
    else:
        forward_cuda_ms = None
    forward_wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return recon, forward_wall_ms, forward_cuda_ms, None


def slice_batch_for_item(torch_module, batch: dict[str, Any], index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = {"segmentation", "norm", "gt", "VENC", "usrate", "seg_idx", "slice_start"}
    for key, value in batch.items():
        if key in tensor_keys and torch_module.is_tensor(value):
            out[key] = value[index : index + 1]
        elif key in ("case_dir", "out_dir", "subj"):
            out[key] = meta_at(value, index)
        else:
            out[key] = value
    return out


def make_callback_outputs(torch_module, batch: dict[str, Any], recon, forward_cuda_ms):
    transfer_start = time.perf_counter()
    batch_size = int(recon.shape[0])
    norm = batch["norm"].to(recon.device, non_blocking=True).view(batch_size, 1, 1, 1, 1, 1)
    recon_cpu = (recon * norm).detach().cpu()
    del recon
    synchronize(torch_module, norm.device)
    cpu_transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    per_segment_ms = None if forward_cuda_ms is None else float(forward_cuda_ms) / max(batch_size, 1)
    outputs = []
    for index in range(batch_size):
        outputs.append(
            {
                "recon": recon_cpu[index],
                "subj": meta_at(batch["subj"], index),
                "case_dir": meta_at(batch["case_dir"], index),
                "out_dir": meta_at(batch["out_dir"], index),
                "seg_idx": int(batch["seg_idx"][index].item()),
                "usrate": int(batch["usrate"][index].item()),
                "recon_ms": per_segment_ms,
            }
        )

    return {
        "outputs": outputs,
        "recon_cpu": recon_cpu,
        "cpu_transfer_ms": float(cpu_transfer_ms),
    }


def list_output_files(out_base_dir: Path) -> list[str]:
    if not out_base_dir.exists():
        return []
    return [str(path) for path in sorted(out_base_dir.rglob("*")) if path.is_file()]


def main() -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from benchmark_flowvn import (
        apply_runtime_options,
        compile_flowvn_activation_modules,
        compile_flowvn_regularizer_modules,
        load_checkpoint,
        load_options,
    )
    from main import CMRSaveCallback
    from networks.flowvn import FlowVN
    from utils.dataloader_CMRx4DFlow import CMRx4DFlowDataSet

    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if args.loader_mode == "dataloader" and bool(args.prefetch_next_item):
        raise ValueError("--prefetch-next-item is only valid with --loader-mode direct")
    if int(args.segment_batch_size) < 1:
        raise ValueError("--segment-batch-size must be >= 1")
    if int(args.segment_batch_size) != 1 and args.loader_mode != "dataloader":
        raise ValueError("--segment-batch-size > 1 is only supported with --loader-mode dataloader")
    if bool(args.test_gpu_preprocess_fe_ifft) and not bool(args.test_gpu_preprocess_adjoint):
        raise ValueError("--test-gpu-preprocess-fe-ifft requires --test-gpu-preprocess-adjoint")
    if bool(args.cuda_graph_replay) and bool(args.cudagraph_mark_step_begin):
        raise ValueError("--cuda-graph-replay and --cudagraph-mark-step-begin are separate benchmark modes")
    if bool(args.cuda_graph_replay) and not args.device.startswith("cuda"):
        raise ValueError("--cuda-graph-replay requires a CUDA device")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_name = args.run_name or f"flowvn_e2e_{int(time.time())}"
    out_base_dir = Path(args.out_base_dir or (ROOT / "outputs" / "e2e" / run_name))
    out_base_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    apply_runtime_options(
        bool(args.allow_tf32),
        deterministic_algorithms=bool(args.deterministic_algorithms),
        cudnn_benchmark=bool(args.cudnn_benchmark),
    )

    options_start = time.perf_counter()
    options = load_options(args.config, args)
    from utils.flowvn_mask_backend import require_mask_backend

    require_mask_backend(options)
    options["mode"] = str(args.real_mode)
    options["network"] = "FlowVN"
    options["loss"] = "supervised"
    options["exp_loss"] = False
    options["lowmem"] = False
    options["input"] = str(args.case_dir)
    options["usrate"] = [int(args.usrate)]
    options["test_roots"] = None
    options["train_roots"] = []
    options["val_roots"] = []
    options["in_base_dir"] = str(Path(args.case_dir).parent)
    options["out_base_dir"] = str(out_base_dir)
    options["test_skip_gt_precompute"] = bool(args.test_skip_gt_precompute)
    options["test_cache_case_assets"] = bool(args.test_cache_case_assets)
    options["test_gpu_preprocess_adjoint"] = bool(args.test_gpu_preprocess_adjoint)
    options["test_gpu_preprocess_fe_ifft"] = bool(args.test_gpu_preprocess_fe_ifft)
    options["profile_preprocess"] = bool(args.profile_preprocess)
    options_ms = (time.perf_counter() - options_start) * 1000.0

    dataset_start = time.perf_counter()
    dataset = CMRx4DFlowDataSet(**options)
    dataset_init_ms = (time.perf_counter() - dataset_start) * 1000.0
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for case_dir={args.case_dir}")

    model_start = time.perf_counter()
    model = FlowVN(**options)
    model_init_ms = (time.perf_counter() - model_start) * 1000.0

    checkpoint_start = time.perf_counter()
    load_checkpoint(model, args.ckpt_path)
    checkpoint_load_ms = (time.perf_counter() - checkpoint_start) * 1000.0

    to_device_start = time.perf_counter()
    model = model.to(device).eval()
    synchronize(torch, device)
    model_to_device_ms = (time.perf_counter() - to_device_start) * 1000.0

    compile_setup_start = time.perf_counter()
    if bool(args.compile_flowvn_activations):
        model = compile_flowvn_activation_modules(
            model,
            compile_flowvn_activations=True,
            compile_mode=str(args.compile_mode),
        )
    if bool(args.compile_flowvn_regularizer):
        model = compile_flowvn_regularizer_modules(
            model,
            compile_flowvn_regularizer=True,
            compile_mode=str(args.compile_mode),
        )
    if bool(args.compile_model):
        model = torch.compile(model, mode=str(args.compile_mode))
    compile_setup_ms = (time.perf_counter() - compile_setup_start) * 1000.0
    cuda_graph_runner = None
    if bool(args.cuda_graph_replay):
        cuda_graph_runner = CudaGraphForwardRunner(
            torch_module=torch,
            model=model,
            precision_mode=str(args.precision_mode),
            device=device,
            warmup_iters=int(args.cuda_graph_warmup_iters),
            verify_eager=bool(args.cuda_graph_verify_eager),
        )

    callback = CMRSaveCallback()
    max_items = int(args.max_items)
    stop = len(dataset) if max_items <= 0 else min(len(dataset), int(args.real_index_start) + max_items)
    base_indices = list(range(int(args.real_index_start), stop))
    repeat_index_cycles = max(1, int(args.repeat_index_cycles))
    indices = base_indices * repeat_index_cycles
    gpu_preprocess_input_diff = None
    if bool(args.check_gpu_preprocess_inputs):
        if not bool(args.test_gpu_preprocess_adjoint):
            raise ValueError("--check-gpu-preprocess-inputs requires --test-gpu-preprocess-adjoint")
        gpu_preprocess_input_diff = compare_gpu_preprocess_inputs(
            torch_module=torch,
            dataset=dataset,
            options=options,
            indices=indices,
            device=device,
        )

    def load_dataset_item(item_idx: int):
        getitem_start = time.perf_counter()
        item = dataset[item_idx]
        dataset_getitem_ms = (time.perf_counter() - getitem_start) * 1000.0
        return item, dataset_getitem_ms

    def process_item(item: dict[str, Any], item_indices: list[int], dataset_getitem_ms, loader_wait_ms: float):
        batch_device_start = time.perf_counter()
        batch = tensor_to_device(torch, item, device)
        synchronize(torch, device)
        batch_to_device_ms = (time.perf_counter() - batch_device_start) * 1000.0
        gpu_preprocess_wall_ms, gpu_preprocess_cuda_ms = finalize_deferred_preprocess(
            torch, batch, device
        )

        recon, forward_wall_ms, forward_cuda_ms, cuda_graph_stats = benchmark_forward(
            torch,
            model,
            batch,
            str(args.precision_mode),
            cudagraph_mark_step_begin=bool(args.cudagraph_mark_step_begin),
            cuda_graph_runner=cuda_graph_runner,
        )
        callback_payload = make_callback_outputs(torch, batch, recon, forward_cuda_ms)

        callback_start = time.perf_counter()
        for segment_pos, output in enumerate(callback_payload["outputs"]):
            item_batch = slice_batch_for_item(torch, batch, segment_pos)
            if metric_accumulator is not None:
                metric_accumulator.update(
                    item_batch,
                    callback_payload["recon_cpu"][segment_pos : segment_pos + 1],
                )
            callback.on_test_batch_end(
                trainer=None,
                pl_module=None,
                outputs=output,
                batch=item_batch,
                batch_idx=item_indices[segment_pos],
            )
        save_callback_ms = (time.perf_counter() - callback_start) * 1000.0
        batch_size = int(batch["seg_idx"].numel())

        record = {
            "dataset_index": int(item_indices[0]),
            "dataset_indices": [int(i) for i in item_indices],
            "batch_size": int(batch_size),
            "case_dir": meta_at(batch["case_dir"], 0),
            "out_dir": meta_at(batch["out_dir"], 0),
            "seg_idx": [int(v) for v in batch["seg_idx"].detach().cpu().view(-1).tolist()],
            "usrate": [int(v) for v in batch["usrate"].detach().cpu().view(-1).tolist()],
            "dataset_getitem_ms": optional_float(dataset_getitem_ms),
            "dataset_getitem_wait_ms": float(loader_wait_ms),
            "loader_wait_ms": float(loader_wait_ms),
            "batch_to_device_ms": float(batch_to_device_ms),
            "gpu_preprocess_wall_ms": float(gpu_preprocess_wall_ms),
            "gpu_preprocess_cuda_ms": gpu_preprocess_cuda_ms,
            "forward_wall_ms": float(forward_wall_ms),
            "forward_cuda_ms": forward_cuda_ms,
            "cpu_transfer_ms": float(callback_payload["cpu_transfer_ms"]),
            "save_callback_ms": float(save_callback_ms),
        }
        if cuda_graph_stats is not None:
            record.update(cuda_graph_stats)
        if "preprocess_profile" in item:
            record["preprocess_profile"] = json_safe(torch, item["preprocess_profile"])
        item_records.append(record)

    item_records = []
    metric_accumulator = ValMetricAccumulator(torch) if str(args.real_mode) == "val" else None
    loop_start = time.perf_counter()
    if args.loader_mode == "dataloader":
        dataloader_kwargs = {
            "batch_size": int(args.segment_batch_size),
            "shuffle": False,
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
        }
        if int(args.num_workers) > 0 and args.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader = DataLoader(IndexedDataset(dataset, indices), **dataloader_kwargs)
        loader_iter = iter(loader)
        for _ in range(len(loader)):
            wait_start = time.perf_counter()
            item = next(loader_iter)
            loader_wait_ms = (time.perf_counter() - wait_start) * 1000.0
            item_indices = [int(v) for v in torch.as_tensor(item["_dataset_index"]).view(-1).tolist()]
            process_item(
                item=item,
                item_indices=item_indices,
                dataset_getitem_ms=None,
                loader_wait_ms=loader_wait_ms,
            )
    else:
        executor = None
        future = None
        if bool(args.prefetch_next_item) and indices:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(load_dataset_item, indices[0])
        try:
            for pos, item_idx in enumerate(indices):
                if future is not None:
                    wait_start = time.perf_counter()
                    item, dataset_getitem_ms = future.result()
                    loader_wait_ms = (time.perf_counter() - wait_start) * 1000.0
                    next_pos = pos + 1
                    future = (
                        executor.submit(load_dataset_item, indices[next_pos])
                        if executor is not None and next_pos < len(indices)
                        else None
                    )
                else:
                    item, dataset_getitem_ms = load_dataset_item(item_idx)
                    loader_wait_ms = dataset_getitem_ms

                process_item(
                    item=item,
                    item_indices=[int(item_idx)],
                    dataset_getitem_ms=dataset_getitem_ms,
                    loader_wait_ms=loader_wait_ms,
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    loop_wall_ms = (time.perf_counter() - loop_start) * 1000.0

    epoch_save_start = time.perf_counter()
    callback.on_test_epoch_end(trainer=None, pl_module=None)
    save_epoch_end_ms = (time.perf_counter() - epoch_save_start) * 1000.0
    metric_summary = None if metric_accumulator is None else metric_accumulator.compute()

    total_process_wall_ms = (time.perf_counter() - SCRIPT_START) * 1000.0
    output_files = list_output_files(out_base_dir)

    result = {
        "mode": "FlowVN end-to-end inference benchmark",
        "source": "real",
        "device": str(device),
        "config": str(args.config),
        "case_dir": str(args.case_dir),
        "real_mode": str(args.real_mode),
        "usrate": int(args.usrate),
        "run_name": str(run_name),
        "out_base_dir": str(out_base_dir),
        "dataset_len": int(len(dataset)),
        "base_indices": base_indices,
        "repeat_index_cycles": int(repeat_index_cycles),
        "indices": indices,
        "compile_model": bool(args.compile_model),
        "compile_flowvn_activations": bool(args.compile_flowvn_activations),
        "compile_flowvn_regularizer": bool(args.compile_flowvn_regularizer),
        "compile_mode": str(args.compile_mode),
        "cudagraph_mark_step_begin": bool(args.cudagraph_mark_step_begin),
        "cuda_graph_replay": bool(args.cuda_graph_replay),
        "cuda_graph_warmup_iters": int(args.cuda_graph_warmup_iters),
        "cuda_graph_verify_eager": bool(args.cuda_graph_verify_eager),
        "flowvn_activation_lowmem": bool(args.flowvn_activation_lowmem),
        "flowvn_transpose_conv_as_conv": bool(args.flowvn_transpose_conv_as_conv),
        "flowvn_dc_centered_fft": str(args.flowvn_dc_centered_fft),
        "test_skip_gt_precompute": bool(args.test_skip_gt_precompute),
        "test_cache_case_assets": bool(args.test_cache_case_assets),
        "test_gpu_preprocess_adjoint": bool(args.test_gpu_preprocess_adjoint),
        "test_gpu_preprocess_fe_ifft": bool(args.test_gpu_preprocess_fe_ifft),
        "check_gpu_preprocess_inputs": bool(args.check_gpu_preprocess_inputs),
        "profile_preprocess": bool(args.profile_preprocess),
        "prefetch_next_item": bool(args.prefetch_next_item),
        "loader_mode": str(args.loader_mode),
        "num_workers": int(args.num_workers),
        "prefetch_factor": None if args.prefetch_factor is None else int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "segment_batch_size": int(args.segment_batch_size),
        "allow_tf32": bool(args.allow_tf32),
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "deterministic_algorithms": bool(args.deterministic_algorithms),
        "precision_mode": str(args.precision_mode),
        "options_load_ms": float(options_ms),
        "dataset_init_ms": float(dataset_init_ms),
        "model_init_ms": float(model_init_ms),
        "checkpoint_load_ms": float(checkpoint_load_ms),
        "model_to_device_ms": float(model_to_device_ms),
        "compile_setup_ms": float(compile_setup_ms),
        "inference_loop_wall_ms": float(loop_wall_ms),
        "save_epoch_end_ms": float(save_epoch_end_ms),
        "total_process_wall_ms": float(total_process_wall_ms),
        "peak_memory": cuda_peak_memory(torch, device),
        "gpu_preprocess_input_diff": gpu_preprocess_input_diff,
        "items": item_records,
        "forward_cuda_ms_sum": float(
            sum(float(item["forward_cuda_ms"] or 0.0) for item in item_records)
        ),
        "forward_wall_ms_sum": float(sum(item["forward_wall_ms"] for item in item_records)),
        "dataset_getitem_ms_sum": sum_optional(item_records, "dataset_getitem_ms"),
        "dataset_getitem_wait_ms_sum": sum_optional(item_records, "dataset_getitem_wait_ms"),
        "loader_wait_ms_sum": sum_optional(item_records, "loader_wait_ms"),
        "batch_to_device_ms_sum": float(sum(item["batch_to_device_ms"] for item in item_records)),
        "gpu_preprocess_wall_ms_sum": sum_optional(item_records, "gpu_preprocess_wall_ms"),
        "gpu_preprocess_cuda_ms_sum": sum_optional(item_records, "gpu_preprocess_cuda_ms"),
        "cuda_graph_capture_wall_ms_sum": sum_optional(item_records, "cuda_graph_capture_wall_ms"),
        "cuda_graph_copy_cuda_ms_sum": sum_optional(item_records, "cuda_graph_copy_cuda_ms"),
        "cuda_graph_replay_cuda_ms_sum": sum_optional(item_records, "cuda_graph_replay_cuda_ms"),
        "cuda_graph_recaptures": int(
            sum(1 for item in item_records if bool(item.get("cuda_graph_recaptured", False)))
        ),
        "cpu_transfer_ms_sum": float(sum(item["cpu_transfer_ms"] for item in item_records)),
        "save_callback_ms_sum": float(sum(item["save_callback_ms"] for item in item_records)),
        "metrics": metric_summary,
        "output_files": output_files,
        "notes": [
            "total_process_wall_ms starts near script import time, after Python interpreter startup.",
            "forward_cuda_ms includes first-use torch.compile work on the first item when compile flags are enabled.",
            "loader_wait_ms is main-thread wait for the next item; in direct mode it matches dataset_getitem_wait_ms.",
            "dataset_getitem_ms is only measured directly in loader_mode=direct; DataLoader worker item time is not directly observable here.",
            "gpu_preprocess_cuda_ms measures deferred GPU preprocessing; it includes FE IFFT only when --test-gpu-preprocess-fe-ifft is enabled.",
            "cudagraph_mark_step_begin calls torch.compiler.cudagraph_mark_step_begin before every model forward when enabled.",
            "cuda_graph_replay captures only the fixed-shape model forward; total_process_wall_ms includes capture overhead, while forward_cuda_ms measures static input copy plus graph replay.",
            "cuda_graph_verify_eager runs an extra eager forward after each graph replay and should not be used for clean speed timing.",
            "output_files is empty until all expected velocity segments for a case/usrate are processed.",
            "repeat_index_cycles repeats the selected base_indices in one Python process for amortization studies.",
        ],
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
