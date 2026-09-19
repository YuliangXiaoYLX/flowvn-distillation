#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.flowvn import FlowVN
from utils.flowvn_checkpoint import remap_state_dict_for_expected_keys

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_OPTIONS = {
    "mode": "test",
    "loss": "supervised",
    "network": "FlowVN",
    "features_in": 1,
    "D_size": 5,
    "T_size": 15,
    "num_act_weights": 71,
    "features_out": 24,
    "kernel_size": 5,
    "num_stages": 8,
    "act": "linear_flowvn",
    "grid": 0.25,
    "weight": 0.025,
    "vmin": -3.5,
    "vmax": 3.5,
    "sgd_momentum": True,
    "exp_loss": False,
    "flowvn_activation_lowmem": True,
    "flowvn_activation_backend": "torch",
    "flowvn_transpose_conv_as_conv": False,
    "flowvn_dc_centered_fft": "original",
}

FLOWVN_ACTIVATION_NAMES = (
    "activation1",
    "activation2",
    "activation3",
    "activation4",
    "activation5",
)


def load_options(config_path: str | None, args: argparse.Namespace) -> dict:
    options = dict(DEFAULT_OPTIONS)
    if config_path:
        if yaml is None:
            raise ImportError("PyYAML is required for --config")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        options.update(config)

    options["mode"] = "test"
    options["network"] = "FlowVN"
    options["loss"] = "supervised"
    options["exp_loss"] = False
    options["lowmem"] = False
    options["flowvn_activation_lowmem"] = bool(getattr(args, "flowvn_activation_lowmem", True))
    options["flowvn_activation_backend"] = str(
        getattr(args, "flowvn_activation_backend", "torch")
    )
    options["flowvn_transpose_conv_as_conv"] = bool(
        getattr(args, "flowvn_transpose_conv_as_conv", False)
    )
    options["flowvn_dc_centered_fft"] = str(
        getattr(args, "flowvn_dc_centered_fft", "original")
    )

    if args.num_stages is not None:
        options["num_stages"] = int(args.num_stages)
    if args.features_out is not None:
        options["features_out"] = int(args.features_out)
    if args.kernel_size is not None:
        options["kernel_size"] = int(args.kernel_size)
    if args.time_frames is not None:
        options["T_size"] = int(args.time_frames)
    if args.depth is not None:
        options["D_size"] = int(args.depth)
    return options


def apply_runtime_options(
    allow_tf32: bool,
    deterministic_algorithms: bool = True,
    cudnn_benchmark: bool = True,
):
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def complex_randn(shape, device):
    return torch.complex(torch.randn(shape, device=device), torch.randn(shape, device=device))


def synthetic_batch(args: argparse.Namespace, options: dict, device: torch.device):
    batch_size = int(args.batch_size)
    v = int(args.velocity_encodings)
    t = int(args.time_frames or options["T_size"])
    d = int(args.depth or options["D_size"])
    h = int(args.spatial_size)
    w = int(args.spatial_size)
    coils = int(args.coils)

    x = complex_randn((batch_size, v, t, d, h, w), device)
    c = complex_randn((batch_size, coils, d, h, w), device)
    c = c / torch.sqrt(torch.sum(torch.abs(c) ** 2, dim=1, keepdim=True).clamp_min(1e-12))

    mask = torch.rand((batch_size, v, t, h, w), device=device) < (1.0 / float(args.usrate))
    f = complex_randn((batch_size, v, coils, t, d, h, w), device)
    f = f * mask.unsqueeze(2).unsqueeze(4)
    usrate_true = torch.full((batch_size,), float(args.usrate), device=device, dtype=torch.float32)
    return x, f, c, usrate_true


def real_batch(args: argparse.Namespace, options: dict, device: torch.device):
    from utils.dataloader_CMRx4DFlow import CMRx4DFlowDataSet
    from utils.flowvn_mask_backend import require_mask_backend

    require_mask_backend(options)

    case_dir = Path(args.case_dir)
    ds_args = dict(options)
    ds_args.update(
        {
            "mode": str(args.real_mode),
            "input": str(case_dir),
            "usrate": [int(args.usrate)],
            "test_roots": None,
            "train_roots": [],
            "val_roots": [],
            "in_base_dir": str(case_dir.parent),
            "out_base_dir": str(case_dir.parent),
            # Prepare the adjoint before model-call timing, even when the
            # inference config defers preprocessing to the device.
            "test_gpu_preprocess_adjoint": False,
            "test_gpu_preprocess_fe_ifft": False,
        }
    )
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    dataset = CMRx4DFlowDataSet(**ds_args)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for case_dir={case_dir} real_mode={args.real_mode}")
    item = dataset[int(args.real_index) % len(dataset)]
    x = torch.as_tensor(item["imdata_p1"]).unsqueeze(0).to(device)
    f = torch.as_tensor(item["kdata_p1"]).unsqueeze(0).to(device)
    c = torch.as_tensor(item["coil_sens"]).unsqueeze(0).to(device)
    usrate_true = torch.as_tensor(item["usrate_true"], dtype=torch.float32, device=device)
    return x, f, c, usrate_true


def build_batch(args: argparse.Namespace, options: dict, device: torch.device):
    if args.case_dir:
        return real_batch(args, options, device)
    return synthetic_batch(args, options, device)


def load_checkpoint(model: FlowVN, ckpt_path: str | None):
    if not ckpt_path:
        return
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("network.") for k in state):
        stripped_state = {
            k[len("network.") :]: v
            for k, v in state.items()
            if k.startswith("network.")
        }
    else:
        stripped_state = dict(state)
    stripped_state = remap_state_dict_for_expected_keys(
        stripped_state, set(model.state_dict())
    )
    model.load_state_dict(stripped_state, strict=True)


def build_model(options: dict, ckpt_path: str | None, device: torch.device):
    model = FlowVN(**options).to(device).eval()
    load_checkpoint(model, ckpt_path)
    return model


def compile_flowvn_activation_modules(model: FlowVN, compile_flowvn_activations: bool, compile_mode: str):
    if not compile_flowvn_activations:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")
    for cell in model.cell_list:
        for activation_name in FLOWVN_ACTIVATION_NAMES:
            module = getattr(cell, activation_name)
            setattr(cell, activation_name, torch.compile(module, mode=compile_mode))
    return model


def compile_flowvn_regularizer_modules(model: FlowVN, compile_flowvn_regularizer: bool, compile_mode: str):
    if not compile_flowvn_regularizer:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")
    for cell in model.cell_list:
        cell.compile_regularizer(compile_mode=compile_mode)
    return model


def maybe_compile_model(
    model,
    compile_model: bool,
    compile_flowvn_activations: bool = False,
    compile_flowvn_regularizer: bool = False,
    compile_mode: str = "default",
):
    if compile_flowvn_activations:
        model = compile_flowvn_activation_modules(
            model,
            compile_flowvn_activations=compile_flowvn_activations,
            compile_mode=compile_mode,
        )
    if compile_flowvn_regularizer:
        model = compile_flowvn_regularizer_modules(
            model,
            compile_flowvn_regularizer=compile_flowvn_regularizer,
            compile_mode=compile_mode,
        )
    if not compile_model:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=compile_mode)


def run_once(model, batch, precision_mode: str):
    x, f, c, usrate = batch
    enabled = precision_mode != "fp32" and x.device.type == "cuda"
    dtype = torch.float16 if precision_mode == "amp_fp16" else torch.bfloat16
    with torch.autocast("cuda", dtype=dtype, enabled=enabled):
        return model(x, f, c, usrate)


def measure_model(model, batch, args: argparse.Namespace):
    device = batch[0].device
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    out = None
    with torch.inference_mode():
        warmup_wall_ms = []
        for _ in range(int(args.num_warmup)):
            warmup_start = time.perf_counter()
            out = run_once(model, batch, args.precision_mode)
            if use_cuda:
                torch.cuda.synchronize(device)
            warmup_wall_ms.append((time.perf_counter() - warmup_start) * 1000.0)
        if use_cuda:
            torch.cuda.synchronize(device)

        event_ms = []
        wall_start = time.perf_counter()
        for _ in range(int(args.num_iters)):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            iter_wall = time.perf_counter()
            out = run_once(model, batch, args.precision_mode)
            if use_cuda:
                end.record()
                torch.cuda.synchronize(device)
                event_ms.append(float(start.elapsed_time(end)))
            else:
                event_ms.append((time.perf_counter() - iter_wall) * 1000.0)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0

    warmup_wall_ms_total = float(np.sum(warmup_wall_ms))
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0
    return {
        "output": out.detach(),
        "first_warmup_wall_ms": float(warmup_wall_ms[0]) if warmup_wall_ms else None,
        "warmup_wall_ms_total": warmup_wall_ms_total,
        "event_ms_mean": float(np.mean(event_ms)),
        "event_ms_std": float(np.std(event_ms)),
        "wall_ms_total": float(wall_ms),
        "total_wall_ms_including_warmup": float(warmup_wall_ms_total + wall_ms),
        "peak_memory": peak_memory,
    }


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor):
    diff = (reference - candidate).abs().float()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def comparison_options(options: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    baseline_options = dict(options)
    candidate_options = dict(options)

    if args.compare_factor == "activation_lowmem":
        shared_transpose_conv_as_conv = bool(args.flowvn_transpose_conv_as_conv)
        baseline_options["flowvn_activation_lowmem"] = True
        candidate_options["flowvn_activation_lowmem"] = bool(args.flowvn_activation_lowmem)
        baseline_options["flowvn_transpose_conv_as_conv"] = shared_transpose_conv_as_conv
        candidate_options["flowvn_transpose_conv_as_conv"] = shared_transpose_conv_as_conv
    elif args.compare_factor == "transpose_conv_as_conv":
        shared_activation_lowmem = bool(args.flowvn_activation_lowmem)
        baseline_options["flowvn_activation_lowmem"] = shared_activation_lowmem
        candidate_options["flowvn_activation_lowmem"] = shared_activation_lowmem
        baseline_options["flowvn_transpose_conv_as_conv"] = False
        candidate_options["flowvn_transpose_conv_as_conv"] = bool(args.flowvn_transpose_conv_as_conv)
    elif args.compare_factor == "runtime":
        shared_activation_lowmem = bool(args.flowvn_activation_lowmem)
        shared_transpose_conv_as_conv = bool(args.flowvn_transpose_conv_as_conv)
        baseline_options["flowvn_activation_lowmem"] = shared_activation_lowmem
        candidate_options["flowvn_activation_lowmem"] = shared_activation_lowmem
        baseline_options["flowvn_transpose_conv_as_conv"] = shared_transpose_conv_as_conv
        candidate_options["flowvn_transpose_conv_as_conv"] = shared_transpose_conv_as_conv
    else:
        raise ValueError(f"Unsupported compare_factor={args.compare_factor}")

    return baseline_options, candidate_options


def parse_args():
    parser = argparse.ArgumentParser(description="FlowVN synthetic/real inference benchmark")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "inference.yaml"))
    parser.add_argument("--case-dir", type=str, default=None, help="optional real case directory")
    parser.add_argument("--real-mode", choices=["train", "val", "test"], default="test", help="dataset mode for --case-dir")
    parser.add_argument("--real-index", type=int, default=0, help="dataset index for --case-dir")
    parser.add_argument("--ckpt-path", type=str, default=None, help="optional FlowVN or Lightning checkpoint")
    parser.add_argument("--usrate", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--baseline-compile-flowvn-activations",
        action="store_true",
        help="compile baseline FlowVN activation modules too, for one-factor comparisons after activation compile",
    )
    parser.add_argument("--compile-flowvn-activations", action="store_true")
    parser.add_argument(
        "--baseline-compile-flowvn-regularizer",
        action="store_true",
        help="compile baseline FlowVN regularizer modules too, for one-factor comparisons after regularizer compile",
    )
    parser.add_argument("--compile-flowvn-regularizer", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--precision-mode", choices=["fp32", "amp_fp16", "amp_bf16"], default="fp32")
    parser.add_argument(
        "--flowvn-activation-lowmem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FlowVN activation mode; meaning depends on --compare-factor",
    )
    parser.add_argument(
        "--flowvn-transpose-conv-as-conv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="FlowVN regularizer transpose mode; meaning depends on --compare-factor",
    )
    parser.add_argument(
        "--flowvn-dc-centered-fft",
        choices=["original", "checkerboard"],
        default="original",
        help="FlowVN data-consistency centered FFT implementation",
    )
    parser.add_argument(
        "--compare-factor",
        choices=["activation_lowmem", "transpose_conv_as_conv", "runtime"],
        default="activation_lowmem",
        help="single factor varied between baseline and candidate",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--velocity-encodings", type=int, default=1)
    parser.add_argument("--coils", type=int, default=8)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--time-frames", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--features-out", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_json).expanduser() if args.output_json else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    options = load_options(args.config, args)
    batch = build_batch(args, options, device)

    baseline_options, candidate_options = comparison_options(options, args)

    apply_runtime_options(False)
    baseline = build_model(baseline_options, args.ckpt_path, device)

    apply_runtime_options(bool(args.allow_tf32))
    candidate = build_model(candidate_options, args.ckpt_path, device)
    candidate.load_state_dict(baseline.state_dict(), strict=True)

    baseline = maybe_compile_model(
        baseline,
        compile_model=False,
        compile_flowvn_activations=bool(args.baseline_compile_flowvn_activations),
        compile_flowvn_regularizer=bool(args.baseline_compile_flowvn_regularizer),
        compile_mode=str(args.compile_mode),
    )
    candidate = maybe_compile_model(
        candidate,
        compile_model=bool(args.compile_model),
        compile_flowvn_activations=bool(args.compile_flowvn_activations),
        compile_flowvn_regularizer=bool(args.compile_flowvn_regularizer),
        compile_mode=str(args.compile_mode),
    )

    apply_runtime_options(False)
    baseline_result = measure_model(baseline, batch, args)

    apply_runtime_options(bool(args.allow_tf32))
    candidate_result = measure_model(candidate, batch, args)

    result = {
        "source": "real" if args.case_dir else "synthetic",
        "device": str(device),
        "config": str(args.config),
        "real_mode": str(args.real_mode) if args.case_dir else None,
        "real_index": int(args.real_index) if args.case_dir else None,
        "shape": {
            "imdata_p1": list(batch[0].shape),
            "kdata_p1": list(batch[1].shape),
            "coil_sens": list(batch[2].shape),
            "usrate_true": list(batch[3].shape),
        },
        "compile_model": bool(args.compile_model),
        "baseline_compile_flowvn_activations": bool(
            args.baseline_compile_flowvn_activations
        ),
        "compile_flowvn_activations": bool(args.compile_flowvn_activations),
        "baseline_compile_flowvn_regularizer": bool(
            args.baseline_compile_flowvn_regularizer
        ),
        "compile_flowvn_regularizer": bool(args.compile_flowvn_regularizer),
        "compile_mode": str(args.compile_mode),
        "allow_tf32": bool(args.allow_tf32),
        "precision_mode": str(args.precision_mode),
        "compare_factor": str(args.compare_factor),
        "baseline_flowvn_activation_lowmem": bool(
            baseline_options.get("flowvn_activation_lowmem", True)
        ),
        "candidate_flowvn_activation_lowmem": bool(
            candidate_options.get("flowvn_activation_lowmem", True)
        ),
        "baseline_flowvn_transpose_conv_as_conv": bool(
            baseline_options.get("flowvn_transpose_conv_as_conv", False)
        ),
        "candidate_flowvn_transpose_conv_as_conv": bool(
            candidate_options.get("flowvn_transpose_conv_as_conv", False)
        ),
        "flowvn_dc_centered_fft": str(options.get("flowvn_dc_centered_fft", "original")),
        "baseline": {k: v for k, v in baseline_result.items() if k != "output"},
        "candidate": {k: v for k, v in candidate_result.items() if k != "output"},
    }
    result.update(diff_stats(baseline_result["output"], candidate_result["output"]))

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if output_path is not None:
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
