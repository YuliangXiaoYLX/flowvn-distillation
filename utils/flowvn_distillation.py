import hashlib
import math
from pathlib import Path

from networks.flowvn import FlowVN
from utils.misc_utils import mri_forward_op
import torch
import torch.nn as nn


def uniform_teacher_stage_map(
    student_num_stages: int,
    teacher_num_stages: int,
) -> tuple[int, ...]:
    student_num_stages = int(student_num_stages)
    teacher_num_stages = int(teacher_num_stages)
    if student_num_stages <= 0 or teacher_num_stages <= 0:
        raise ValueError("Student and teacher stage counts must be positive")
    if teacher_num_stages < student_num_stages:
        raise ValueError("Teacher must have at least as many stages as the student")
    stage_map = tuple(
        ((student_index + 1) * teacher_num_stages) // student_num_stages - 1
        for student_index in range(student_num_stages)
    )
    if len(set(stage_map)) != student_num_stages:
        raise ValueError(f"Teacher stage map is not one-to-one: {stage_map}")
    return stage_map


def _strip_teacher_state_dict(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("network."):
            key = key[len("network.") :]
        key = key.replace("._orig_mod.", ".")
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        stripped[key] = value
    return stripped


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_student_from_teacher_checkpoint(
    student: nn.Module,
    checkpoint_path: str | Path,
    teacher_num_stages: int,
) -> dict:
    if not hasattr(student, "cell_list"):
        raise ValueError("Teacher initialization requires a FlowVN cell_list")
    student_num_stages = len(student.cell_list)
    stage_map = uniform_teacher_stage_map(
        student_num_stages,
        teacher_num_stages,
    )
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_state = _strip_teacher_state_dict(
        checkpoint.get("state_dict", checkpoint)
    )
    student_state = student.state_dict()
    initialized_state = dict(student_state)
    copied_per_stage = [0] * student_num_stages

    for student_key, student_value in student_state.items():
        parts = student_key.split(".", 2)
        if len(parts) != 3 or parts[0] != "cell_list":
            continue
        student_stage = int(parts[1])
        teacher_stage = stage_map[student_stage]
        teacher_key = f"cell_list.{teacher_stage}.{parts[2]}"
        if teacher_key not in teacher_state:
            raise KeyError(
                f"Teacher checkpoint is missing mapped tensor {teacher_key}"
            )
        teacher_value = teacher_state[teacher_key]
        if tuple(teacher_value.shape) != tuple(student_value.shape):
            raise ValueError(
                f"Shape mismatch for {student_key} <- {teacher_key}: "
                f"student={tuple(student_value.shape)} "
                f"teacher={tuple(teacher_value.shape)}"
            )
        initialized_state[student_key] = teacher_value.to(
            dtype=student_value.dtype,
            device=student_value.device,
        )
        copied_per_stage[student_stage] += 1

    if any(count == 0 for count in copied_per_stage):
        raise ValueError(
            "Teacher initialization did not copy every student stage: "
            f"{copied_per_stage}"
        )
    student.load_state_dict(initialized_state, strict=True)
    return {
        "teacher_checkpoint_path": str(checkpoint_path),
        "teacher_checkpoint_sha256": _sha256_file(checkpoint_path),
        "student_num_stages": student_num_stages,
        "teacher_num_stages": int(teacher_num_stages),
        "teacher_stage_map": list(stage_map),
        "copied_tensor_count": sum(copied_per_stage),
        "copied_tensors_per_stage": copied_per_stage,
    }


class FlowVNTeacher:
    def __init__(self, options: dict, ckpt_path: str | None, device: torch.device):
        self.device = device
        teacher_options = dict(options)
        teacher_options["num_stages"] = int(options.get("teacher_num_stages", 16))
        teacher_options["mode"] = "test"
        teacher_options["exp_loss"] = False
        self.model = self.build_teacher(teacher_options, ckpt_path)

    @torch.inference_mode()
    def run_once(
        self,
        batch,
        precision_mode: str,
        return_intermediates: bool = False,
    ):
        if self.device is None or self.device != batch["imdata_p1"].device:
            self.device = batch["imdata_p1"].device
            self.model.to(self.device)
        enabled = precision_mode != "fp32" and batch["imdata_p1"].device.type == "cuda"
        dtype = torch.float16 if precision_mode == "amp_fp16" else torch.bfloat16
        with torch.autocast("cuda", dtype=dtype, enabled=enabled):
            return self.model(
                batch["imdata_p1"],
                batch["kdata_p1"],
                batch["coil_sens"],
                batch["usrate_true"],
                return_intermediates=return_intermediates,
            )

    def requires_grad(self, model, requires_grad: bool = False):
        for param in model.parameters():
            param.requires_grad_(requires_grad)
        return model

    def load_checkpoint(self, model, ckpt_path: str | None):
        if not ckpt_path:
            return None
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        stripped_state = _strip_teacher_state_dict(
            checkpoint.get("state_dict", checkpoint)
        )
        model.load_state_dict(stripped_state, strict=True)
        return model

    def build_teacher(self, options: dict, ckpt_path: str | None):
        model = FlowVN(**options).to(self.device).eval()
        model = self.load_checkpoint(model, ckpt_path)
        if model is None:
            return None
        model = self.requires_grad(model, requires_grad=False)
        return model


class DistillationLossHelper:
    def __init__(self, options: dict, device: torch.device):
        self.options = options
        self.device = device
        self.L1Loss = nn.L1Loss()

    def teacher_schedule_scale(self, current_epoch) -> float:
        schedule = str(
            self.options.get("distill_teacher_schedule", "constant")
        ).strip().lower()
        if schedule == "constant":
            return 1.0
        if schedule != "warmup_cosine":
            raise ValueError(
                "distill_teacher_schedule must be constant or warmup_cosine"
            )

        epoch = max(0, int(current_epoch))
        max_epochs = max(1, int(self.options.get("epoch", 1)))
        active_epochs = int(
            self.options.get("distill_teacher_active_epochs", 0) or 0
        )
        if active_epochs < 0:
            raise ValueError("distill_teacher_active_epochs cannot be negative")
        if active_epochs > 0:
            if epoch >= active_epochs:
                return 0.0
            max_epochs = min(max_epochs, active_epochs)
        warmup_epochs = max(
            0,
            int(self.options.get("distill_teacher_warmup_epochs", 0)),
        )
        final_scale = float(
            self.options.get("distill_teacher_final_scale", 0.1)
        )
        if not 0.0 <= final_scale <= 1.0:
            raise ValueError("distill_teacher_final_scale must be in [0, 1]")
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return min(1.0, float(epoch + 1) / float(warmup_epochs))
        decay_steps = max(1, max_epochs - warmup_epochs - 1)
        progress = min(
            1.0,
            max(0.0, float(epoch - warmup_epochs) / float(decay_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_scale + (1.0 - final_scale) * cosine

    def _teacher_l1(self, prediction, target):
        loss = self.L1Loss(prediction, target)
        if bool(self.options.get("distill_normalize_teacher_losses", False)):
            epsilon = float(self.options.get("distill_normalization_epsilon", 1e-6))
            denominator = target.detach().abs().mean().clamp_min(epsilon)
            loss = loss / denominator
        return loss

    def compute_loss(
        self,
        batch,
        student_outputs,
        teacher_outputs,
        current_epoch,
        student_intermediates=None,
        teacher_intermediates=None,
    ):
        distill_teacher_weight = float(
            self.options.get("distill_teacher_weight", 1.0)
        )
        distill_kspace_weight = float(
            self.options.get("distill_kspace_weight", 1.0)
        )
        distill_gt_weight = float(self.options.get("distill_gt_weight", 1.0))
        distill_trajectory_weight = float(
            self.options.get("distill_trajectory_weight", 0.0)
        )
        distill_update_weight = float(
            self.options.get("distill_update_weight", 0.0)
        )
        teacher_scale = self.teacher_schedule_scale(current_epoch)
        zero = student_outputs.real.sum() * 0.0
        losses = {
            "image_teacher_loss": zero,
            "kspace_teacher_consistency_loss": zero,
            "student_gt_loss": zero,
            "trajectory_teacher_loss": zero,
            "update_teacher_loss": zero,
        }
        if student_outputs is not None and distill_gt_weight != 0.0:
            losses["student_gt_loss"] = self.student_gt_loss(
                batch, student_outputs, current_epoch
            )

        teacher_weights = (
            distill_teacher_weight,
            distill_kspace_weight,
            distill_trajectory_weight,
            distill_update_weight,
        )
        teacher_active = teacher_scale > 0.0 and any(
            weight != 0.0 for weight in teacher_weights
        )
        if teacher_active and teacher_outputs is None:
            raise ValueError(
                "Active teacher loss weights require teacher outputs"
            )
        if teacher_active and student_outputs is not None:
            if distill_teacher_weight != 0.0:
                losses["image_teacher_loss"] = self.image_teacher_loss(
                    student_outputs, teacher_outputs
                )
            if distill_kspace_weight != 0.0:
                losses["kspace_teacher_consistency_loss"] = (
                    self.kspace_teacher_consistency_loss(
                        batch, student_outputs, teacher_outputs
                    )
                )
            if distill_trajectory_weight != 0.0 or distill_update_weight != 0.0:
                if student_intermediates is None or teacher_intermediates is None:
                    raise ValueError(
                        "Active trajectory/update distillation requires both "
                        "student and teacher intermediates"
                    )
                if distill_trajectory_weight != 0.0:
                    losses[
                        "trajectory_teacher_loss"
                    ] = self.trajectory_teacher_loss(
                        student_intermediates,
                        teacher_intermediates,
                    )
                if distill_update_weight != 0.0:
                    losses["update_teacher_loss"] = self.update_teacher_loss(
                        batch["imdata_p1"],
                        student_intermediates,
                        teacher_intermediates,
                    )

        total_losses = (
            teacher_scale
            * distill_teacher_weight
            * losses.get("image_teacher_loss", 0.0)
            + teacher_scale
            * distill_kspace_weight
            * losses.get("kspace_teacher_consistency_loss", 0.0)
            + distill_gt_weight * losses.get("student_gt_loss", 0.0)
            + teacher_scale
            * distill_trajectory_weight
            * losses.get("trajectory_teacher_loss", 0.0)
            + teacher_scale
            * distill_update_weight
            * losses.get("update_teacher_loss", 0.0)
        )
        losses["teacher_schedule_scale"] = torch.as_tensor(
            teacher_scale,
            dtype=student_outputs.real.dtype,
            device=student_outputs.device,
        )
        losses["total_loss"] = total_losses
        return total_losses, losses

    def image_teacher_loss(self, student_outputs, teacher_outputs):
        return self._teacher_l1(student_outputs, teacher_outputs)

    def student_gt_loss(self, batch, student_outputs, current_epoch):
        if self.device is None or batch["imdata_p1"].device != self.device:
            self.device = batch["imdata_p1"].device
        if self.options["exp_loss"]:
            tau = current_epoch / 10
            w = torch.exp(
                torch.Tensor(
                    [
                        -tau * (self.options["num_stages"] - k + 1)
                        for k in range(self.options["num_stages"])
                    ]
                ).to(self.device)
            )
            w /= torch.sum(w)
            return (
                torch.sum(
                    w
                    * torch.norm(
                        student_outputs - batch["gt"],
                        p=1,
                        dim=[1, 2, 3, 4, 5, 6],
                    )
                )
                / 40000
            )
        return self.L1Loss(
            student_outputs - batch["gt"][:, 0], torch.zeros_like(student_outputs)
        )

    def kspace_teacher_consistency_loss(self, batch, student_outputs, teacher_outputs):
        acquired_mask = abs(batch["kdata_p1"][:, :, 0, :, 0, :, :]) != 0
        region = str(
            self.options.get("distill_kspace_region", "acquired")
        ).strip().lower()
        if region == "acquired":
            loss_mask = acquired_mask
        elif region == "unacquired":
            loss_mask = ~acquired_mask
        elif region == "all":
            loss_mask = torch.ones_like(acquired_mask, dtype=torch.bool)
        else:
            raise ValueError(
                "distill_kspace_region must be acquired, unacquired, or all"
            )
        student_kspace = mri_forward_op(
            student_outputs, batch["coil_sens"], loss_mask.float()
        )
        teacher_kspace = mri_forward_op(
            teacher_outputs, batch["coil_sens"], loss_mask.float()
        )
        delta = torch.view_as_real(student_kspace - teacher_kspace)
        return self.L1Loss(delta, torch.zeros_like(delta))

    def _paired_teacher_intermediates(
        self,
        student_intermediates,
        teacher_intermediates,
    ):
        student_intermediates = tuple(student_intermediates)
        teacher_intermediates = tuple(teacher_intermediates)
        stage_map = uniform_teacher_stage_map(
            len(student_intermediates),
            len(teacher_intermediates),
        )
        return student_intermediates, tuple(
            teacher_intermediates[index] for index in stage_map
        )

    def trajectory_teacher_loss(
        self,
        student_intermediates,
        teacher_intermediates,
    ):
        student_layers, teacher_layers = self._paired_teacher_intermediates(
            student_intermediates,
            teacher_intermediates,
        )
        losses = [
            self._teacher_l1(student_layer, teacher_layer)
            for student_layer, teacher_layer in zip(
                student_layers,
                teacher_layers,
            )
        ]
        return torch.stack(losses).mean()

    def update_teacher_loss(
        self,
        initial_reconstruction,
        student_intermediates,
        teacher_intermediates,
    ):
        student_layers, teacher_layers = self._paired_teacher_intermediates(
            student_intermediates,
            teacher_intermediates,
        )
        student_previous = initial_reconstruction
        teacher_previous = initial_reconstruction
        losses = []
        for student_layer, teacher_layer in zip(student_layers, teacher_layers):
            student_update = student_layer - student_previous
            teacher_update = teacher_layer - teacher_previous
            losses.append(self._teacher_l1(student_update, teacher_update))
            student_previous = student_layer
            teacher_previous = teacher_layer
        return torch.stack(losses).mean()
