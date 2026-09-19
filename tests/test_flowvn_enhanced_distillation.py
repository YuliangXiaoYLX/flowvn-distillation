import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from main import UnrolledNetwork
from networks.flowvn import FlowVN
from utils.flowvn_distillation import (
    DistillationLossHelper,
    initialize_student_from_teacher_checkpoint,
    uniform_teacher_stage_map,
)


class _AddStage(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = float(amount)

    def forward(self, x, _f, _c, _usrate, _state=None, dc_cache=None):
        del dc_cache
        update = torch.zeros_like(x)
        update[..., 0] = self.amount
        return x + update, update


def _minimal_flowvn(stage_amounts):
    model = FlowVN.__new__(FlowVN)
    nn.Module.__init__(model)
    model.nc = len(stage_amounts)
    model.exp_loss = False
    model.options = {
        "flowvn_dc_centered_fft": "original",
        "grad_check": False,
    }
    model.cell_list = nn.ModuleList(_AddStage(value) for value in stage_amounts)
    return model


class EnhancedFlowVNDistillationTests(unittest.TestCase):
    def test_uniform_stage_map_aligns_student_endpoints_to_teacher_pairs(self):
        self.assertEqual(uniform_teacher_stage_map(8, 16), tuple(range(1, 16, 2)))
        self.assertEqual(uniform_teacher_stage_map(2, 5), (1, 4))

    def test_teacher_checkpoint_initializes_uniform_student_stages(self):
        student = nn.Module()
        student.cell_list = nn.ModuleList(
            [nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)]
        )
        state = {}
        for stage_index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            state[f"network.cell_list.{stage_index}.weight"] = torch.tensor([[value]])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "teacher.ckpt"
            torch.save({"state_dict": state}, checkpoint)
            report = initialize_student_from_teacher_checkpoint(
                student,
                checkpoint,
                teacher_num_stages=4,
            )

        self.assertEqual(report["teacher_stage_map"], [1, 3])
        self.assertEqual(report["copied_tensor_count"], 2)
        self.assertEqual(student.cell_list[0].weight.item(), 2.0)
        self.assertEqual(student.cell_list[1].weight.item(), 4.0)

    def test_flowvn_can_return_final_and_intermediate_reconstructions(self):
        model = _minimal_flowvn((1.0, 2.0, 3.0))
        image = torch.zeros((1, 1, 1), dtype=torch.complex64)

        final, intermediates = model(
            image,
            torch.empty(0),
            torch.empty(0),
            torch.tensor([10.0]),
            return_intermediates=True,
        )

        self.assertEqual(len(intermediates), 3)
        self.assertTrue(torch.equal(intermediates[0], torch.ones_like(image)))
        self.assertTrue(torch.equal(intermediates[1], torch.full_like(image, 3.0)))
        self.assertTrue(torch.equal(intermediates[2], torch.full_like(image, 6.0)))
        self.assertTrue(torch.equal(final, intermediates[-1]))

    def test_trajectory_and_update_losses_use_paired_teacher_endpoints(self):
        helper = DistillationLossHelper(
            {
                "num_stages": 2,
                "teacher_num_stages": 4,
                "exp_loss": False,
                "distill_gt_weight": 0.0,
                "distill_teacher_weight": 0.0,
                "distill_kspace_weight": 0.0,
                "distill_trajectory_weight": 1.0,
                "distill_update_weight": 1.0,
                "distill_normalize_teacher_losses": False,
            },
            torch.device("cpu"),
        )
        zero = torch.zeros((1, 1, 1), dtype=torch.complex64)
        batch = {
            "imdata_p1": zero,
            "gt": zero.unsqueeze(1),
            "kdata_p1": torch.zeros((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64),
            "coil_sens": torch.empty(0),
        }
        student_intermediates = (
            torch.full_like(zero, 1.0),
            torch.full_like(zero, 4.0),
        )
        teacher_intermediates = tuple(
            torch.full_like(zero, value) for value in (1.0, 2.0, 4.0, 6.0)
        )

        total, losses = helper.compute_loss(
            batch,
            student_intermediates[-1],
            teacher_intermediates[-1],
            current_epoch=0,
            student_intermediates=student_intermediates,
            teacher_intermediates=teacher_intermediates,
        )

        self.assertAlmostEqual(losses["trajectory_teacher_loss"].item(), 1.5)
        self.assertAlmostEqual(losses["update_teacher_loss"].item(), 1.0)
        self.assertAlmostEqual(total.item(), 2.5)

    def test_unacquired_kspace_matching_uses_complement_of_acquired_mask(self):
        helper = DistillationLossHelper(
            {
                "num_stages": 1,
                "exp_loss": False,
                "distill_gt_weight": 0.0,
                "distill_teacher_weight": 0.0,
                "distill_kspace_weight": 1.0,
                "distill_kspace_region": "unacquired",
            },
            torch.device("cpu"),
        )
        student = torch.ones((1, 1, 1, 2, 2), dtype=torch.complex64)
        teacher = torch.zeros_like(student)
        kdata = torch.zeros((1, 1, 1, 1, 1, 2, 2), dtype=torch.complex64)
        kdata[..., 0, 0] = 1.0
        batch = {
            "imdata_p1": torch.zeros_like(student),
            "gt": torch.zeros_like(student).unsqueeze(1),
            "kdata_p1": kdata,
            "coil_sens": torch.empty(0),
        }

        def masked_identity(image, _coil_sens, mask):
            return image * mask

        with mock.patch(
            "utils.flowvn_distillation.mri_forward_op",
            side_effect=masked_identity,
        ):
            _, losses = helper.compute_loss(
                batch,
                student,
                teacher,
                current_epoch=0,
            )

        self.assertAlmostEqual(
            losses["kspace_teacher_consistency_loss"].item(),
            3.0 / 8.0,
        )

    def test_teacher_schedule_warms_up_then_cosine_decays(self):
        helper = DistillationLossHelper(
            {
                "epoch": 5,
                "distill_teacher_schedule": "warmup_cosine",
                "distill_teacher_warmup_epochs": 2,
                "distill_teacher_final_scale": 0.2,
            },
            torch.device("cpu"),
        )
        self.assertAlmostEqual(helper.teacher_schedule_scale(0), 0.5)
        self.assertAlmostEqual(helper.teacher_schedule_scale(1), 1.0)
        self.assertAlmostEqual(helper.teacher_schedule_scale(4), 0.2)

    def test_teacher_schedule_can_stop_before_gt_only_finetuning(self):
        helper = DistillationLossHelper(
            {
                "epoch": 8,
                "distill_teacher_schedule": "warmup_cosine",
                "distill_teacher_warmup_epochs": 1,
                "distill_teacher_active_epochs": 4,
                "distill_teacher_final_scale": 0.2,
            },
            torch.device("cpu"),
        )
        self.assertAlmostEqual(helper.teacher_schedule_scale(0), 1.0)
        self.assertAlmostEqual(helper.teacher_schedule_scale(1), 1.0)
        self.assertAlmostEqual(helper.teacher_schedule_scale(2), 0.6)
        self.assertAlmostEqual(helper.teacher_schedule_scale(3), 0.2)
        self.assertAlmostEqual(helper.teacher_schedule_scale(4), 0.0)
        self.assertAlmostEqual(helper.teacher_schedule_scale(7), 0.0)

    def test_gt_only_finetuning_does_not_require_teacher_outputs(self):
        helper = DistillationLossHelper(
            {
                "epoch": 8,
                "num_stages": 1,
                "exp_loss": False,
                "distill_gt_weight": 1.0,
                "distill_teacher_weight": 0.25,
                "distill_kspace_weight": 0.0,
                "distill_trajectory_weight": 0.25,
                "distill_update_weight": 0.1,
                "distill_teacher_schedule": "warmup_cosine",
                "distill_teacher_warmup_epochs": 1,
                "distill_teacher_active_epochs": 4,
                "distill_teacher_final_scale": 0.2,
            },
            torch.device("cpu"),
        )
        student = torch.ones((1, 1, 1), dtype=torch.complex64)
        batch = {
            "imdata_p1": torch.zeros_like(student),
            "gt": torch.zeros_like(student).unsqueeze(1),
            "kdata_p1": torch.empty(0),
            "coil_sens": torch.empty(0),
        }

        total, losses = helper.compute_loss(
            batch,
            student,
            None,
            current_epoch=4,
        )

        self.assertAlmostEqual(total.item(), 1.0)
        self.assertAlmostEqual(losses["student_gt_loss"].item(), 1.0)
        self.assertAlmostEqual(losses["teacher_schedule_scale"].item(), 0.0)
        self.assertAlmostEqual(losses["trajectory_teacher_loss"].item(), 0.0)

    def test_network_skips_teacher_forward_after_active_epochs(self):
        options = {
            "epoch": 10,
            "num_stages": 1,
            "exp_loss": False,
            "distill_gt_weight": 1.0,
            "distill_teacher_weight": 0.25,
            "distill_kspace_weight": 0.0,
            "distill_trajectory_weight": 0.25,
            "distill_update_weight": 0.1,
            "distill_teacher_schedule": "warmup_cosine",
            "distill_teacher_warmup_epochs": 1,
            "distill_teacher_active_epochs": 5,
            "distill_teacher_final_scale": 0.2,
        }
        fake_network = SimpleNamespace(
            current_epoch=5,
            distillation_loss_helper=DistillationLossHelper(
                options,
                torch.device("cpu"),
            ),
        )
        student = torch.ones((1, 1, 1), dtype=torch.complex64)
        batch = {
            "imdata_p1": torch.zeros_like(student),
            "gt": torch.zeros_like(student).unsqueeze(1),
            "kdata_p1": torch.empty(0),
            "coil_sens": torch.empty(0),
        }

        total, losses = UnrolledNetwork._flowvn_distillation_loss(
            fake_network,
            student,
            batch,
        )

        self.assertAlmostEqual(total.item(), 1.0)
        self.assertAlmostEqual(losses["teacher_schedule_scale"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
