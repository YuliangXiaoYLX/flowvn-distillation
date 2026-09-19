import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_flowvn_revision_ablation import (
    analyze_revision_ablation,
    flatten_analysis_records,
    load_accepted_revision_runs,
    render_revision_latex_tables,
    write_analysis_outputs,
)
from utils.flowvn_results import REVISION_METRICS


ARMS = ("supervised", "final_only", "final_trajectory", "full_kd")
SEEDS = ("12345", "23456")


def _rows(seed, arm):
    rows = []
    for case_index in range(2):
        for usrate in (10, 20):
            base = 10.0 + case_index + usrate / 100.0
            values = {
                metric: (0.70 + case_index / 100.0 if metric == "ssim" else base)
                for metric in REVISION_METRICS
            }
            if arm in ("final_only", "final_trajectory", "full_kd"):
                for metric in REVISION_METRICS:
                    values[metric] += 0.02 if metric == "ssim" else -0.2
                values["velocity_vector_rmse_cm_s"] -= 0.3
            if arm in ("final_trajectory", "full_kd"):
                values["nrmse"] -= 0.1
                values["velocity_vector_rmse_cm_s"] += 0.1
            rows.append(
                {
                    "case_id": f"case_{case_index:03d}-{'a' * 12}",
                    "slice_start": 0,
                    "usrate": usrate,
                    **values,
                }
            )
    return rows


class RevisionAblationAnalysisTests(unittest.TestCase):
    def test_component_gates_follow_the_frozen_co_primary_rule(self):
        runs = {
            (seed, arm): _rows(seed, arm)
            for seed in SEEDS
            for arm in ARMS
        }

        result = analyze_revision_ablation(
            runs,
            expected_seeds=SEEDS,
            expected_cases=2,
            expected_usrates=(10, 20),
            bootstrap_samples=200,
            bootstrap_seed=7,
        )

        decisions = result["component_decisions"]
        self.assertEqual(decisions["final_output_matching"]["status"], "supported")
        self.assertEqual(
            decisions["trajectory_matching"]["status"],
            "unfavorable_co_primary",
        )
        self.assertEqual(decisions["update_matching"]["status"], "inconclusive")
        final_nrmse = result["contrasts"]["final_output_matching"]["overall"][
            "nrmse"
        ]
        self.assertTrue(final_nrmse["both_seeds_favorable"])
        self.assertTrue(final_nrmse["bootstrap_supports_favorable"])
        self.assertAlmostEqual(final_nrmse["mean_delta_treatment_minus_control"], -0.2)
        self.assertLess(final_nrmse["mean_relative_delta_percent"], 0.0)
        self.assertEqual(
            len(final_nrmse["case_bootstrap_95_ci_relative_delta_percent"]),
            2,
        )
        trajectory_velocity = result["contrasts"]["trajectory_matching"][
            "overall"
        ]["velocity_vector_rmse_cm_s"]
        self.assertTrue(trajectory_velocity["bootstrap_supports_unfavorable"])
        self.assertIn("10", result["arm_summaries"]["final_only"]["by_usrate"])

        records = flatten_analysis_records(result)
        self.assertEqual(len(records["arm_summaries"]), 4 * 3 * len(REVISION_METRICS))
        self.assertEqual(len(records["contrasts"]), 4 * 3 * len(REVISION_METRICS))
        self.assertEqual(len(records["component_decisions"]), 3)
        final_record = next(
            row
            for row in records["contrasts"]
            if row["contrast"] == "final_output_matching"
            and row["scope"] == "overall"
            and row["metric"] == "nrmse"
        )
        self.assertAlmostEqual(
            final_record["mean_delta_treatment_minus_control"], -0.2
        )
        self.assertAlmostEqual(final_record["seed_12345_delta"], -0.2)

    def test_output_bundle_is_hash_bound_and_non_overwriting(self):
        runs = {
            (seed, arm): _rows(seed, arm)
            for seed in SEEDS
            for arm in ARMS
        }
        result = analyze_revision_ablation(
            runs,
            expected_seeds=SEEDS,
            expected_cases=2,
            expected_usrates=(10, 20),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_analysis_outputs(
                result,
                output_dir=Path(tmpdir),
                prefix="revision_fixture",
            )
            manifest = json.loads(outputs["manifest"].read_text())
            with outputs["contrasts"].open(newline="") as handle:
                contrast_rows = list(csv.DictReader(handle))

            self.assertEqual(manifest["schema_version"], 1)
            for record in manifest["outputs"]:
                path = Path(record["path"])
                self.assertEqual(
                    record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )
            self.assertEqual(
                len(contrast_rows), 4 * 3 * len(REVISION_METRICS)
            )
            self.assertTrue(outputs["seed_arm_means_latex"].is_file())
            self.assertTrue(outputs["overall_latex"].is_file())
            self.assertTrue(outputs["component_contrasts_full_latex"].is_file())
            self.assertTrue(outputs["joint_recipe_overall_latex"].is_file())
            self.assertTrue(outputs["joint_recipe_uncertainty_latex"].is_file())
            self.assertTrue(outputs["by_usrate_latex"].is_file())
            self.assertTrue(
                outputs["by_usrate_component_contrasts_latex"].is_file()
            )
            self.assertTrue(outputs["joint_recipe_by_usrate_latex"].is_file())
            output_labels = {record["label"] for record in manifest["outputs"]}
            self.assertIn("overall_latex", output_labels)
            self.assertIn("seed_arm_means_latex", output_labels)
            self.assertIn("component_contrasts_full_latex", output_labels)
            self.assertIn("joint_recipe_overall_latex", output_labels)
            self.assertIn("joint_recipe_uncertainty_latex", output_labels)
            self.assertIn("by_usrate_latex", output_labels)
            self.assertIn("by_usrate_component_contrasts_latex", output_labels)
            self.assertIn("joint_recipe_by_usrate_latex", output_labels)
            with self.assertRaisesRegex(FileExistsError, "preserve"):
                write_analysis_outputs(
                    result,
                    output_dir=Path(tmpdir),
                    prefix="revision_fixture",
                )

    def test_run_loader_binds_physical_validation_audits_and_csv_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            specs = []
            for seed in SEEDS:
                for arm in ARMS:
                    csv_path = root / f"{seed}_{arm}.csv"
                    rows = _rows(seed, arm)
                    with csv_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    audit_path = root / f"{seed}_{arm}.audit.json"
                    audit_path.write_text(
                        json.dumps(
                            {
                                "accepted": True,
                                "paper_metrics_eligible": True,
                                "physical_velocity_required": True,
                                "required_metrics": list(REVISION_METRICS),
                                "validation_csv": {
                                    "path": str(csv_path),
                                    "sha256": hashlib.sha256(
                                        csv_path.read_bytes()
                                    ).hexdigest(),
                                },
                                "cohort": {
                                    "n_groups": 4,
                                    "n_cases": 2,
                                    "usrates": [10, 20],
                                },
                                "provenance": {
                                    "exit_status": 0,
                                    "config_sha256": "1" * 64,
                                    "checkpoint_sha256": "2" * 64,
                                    "mask_backend": {
                                        "backend": "challenge",
                                        "verified": True,
                                    },
                                },
                            }
                        )
                    )
                    specs.append((seed, arm, csv_path, audit_path))

            runs, provenance = load_accepted_revision_runs(
                specs,
                expected_seeds=SEEDS,
                expected_cases=2,
                expected_usrates=(10, 20),
            )

            self.assertEqual(len(runs), 8)
            self.assertEqual(len(provenance), 8)
            csv_path.write_text(csv_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_accepted_revision_runs(
                    specs,
                    expected_seeds=SEEDS,
                    expected_cases=2,
                    expected_usrates=(10, 20),
                )

    def test_latex_tables_are_complete_and_claim_safe(self):
        runs = {
            (seed, arm): _rows(seed, arm)
            for seed in SEEDS
            for arm in ARMS
        }
        result = analyze_revision_ablation(
            runs,
            expected_seeds=SEEDS,
            expected_cases=2,
            expected_usrates=(10, 20),
            bootstrap_samples=20,
            bootstrap_seed=7,
        )

        tables = render_revision_latex_tables(result)

        self.assertEqual(
            set(tables),
            {
                "overall",
                "seed_arm_means",
                "component_contrasts_full",
                "joint_recipe_overall",
                "joint_recipe_uncertainty",
                "by_usrate",
                "by_usrate_component_contrasts",
                "joint_recipe_by_usrate",
            },
        )
        self.assertIn(r"\label{tab:revision_overall}", tables["overall"])
        self.assertIn("Supervised S8", tables["overall"])
        self.assertIn("VENC RMSE", tables["overall"])
        self.assertIn(
            r"\label{tab:revision_seed_arm_means}",
            tables["seed_arm_means"],
        )
        self.assertIn("Supervised S8", tables["seed_arm_means"])
        self.assertIn("Final-only KD", tables["seed_arm_means"])
        self.assertIn("Final+trajectory KD", tables["seed_arm_means"])
        self.assertIn("Full KD", tables["seed_arm_means"])
        self.assertIn("VENC RMSE", tables["seed_arm_means"])
        self.assertIn("Seed 12345", tables["seed_arm_means"])
        self.assertIn("Seed 23456", tables["seed_arm_means"])
        self.assertIn(
            r"\label{tab:revision_component_contrasts_full}",
            tables["component_contrasts_full"],
        )
        for metric_label in (
            "nRMSE",
            "SSIM",
            "RelErr",
            "AngErr",
            "Norm. $L_1$",
            "VENC RMSE",
        ):
            self.assertIn(metric_label, tables["component_contrasts_full"])
        self.assertIn(
            r"\label{tab:joint_recipe_overall}",
            tables["joint_recipe_overall"],
        )
        self.assertIn("VENC RMSE", tables["joint_recipe_overall"])
        self.assertIn("Paired change", tables["joint_recipe_overall"])
        self.assertIn(
            r"\label{tab:joint_recipe_uncertainty}",
            tables["joint_recipe_uncertainty"],
        )
        self.assertIn("VENC RMSE", tables["joint_recipe_uncertainty"])
        self.assertIn("Relative 95\\% CI", tables["joint_recipe_uncertainty"])
        self.assertIn(
            "descriptive and unadjusted for multiplicity",
            tables["by_usrate"],
        )
        self.assertIn("10 & Supervised S8", tables["by_usrate"])
        for metric_label in (
            "nRMSE",
            "SSIM",
            "RelErr",
            "AngErr",
            "Norm. $L_1$",
            "VENC RMSE",
        ):
            self.assertIn(metric_label, tables["by_usrate"])
        self.assertIn(
            r"\label{tab:supp_revision_physical_by_rate}",
            tables["by_usrate_component_contrasts"],
        )
        self.assertIn("Final-output matching", tables["by_usrate_component_contrasts"])
        self.assertIn("Trajectory matching", tables["by_usrate_component_contrasts"])
        self.assertIn("Update matching", tables["by_usrate_component_contrasts"])
        self.assertIn("95\\% CI", tables["by_usrate_component_contrasts"])
        for metric_label in (
            "nRMSE",
            "SSIM",
            "RelErr",
            "AngErr",
            "Norm. $L_1$",
            "VENC RMSE",
        ):
            self.assertIn(metric_label, tables["by_usrate_component_contrasts"])
        for usrate in (10, 20):
            self.assertIn(f"{usrate} & Norm. $L_1$", tables["by_usrate_component_contrasts"])
        self.assertIn(
            r"\label{tab:joint_recipe_by_usrate}",
            tables["joint_recipe_by_usrate"],
        )
        self.assertIn("Full KD versus supervised S8", tables["joint_recipe_by_usrate"])
        self.assertIn("VENC RMSE", tables["joint_recipe_by_usrate"])
        for table_name, latex in tables.items():
            self.assertIn(
                "VENC RMSE",
                latex,
                msg=f"quality table {table_name} omitted the physical endpoint",
            )
        self.assertNotIn("nan", "".join(tables.values()).lower())


if __name__ == "__main__":
    unittest.main()
