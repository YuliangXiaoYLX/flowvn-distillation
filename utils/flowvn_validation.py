import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from utils.flowvn_results import CASE_ID_CONVENTION as FLOWVN_CASE_ID_CONVENTION


VALIDATION_METRIC_FIELDS = (
    "nrmse",
    "ssim",
    "relerr",
    "angerr",
    "normalized_l1",
    "velocity_vector_rmse_cm_s",
)


def stable_case_id(case_dir: Path | str) -> str:
    path = Path(str(case_dir))
    relative_tail = "/".join(path.parts[-3:])
    digest = hashlib.sha256(relative_tail.encode("utf-8")).hexdigest()[:12]
    return f"{path.name}-{digest}"


def select_validation_filename_indices(
    filenames,
    *,
    case_id: Optional[str] = None,
    usrate: Optional[int] = None,
) -> list[int]:
    """Resolve a semantic validation filter against dataset filename metadata."""
    case_id = None if case_id in (None, "", "None") else str(case_id)
    usrate = None if usrate is None else int(usrate)
    if case_id is None and usrate is None:
        raise ValueError("At least one of case_id or usrate must be specified")

    indices = []
    matched_entries = []
    for index, entry in enumerate(filenames):
        if len(entry) < 4:
            raise ValueError(f"Invalid validation filename entry at index {index}: {entry}")
        entry_case_id = stable_case_id(entry[0])
        entry_usrate = int(entry[2])
        if case_id is not None and entry_case_id != case_id:
            continue
        if usrate is not None and entry_usrate != usrate:
            continue
        indices.append(index)
        matched_entries.append(entry)

    if not indices:
        raise ValueError(
            f"No validation samples match case_id={case_id!r}, usrate={usrate!r}"
        )
    if case_id is not None and usrate is not None:
        group_keys = {
            (stable_case_id(entry[0]), int(entry[1]), int(entry[2]))
            for entry in matched_entries
        }
        encodings = sorted(int(entry[3]) for entry in matched_entries)
        if len(group_keys) != 1 or encodings != [0, 1, 2, 3]:
            raise ValueError(
                "Semantic validation filter must resolve exactly one complete "
                f"four-encoding group; groups={sorted(group_keys)}, encodings={encodings}"
            )
    return indices


def build_qualitative_source(
    *,
    mag_pred: np.ndarray,
    mag_gt: np.ndarray,
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    segmentation: np.ndarray,
    time_index: Optional[int],
) -> dict[str, np.ndarray]:
    """Extract bounded 2D arrays without applying display-dependent scaling."""
    mag_pred = np.asarray(mag_pred)
    mag_gt = np.asarray(mag_gt)
    flow_pred = np.asarray(flow_pred)
    flow_gt = np.asarray(flow_gt)
    segmentation = np.asarray(segmentation, dtype=bool)
    if mag_pred.ndim != 5 or mag_gt.shape != mag_pred.shape:
        raise ValueError(
            f"Expected matching magnitude arrays [encoding,time,slice,y,x], got "
            f"{mag_pred.shape} and {mag_gt.shape}"
        )
    if flow_pred.ndim != 5 or flow_gt.shape != flow_pred.shape:
        raise ValueError(
            f"Expected matching velocity arrays [direction,time,slice,y,x], got "
            f"{flow_pred.shape} and {flow_gt.shape}"
        )
    if segmentation.ndim != 3 or segmentation.shape != mag_pred.shape[2:]:
        raise ValueError(
            f"Expected segmentation {mag_pred.shape[2:]}, got {segmentation.shape}"
        )

    slice_index = int(mag_pred.shape[2] // 2)
    if time_index is None:
        roi_native = segmentation[slice_index].astype(np.float32)
        velocity_energy = np.sum(
            np.square(np.abs(flow_gt[:, :, slice_index]))
            * roi_native[None, None, :, :],
            axis=(0, 2, 3),
        )
        time_index = int(np.argmax(velocity_energy))
    else:
        time_index = int(np.clip(time_index, 0, mag_pred.shape[1] - 1))
    roi = segmentation[slice_index].T.astype(np.uint8)
    magnitude_pred = np.stack(
        [np.abs(value[time_index, slice_index]).T * roi for value in mag_pred],
        axis=0,
    ).astype(np.float32, copy=False)
    magnitude_gt = np.stack(
        [np.abs(value[time_index, slice_index]).T * roi for value in mag_gt],
        axis=0,
    ).astype(np.float32, copy=False)
    velocity_pred = np.stack(
        [value[time_index, slice_index].T * roi for value in flow_pred],
        axis=0,
    ).astype(np.float32, copy=False)
    velocity_gt = np.stack(
        [value[time_index, slice_index].T * roi for value in flow_gt],
        axis=0,
    ).astype(np.float32, copy=False)
    return {
        "magnitude_pred": magnitude_pred,
        "magnitude_gt": magnitude_gt,
        "velocity_pred": velocity_pred,
        "velocity_gt": velocity_gt,
        "roi": roi,
        "time_index": np.asarray(time_index, dtype=np.int64),
        "slice_index": np.asarray(slice_index, dtype=np.int64),
    }


class StreamingValidationAccumulator:
    """Hold only incomplete velocity-encoding groups during validation."""

    CASE_ID_CONVENTION = FLOWVN_CASE_ID_CONVENTION

    def __init__(
        self,
        evaluate_group: Callable,
        expected_encodings: tuple[int, ...] = (0, 1, 2, 3),
        max_visualizations: int = 5,
    ):
        self._evaluate_group = evaluate_group
        self._expected_encodings = tuple(int(i) for i in expected_encodings)
        self._max_visualizations = max(int(max_visualizations), 0)
        self._pending = {}
        self._rows = []
        self._visualizations = []
        self._groups_seen = 0
        self._completed_groups = 0
        self._max_pending_groups = 0

    @property
    def pending_group_count(self) -> int:
        return len(self._pending)

    @property
    def completed_group_count(self) -> int:
        return self._completed_groups

    @property
    def rows(self) -> list[dict]:
        return list(self._rows)

    @property
    def visualizations(self) -> list[dict]:
        return list(self._visualizations)

    @staticmethod
    def _finite_mean(rows: list[dict], field: str) -> Optional[float]:
        values = []
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else None

    @staticmethod
    def _clean_scalar(value):
        if value is None:
            return None
        if isinstance(value, (float, int, np.floating, np.integer)):
            value = float(value)
            return value if np.isfinite(value) else None
        return value

    @staticmethod
    def _stable_case_id(case_dir: str) -> str:
        return stable_case_id(case_dir)

    @classmethod
    def _metric_summary(cls, rows: list[dict]) -> dict:
        return {
            field: cls._finite_mean(rows, field)
            for field in VALIDATION_METRIC_FIELDS
        }

    def summary(self) -> dict:
        return {
            **self._metric_summary(self._rows),
            "n_groups": self._groups_seen,
            "n_complete": self._completed_groups,
            "n_incomplete": len(self._pending),
            "max_pending_groups": self._max_pending_groups,
        }

    def summary_by_usrate(self) -> dict[int, dict]:
        by_usrate = {}
        for usrate in sorted({int(row["usrate"]) for row in self._rows}):
            rows = [row for row in self._rows if int(row["usrate"]) == usrate]
            by_usrate[usrate] = {
                **self._metric_summary(rows),
                "n_complete": len(rows),
            }
        return by_usrate

    def write_outputs(self, csv_path: Path | str) -> tuple[Path, Path]:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case_id",
            "slice_start",
            "usrate",
            "normalized_l1",
            "nrmse",
            "ssim",
            "relerr",
            "angerr",
            "velocity_vector_rmse_cm_s",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({field: row.get(field) for field in fieldnames})

        json_path = csv_path.with_suffix(".summary.json")
        report = {
            "case_id_convention": self.CASE_ID_CONVENTION,
            "summary": self.summary(),
            "by_usrate": self.summary_by_usrate(),
            "rows": self.rows,
        }
        with json_path.open("w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        return csv_path, json_path

    def write_visualizations(self, npz_path: Path | str) -> tuple[Path, Path]:
        """Persist bounded, publication-source visualization slices and provenance."""
        npz_path = Path(npz_path)
        if npz_path.suffix != ".npz":
            raise ValueError(f"Visualization archive must use .npz: {npz_path}")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {}
        samples = []
        for index, payload in enumerate(self._visualizations):
            prefix = f"sample_{index:03d}"
            sample_arrays = {"panel": np.asarray(payload["panel"])}
            sample_arrays.update(
                {
                    str(name): np.asarray(value)
                    for name, value in payload.get("source", {}).items()
                }
            )
            for name, value in sample_arrays.items():
                if not name.replace("_", "").isalnum():
                    raise ValueError(f"Invalid visualization array name: {name}")
                if value.dtype.hasobject:
                    raise ValueError(f"Object arrays are not allowed: {name}")
                arrays[f"{prefix}_{name}"] = value

            meta = dict(payload.get("meta", {}))
            samples.append(
                {
                    "index": index,
                    "case_id": str(meta.get("case_id", "")),
                    "slice_start": int(meta.get("slice_start", -1)),
                    "usrate": int(meta.get("usrate", -1)),
                    "arrays": {
                        name: {
                            "archive_key": f"{prefix}_{name}",
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                        }
                        for name, value in sorted(sample_arrays.items())
                    },
                }
            )

        np.savez_compressed(npz_path, **arrays)
        archive_sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        manifest_path = npz_path.with_suffix(".manifest.json")
        report = {
            "schema_version": 1,
            "case_id_convention": self.CASE_ID_CONVENTION,
            "count": len(samples),
            "archive": str(npz_path),
            "archive_sha256": archive_sha256,
            "panel_layout": (
                "rows: magnitude encodings then phase-difference velocity components; "
                "columns: prediction, ground truth, error"
            ),
            "scope": (
                "Bounded 2D source slices for qualitative figure construction; "
                "not additional independent validation observations"
            ),
            "samples": samples,
        }
        with manifest_path.open("w") as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        return npz_path, manifest_path

    def add_encoding(
        self,
        *,
        key: tuple[str, int, int],
        encoding: int,
        pred: np.ndarray,
        gt: np.ndarray,
        segmentation: np.ndarray,
        venc: Optional[np.ndarray] = None,
        normalized_l1: Optional[float] = None,
    ) -> Optional[dict]:
        encoding = int(encoding)
        if key not in self._pending:
            self._pending[key] = {
                "pred": {},
                "gt": {},
                "normalized_l1": {},
                "segmentation": np.asarray(segmentation),
                "venc": None if venc is None else np.asarray(venc),
            }
            self._groups_seen += 1
            self._max_pending_groups = max(
                self._max_pending_groups, len(self._pending)
            )

        pack = self._pending[key]
        pack["pred"][encoding] = np.asarray(pred)
        pack["gt"][encoding] = np.asarray(gt)
        if normalized_l1 is not None:
            pack["normalized_l1"][encoding] = float(normalized_l1)

        if any(i not in pack["pred"] or i not in pack["gt"] for i in self._expected_encodings):
            return None

        # Pop before metric evaluation so full arrays are not retained after completion,
        # including when a metric implementation raises.
        pack = self._pending.pop(key)
        pred_stack = np.stack([pack["pred"][i] for i in self._expected_encodings], axis=0)
        gt_stack = np.stack([pack["gt"][i] for i in self._expected_encodings], axis=0)
        want_visualization = len(self._visualizations) < self._max_visualizations
        metrics, visualization = self._evaluate_group(
            key,
            pred_stack,
            gt_stack,
            pack["segmentation"],
            pack["venc"],
            want_visualization,
        )

        case_dir, slice_start, usrate = key
        row = {
            "case_id": self._stable_case_id(str(case_dir)),
            "slice_start": int(slice_start),
            "usrate": int(usrate),
            **{
                field: self._clean_scalar(value)
                for field, value in dict(metrics).items()
            },
        }
        loss_values = list(pack["normalized_l1"].values())
        row["normalized_l1"] = (
            float(np.mean(loss_values)) if len(loss_values) > 0 else None
        )
        self._rows.append(row)
        self._completed_groups += 1

        if visualization is not None and want_visualization:
            if isinstance(visualization, dict):
                panel = visualization.get("panel")
                source = visualization.get("source", {})
            else:
                panel = visualization
                source = {}
            if panel is not None:
                self._visualizations.append(
                    {
                        "panel": np.asarray(panel),
                        "source": {
                            str(name): np.asarray(value)
                            for name, value in dict(source).items()
                        },
                        "meta": {
                            "case_id": row["case_id"],
                            "slice_start": row["slice_start"],
                            "usrate": row["usrate"],
                        },
                    }
                )
        return row
