import numpy as np
import torch

from utils.pytorch_ssim import SSIM3D


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def SSIM(pred, gt, segmask=None, eps=1e-12):
    """
    SSIM within segmask (3D). Returns the mean SSIM over voxels where segmask==1.

    pred, gt: shape (Nv, Nt, SPE, PE, FE)
    segmask : shape (SPE, PE, FE)
    """
    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    ssim_fn = SSIM3D(window_size=11, size_average=False)

    pred = pred * segmask[None, None]
    gt = gt * segmask[None, None]

    gt_max = float(np.max(np.abs(gt)))
    if (not np.isfinite(gt_max)) or (gt_max <= eps):
        return float("nan")

    pred = pred / gt_max
    gt = gt / gt_max

    pred_t = _to_tensor(pred).float()
    gt_t = _to_tensor(gt).float()

    ssim_map = ssim_fn(pred_t, gt_t)
    roi = _to_tensor(segmask.astype(bool)).to(ssim_map.device)
    roi = roi.unsqueeze(0).unsqueeze(0)

    roi_sum = (ssim_map * roi).sum()
    roi_vox = roi.sum()
    if torch.is_tensor(roi_vox):
        if roi_vox.item() <= 0:
            return float("nan")
    elif float(roi_vox) <= 0:
        return float("nan")

    roi_cnt = roi_vox * gt.shape[1] * gt.shape[0]

    return (roi_sum / roi_cnt).item()


def nRMSE(pred, gt, segmask=None, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    while segmask.ndim < gt.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt.shape).astype(np.float32)

    n = np.sum(mask)
    mse = np.sum(((pred - gt) ** 2) * mask) / (n + eps)
    denom = np.max(gt * mask) + eps
    return np.sqrt(mse) / denom


def RelErr(pred, gt, segmask=None, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    gt_mag = np.linalg.norm(gt, axis=0)
    pred_mag = np.linalg.norm(pred, axis=0)

    while segmask.ndim < gt_mag.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, gt_mag.shape).astype(np.float32)

    numerator = np.sum(((gt_mag - pred_mag) ** 2) * mask)
    denominator = np.sum((gt_mag ** 2) * mask) + eps
    return np.sqrt(numerator / denominator)


def AngErr(pred, gt, segmask=None, eps=1e-8):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if segmask is None:
        segmask = np.ones(gt.shape[-3:], dtype=bool)
    segmask = np.asarray(segmask, dtype=bool)

    dot = np.sum(pred * gt, axis=0)
    norm_p = np.linalg.norm(pred, axis=0)
    norm_g = np.linalg.norm(gt, axis=0)
    cos_sim = np.clip(dot / (norm_p * norm_g + eps), -1.0, 1.0)

    error_map = np.arccos(cos_sim)

    while segmask.ndim < error_map.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, error_map.shape).astype(np.float32)

    n_valid = np.sum(mask) + 1e-12
    return (np.sum(error_map * mask) / n_valid) / np.pi * 180.0


def VelocityVectorRMSE(pred_phase, gt_phase, venc_cm_s, segmask):
    """Return circular velocity-vector RMSE inside a spatial ROI, in cm/s."""
    pred_phase = np.asarray(pred_phase, dtype=np.float64)
    gt_phase = np.asarray(gt_phase, dtype=np.float64)
    if pred_phase.ndim != 5 or gt_phase.ndim != 5:
        raise ValueError(
            "Phase fields must have shape (direction,time,SPE,PE,FE)"
        )
    if pred_phase.shape != gt_phase.shape:
        raise ValueError(
            f"Phase-field shape mismatch: {pred_phase.shape} vs {gt_phase.shape}"
        )
    if not np.all(np.isfinite(pred_phase)) or not np.all(np.isfinite(gt_phase)):
        raise ValueError("Phase fields must contain only finite values")

    venc = np.asarray(venc_cm_s, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(venc)):
        raise ValueError("VENC values must contain only finite values")
    if np.any(venc <= 0.0):
        raise ValueError("VENC values must be strictly positive")
    if venc.size == 1:
        venc = np.repeat(venc, pred_phase.shape[0])
    if venc.size != pred_phase.shape[0]:
        raise ValueError(
            f"Expected {pred_phase.shape[0]} directional VENC values, got {venc.size}"
        )

    circular_phase_error = np.angle(np.exp(1j * (pred_phase - gt_phase)))
    velocity_error = circular_phase_error / np.pi * venc.reshape(
        (venc.size,) + (1,) * (pred_phase.ndim - 1)
    )
    vector_squared_error = np.sum(np.square(velocity_error), axis=0)

    roi = np.asarray(segmask, dtype=bool)
    expected_roi_shape = pred_phase.shape[-3:]
    if roi.shape != expected_roi_shape:
        raise ValueError(
            f"Expected segmask shape {expected_roi_shape}, got {roi.shape}"
        )
    roi = np.broadcast_to(roi[None, ...], vector_squared_error.shape)
    if not np.any(roi):
        raise ValueError("segmask has no True elements")
    return float(np.sqrt(np.mean(vector_squared_error[roi])))


def ComplexDiffErr(pred, ref, segmask, eps=1e-12):
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    segmask = np.asarray(segmask, dtype=bool)

    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {ref.shape}")

    while segmask.ndim < pred.ndim:
        segmask = segmask[None, ...]
    mask = np.broadcast_to(segmask, pred.shape)

    if not np.any(mask):
        raise ValueError("segmask has no True elements.")

    err = np.sqrt(np.sum(np.abs(pred[mask] - ref[mask]) ** 2))
    predn = np.sqrt(np.sum(np.abs(pred[mask]) ** 2))
    refn = np.sqrt(np.sum(np.abs(ref[mask]) ** 2))

    return float(err / (predn + refn + eps))
