import numpy as np


def _gaussian_weights(pe, spe, sigma_x, sigma_y):
    x = np.arange(pe, dtype=np.float64)
    y = np.arange(spe, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    cx = (pe - 1) / 2.0
    cy = (spe - 1) / 2.0

    sigma_x = max(float(sigma_x), 1e-6)
    sigma_y = max(float(sigma_y), 1e-6)

    wx = ((xx - cx) ** 2) / (2.0 * sigma_x ** 2)
    wy = ((yy - cy) ** 2) / (2.0 * sigma_y ** 2)
    w = np.exp(-(wx + wy))
    return w


def _center_mask(pe, spe, center_radius_x, center_radius_y):
    x = np.arange(pe, dtype=np.float64)
    y = np.arange(spe, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    cx = (pe - 1) / 2.0
    cy = (spe - 1) / 2.0

    x_norm = (xx - cx) / (pe / 2.0)
    y_norm = (yy - cy) / (spe / 2.0)

    rx = max(float(center_radius_x), 1e-6)
    ry = max(float(center_radius_y), 1e-6)
    return ((x_norm / rx) ** 2 + (y_norm / ry) ** 2) <= 1.0


def fun_mask_gen_2d(
    mask_size,
    total_points,
    pattern_num,
    sigma_x,
    sigma_y,
    center_radius_x=0.5,
    center_radius_y=0.5,
    min_dist_factor=3,
    rep_decay_factor=0.5,
):
    """
    Generate ktGaussian masks with shape (SPE, PE, T).

    This local implementation replaces the external CMRx4DFlowMaskGeneration
    dependency so FlowVN can run as a standalone package.

    Args follow the existing call-sites for compatibility.
    """
    pe, spe = int(mask_size[0]), int(mask_size[1])
    t = int(pattern_num)
    total_points = int(total_points)

    total_grid = pe * spe
    total_points = int(np.clip(total_points, 1, total_grid))

    base_w = _gaussian_weights(pe, spe, sigma_x, sigma_y)
    c_mask = _center_mask(pe, spe, center_radius_x, center_radius_y)

    all_indices = np.arange(total_grid, dtype=np.int64)
    center_indices = np.flatnonzero(c_mask.ravel())
    non_center_indices = np.flatnonzero(~c_mask.ravel())

    # Track occupancy to slightly discourage repeatedly selecting exactly
    # the same locations over time (temporal diversity).
    occupancy = np.zeros(total_grid, dtype=np.float64)

    out = np.zeros((spe, pe, t), dtype=np.float32)

    for ti in range(t):
        selected = np.zeros(total_grid, dtype=bool)

        if center_indices.size > 0:
            if center_indices.size <= total_points:
                selected[center_indices] = True
            else:
                center_w = base_w.ravel()[center_indices]
                center_w = center_w / np.sum(center_w)
                pick = np.random.choice(center_indices, size=total_points, replace=False, p=center_w)
                selected[pick] = True

        n_selected = int(np.sum(selected))
        need = total_points - n_selected

        if need > 0:
            candidates = non_center_indices if non_center_indices.size > 0 else all_indices
            cand_w = base_w.ravel()[candidates].copy()

            decay = float(np.clip(rep_decay_factor, 0.0, 1.0))
            if decay < 1.0:
                cand_w *= np.power(decay, occupancy[candidates])

            # Keep parameter for compatibility; external implementation uses it
            # for spatial repulsion. Here we maintain API but do not apply
            # explicit geometric rejection to keep generation stable and fast.
            _ = min_dist_factor

            s = np.sum(cand_w)
            if s <= 0:
                cand_w = np.ones_like(cand_w) / cand_w.size
            else:
                cand_w /= s

            need = min(need, candidates.size)
            pick = np.random.choice(candidates, size=need, replace=False, p=cand_w)
            selected[pick] = True

        occupancy[selected] += 1.0
        out[:, :, ti] = selected.reshape(spe, pe).astype(np.float32)

    return out
