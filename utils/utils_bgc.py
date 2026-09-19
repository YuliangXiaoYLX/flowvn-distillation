"""
Adopted from
https://github.com/lolacaro/PCMRI-MSAC
"""

import numpy as np
from skimage import filters
from einops import rearrange


def execute_MSAC(im, corr_fit_order=3, th=0.01):
    """
    dimensions if 4d [dim1 dim2 slice time velocity]
    dimensions if 2d [dim1 dim2 1 time velocity]
    """
    np.random.seed(274612)
    im = rearrange(im, "nv nt spe pe fe -> fe pe spe nt nv")
    im = im[..., 1:] * np.conj(im[..., 0:1]) / (np.abs(im[..., 0:1]) + 1e-9)
    phase_im_t = np.angle(im) / np.pi
    magnitude_im_t = np.abs(im)

    parameters = {}
    parameters["msac_thresh"] = th
    parameters["samples"] = 10
    parameters["trials"] = 100
    parameters["msac_fit_order"] = 1
    parameters["n_enc"] = im.shape[-1]
    magnitude_im_t = np.mean(magnitude_im_t, axis=-1)
    magnitude_im_t = magnitude_im_t / np.max(magnitude_im_t)

    magnitude = np.mean(magnitude_im_t, 3)
    phase = np.mean(phase_im_t, 3)
    mask_mgn = magnitude > filters.threshold_multiotsu(magnitude, classes=5)[0]
    m, n, l, t, d = phase_im_t.shape
    run_4d = d == 3

    if run_4d:
        allm = np.ones([m, n, l])
        aa = np.argwhere(allm == 1)
        ma = np.argwhere(mask_mgn == 1)

        ap = np.zeros((aa.shape[0], 6))
        mp = np.zeros((ma.shape[0], 6))
        ap[:, 3:6] = aa
        mp[:, 3:6] = ma
        for dir_idx in np.arange(d):
            pdir = phase[:, :, :, dir_idx]
            ap[:, dir_idx] = pdir[allm == 1]
            mp[:, dir_idx] = pdir[mask_mgn == 1]
    else:
        allm = np.ones([m, n])
        aa = np.argwhere(allm == 1)
        ma = np.argwhere(mask_mgn[:, :, 0] == 1)

        ap = np.zeros((aa.shape[0], 3))
        mp = np.zeros((ma.shape[0], 3))
        ap[:, 1:3] = aa
        mp[:, 1:3] = ma

        ap[:, 0] = np.squeeze(phase[allm == 1])
        mp[:, 0] = np.squeeze(phase[mask_mgn[:, :, 0] == 1])

    mfunc, cfunc, fmfunc, fcfunc, dfunc = get_functions(run_4d, parameters["msac_fit_order"], corr_fit_order)

    functions = {}
    functions["msac_fit"] = mfunc
    functions["msac_dist"] = dfunc
    cost, inlier_indx = msac(mp, parameters, functions)

    model = cfunc(mp, **{"inlierIndx": inlier_indx})
    est, xyz = fcfunc(model, ap)

    bgr_msac = np.zeros([m, n, l, d])
    aw = np.where(allm == 1)

    if run_4d:
        bgr_msac[aw[0], aw[1], aw[2], :] = est
    else:
        bgr_msac[aw[0], aw[1]] = est[:, :, np.newaxis]

    corr_t = np.swapaxes(np.expand_dims(bgr_msac, -1), 3, 4)
    corr_t = rearrange(corr_t, "fe pe spe nt nv -> nv nt spe pe fe")
    return corr_t * np.pi


def get_functions(run_4d, fitorder_msac, fitorder_corr):
    if run_4d:
        mfunc = lambda points, **kwargs: fit4d(fitorder_msac, points, **kwargs)
        cfunc = lambda points, **kwargs: fit4d(fitorder_corr, points, **kwargs)
        fmfunc = lambda coeffs, points: eval4d(fitorder_msac, coeffs, points)
        fcfunc = lambda coeffs, points: eval4d(fitorder_corr, coeffs, points)
        dfunc = lambda coeffs, points: dist4d(fitorder_msac, coeffs, points)
    else:
        mfunc = lambda points, **kwargs: fit2d(fitorder_msac, points, **kwargs)
        cfunc = lambda points, **kwargs: fit2d(fitorder_corr, points, **kwargs)
        fmfunc = lambda coeffs, points: eval2d(fitorder_msac, coeffs, points)
        fcfunc = lambda coeffs, points: eval2d(fitorder_corr, coeffs, points)
        dfunc = lambda coeffs, points: dist2d(fitorder_msac, coeffs, points)
    return mfunc, cfunc, fmfunc, fcfunc, dfunc


def get_inout4d(order, points):
    xyz = points[:, 0:3]
    p1 = points[:, 3]
    p2 = points[:, 4]
    p3 = points[:, 5]
    no_p = points.shape[0]

    if order == 0:
        A = np.ones([no_p, 1])
        no = 1
    elif order == 1:
        A = np.ones([no_p, 4])
        A[:, 1] = p1
        A[:, 2] = p2
        A[:, 3] = p3
        no = 4
    elif order == 2:
        A = np.ones([no_p, 10])
        A[:, 1] = p1
        A[:, 2] = p2
        A[:, 3] = p3
        A[:, 4] = p1 ** 2
        A[:, 5] = p1 * p2
        A[:, 6] = p1 * p3
        A[:, 7] = p2 ** 2
        A[:, 8] = p2 * p3
        A[:, 9] = p3 ** 2
        no = 10
    elif order == 3:
        A = np.ones([no_p, 20])
        A[:, 1] = p1
        A[:, 2] = p2
        A[:, 3] = p3
        A[:, 4] = p1 ** 2
        A[:, 5] = p1 * p2
        A[:, 6] = p1 * p3
        A[:, 7] = p2 ** 2
        A[:, 8] = p2 * p3
        A[:, 9] = p3 ** 2
        A[:, 10] = p1 ** 3
        A[:, 11] = (p1 ** 2) * p2
        A[:, 12] = (p1 ** 2) * p3
        A[:, 13] = p2 ** 3
        A[:, 14] = (p2 ** 2) * p1
        A[:, 15] = (p2 ** 2) * p3
        A[:, 16] = p3 ** 3
        A[:, 17] = (p3 ** 2) * p1
        A[:, 18] = (p3 ** 2) * p2
        A[:, 19] = p1 * p2 * p3
        no = 20
    return xyz, A, no


def fit4d(order, points, **kwargs):
    inlier_indx = np.ones((points.shape[0], 3), dtype=bool)
    if len(kwargs) > 0:
        inlier_indx = kwargs["inlierIndx"]

    indx = inlier_indx == 1
    xyz, A, no = get_inout4d(order, points)
    coeffs = np.zeros((no, 3))
    for d in np.arange(3):
        if order == 0:
            coeffs[0, d] = np.mean(xyz[indx[:, d], d], 0)
        else:
            inlier = indx[:, d]
            coeffs[:, d] = np.linalg.lstsq(A[inlier, :], xyz[inlier, d], rcond=None)[0]
    return coeffs


def dist4d(order, coeffs, points):
    est, xyz = eval4d(order, coeffs, points)
    dist = np.abs((xyz - est) / 2)
    return dist


def eval4d(order, coeffs, points):
    xyz, A, no = get_inout4d(order, points)
    est = A @ coeffs
    return est, xyz


def get_inout2d(order, points):
    xyz = points[:, 0:1]
    p1 = points[:, 1]
    p2 = points[:, 2]
    no_p = points.shape[0]

    if order == 0:
        A = np.ones([no_p, 1])
        no = 1
    elif order == 1:
        A = np.ones([no_p, 3])
        A[:, 1] = p1
        A[:, 2] = p2
        no = 3
    elif order == 2:
        A = np.ones([no_p, 6])
        A[:, 1] = p1
        A[:, 2] = p2
        A[:, 3] = p1 ** 2
        A[:, 4] = p1 * p2
        A[:, 5] = p2 ** 2
        no = 6
    elif order == 3:
        A = np.ones([no_p, 10])
        A[:, 1] = p1
        A[:, 2] = p2
        A[:, 3] = p1 ** 2
        A[:, 4] = p1 * p2
        A[:, 5] = p2 ** 2
        A[:, 6] = p1 ** 3
        A[:, 7] = (p1 ** 2) * p2
        A[:, 8] = p2 ** 3
        A[:, 9] = (p2 ** 2) * p1
        no = 10
    return xyz, A, no


def fit2d(order, points, **kwargs):
    inlier_indx = np.ones((points.shape[0], 1), dtype=bool)
    if len(kwargs) > 0:
        inlier_indx = kwargs["inlierIndx"]

    indx = (inlier_indx == 1)[:, 0]
    xyz, A, no = get_inout2d(order, points)
    coeffs = np.zeros((no, 1))

    if order == 0:
        coeffs[0, 0] = np.mean(xyz[indx, 0], 0)
    else:
        coeffs[:, 0] = np.linalg.lstsq(A[indx, :], xyz[indx, 0], rcond=None)[0]
    return coeffs


def dist2d(order, coeffs, points):
    est, xyz = eval2d(order, coeffs, points)
    dist = np.abs((xyz - est) / 2)
    return dist


def eval2d(order, coeffs, points):
    xyz, A, no = get_inout2d(order, points)
    est = A @ coeffs
    return est, xyz


def msac(points, parameters, functions):
    samples = parameters["samples"]
    threshold = parameters["msac_thresh"]
    trials = parameters["trials"]
    n_enc = parameters["n_enc"]

    msac_fit = functions["msac_fit"]
    msac_dist = functions["msac_dist"]

    no_p = points.shape[0]

    best_cost = np.ones(n_enc) * threshold * no_p
    best_inliers = np.zeros([no_p, n_enc])

    for i in np.arange(trials):
        indx = np.random.permutation(no_p)[0:samples]
        sample = points[indx, :]

        coeffs = msac_fit(sample)
        residuals = msac_dist(coeffs, points)

        residuals[residuals > threshold] = threshold
        inliers = residuals < threshold

        cost = np.sum(residuals, 0)
        comp_cost = best_cost > cost

        best_cost[comp_cost] = cost[comp_cost]
        best_inliers[:, comp_cost] = inliers[:, comp_cost]

    return best_cost, best_inliers
