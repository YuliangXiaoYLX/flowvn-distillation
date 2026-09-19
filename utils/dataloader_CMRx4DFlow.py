import os
import json
import random
import hashlib
import time
from pathlib import Path

import numpy as np
from einops import rearrange
from torch.utils.data import Dataset

from utils.utils_datasl import load_mat, read_params_csv
from utils.utils_flow import k2i_numpy
from utils.flowvn_mask_backend import fun_mask_gen_2d
from utils.misc_utils import mriAdjointOp


REQUIRED_FILES = ("kdata_full.mat", "coilmap.mat", "segmask.mat", "params.csv")

DEFAULT_OPTS = {
    "train_roots": [],
    "val_roots": [],
}


PREPROCESS_PROFILE_KEYS = (
    "load_coilmap_ms",
    "load_segmask_ms",
    "read_params_csv_ms",
    "load_kdata_ms",
    "load_mask_ms",
    "fe_ifft_ms",
    "gt_image_ms",
    "mask_rearrange_normalize_ms",
    "mri_adjoint_ms",
    "final_cast_packaging_ms",
)


def _new_preprocess_profile(enabled):
    if not enabled:
        return None
    profile = {key: 0.0 for key in PREPROCESS_PROFILE_KEYS}
    profile["case_asset_cache_hit"] = False
    profile["gt_skipped"] = False
    return profile


def _add_profile_ms(profile, key, start):
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0) + (time.perf_counter() - start) * 1000.0)


def _compute_out_dir(case_dir, in_base_dir, out_base_dir):
    """
    Map input case_dir -> output directory by preserving relative path under in_base_dir.
    If in_base_dir/out_base_dir is missing, fall back to out_base_dir/case_basename or case_dir.
    """
    case_dir = str(case_dir)
    if out_base_dir is None or str(out_base_dir) == "":
        return case_dir

    out_base_dir = str(out_base_dir)
    if in_base_dir is None or str(in_base_dir) == "":
        return str(Path(out_base_dir) / Path(case_dir).name)

    in_base_dir = str(in_base_dir)
    rel = os.path.relpath(case_dir, in_base_dir)
    return str(Path(out_base_dir) / rel)


def find_valid_cases(roots, required_files=REQUIRED_FILES, anchor="kdata_full.mat"):
    req = tuple(required_files)
    out, seen = [], set()
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
        for kpath in root.rglob(anchor):
            case_dir = kpath.parent
            if all((case_dir / f).is_file() for f in req):
                p = str(case_dir)
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    out.sort()
    return out


def load_split_cases(split_json, base_dir, required_files=REQUIRED_FILES):
    if split_json is None or str(split_json) == "":
        return []
    if base_dir is None or str(base_dir) == "":
        raise ValueError("split_base_dir (or in_base_dir) must be set when using split JSONs")

    split_path = Path(split_json)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split JSON not found: {split_path}")

    with open(split_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Split JSON must be a list of relative paths: {split_path}")

    base = Path(base_dir)
    req = tuple(required_files)
    out = []
    missing = []
    invalid = []

    for rel in data:
        case_dir = base / str(rel)
        if not case_dir.is_dir():
            missing.append(str(case_dir))
            continue
        if not all((case_dir / f).is_file() for f in req):
            invalid.append(str(case_dir))
            continue
        out.append(str(case_dir))

    if missing:
        raise FileNotFoundError(f"Missing case directories from split JSON: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if invalid:
        raise FileNotFoundError(f"Missing required files in split cases: {invalid[:5]}{' ...' if len(invalid) > 5 else ''}")

    out.sort()
    return out


def load_usmask_ktGaussian(case_dir, usrate, Nt, SPE, PE):
    mask_path = Path(case_dir) / f"usmask_ktGaussian{usrate}.mat"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    m = load_mat(str(mask_path), "usmask_ktGaussian")[()]
    expected = (1, Nt, 1, SPE, PE, 1)
    if m.shape != expected:
        raise ValueError(f"Unexpected mask shape {m.shape}, expected {expected}")
    m = np.squeeze(m, axis=(0, 2, 5))
    m = np.transpose(m, (1, 2, 0))
    mask = rearrange(m, "spe pe t -> 1 1 t 1 pe spe").astype(np.float32)
    return mask


def _stable_u32_seed(*items) -> int:
    key = "|".join(str(x) for x in items)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def generate_ktgaussian_mask_deterministic(usrate, Nt, SPE, PE, seed):
    total_points = (PE * SPE) // int(usrate)

    # fun_mask_gen_2d uses global RNG state; save/restore so caller behavior stays unchanged.
    np_state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        masks_spe_pe_t = fun_mask_gen_2d(
            mask_size=(PE, SPE),
            center_radius_x=0.5,
            center_radius_y=0.5,
            total_points=total_points,
            pattern_num=Nt,
            sigma_x=PE / 5,
            sigma_y=SPE / 5,
            min_dist_factor=3,
            rep_decay_factor=0.5,
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)

    return rearrange(masks_spe_pe_t, "spe pe t -> 1 1 t 1 pe spe").astype(np.float32)


def sorted_read_then_gather(x, axis, idx, fixed_slices=None):
    # h5py allows only one fancy-index array per __getitem__ call.
    # Apply fixed-axis indexing first, then temporal gather in a second step.
    if fixed_slices:
        for ax, sl in sorted(fixed_slices.items()):
            key_fixed = [slice(None)] * len(x.shape)
            if isinstance(sl, (int, np.integer)):
                sl = slice(int(sl), int(sl) + 1)
            key_fixed[ax] = sl
            x = x[tuple(key_fixed)]

    idx = np.asarray(idx, dtype=np.int64)
    uniq, inv = np.unique(idx, return_inverse=True)
    ndim = len(x.shape)
    key = [slice(None)] * ndim
    key[axis] = uniq
    out_uniq = x[tuple(key)]
    out = np.take(out_uniq, inv, axis=axis)
    return out


class CMRx4DFlowDataSet(Dataset):
    def __init__(self, **kwargs):
        options = DEFAULT_OPTS.copy()
        options.update(kwargs)
        self.options = options

        self.usrate_list = [10, 20, 30, 40, 50]
        self.D_size = int(options["D_size"])
        self.T_size = int(options["T_size"])
        self.V_size = int(options.get("V_size", 1))
        self.input = options.get("input", None)
        self.loss = options["loss"]
        self.network = options.get("network", "")
        self.in_base_dir = options.get("in_base_dir", None)
        self.out_base_dir = options.get("out_base_dir", None)
        self.val_on_the_fly_mask = bool(options.get("val_on_the_fly_mask", False))
        self.val_mask_seed = int(options.get("val_mask_seed", 12345))
        self.sdum_debug_shapes = bool(options.get("sdum_debug_shapes", False))
        self.sdum_debug_max_prints = int(options.get("sdum_debug_max_prints", 2))
        self._sdum_debug_print_count = 0
        self.test_skip_gt_precompute = bool(options.get("test_skip_gt_precompute", False))
        self.test_cache_case_assets = bool(options.get("test_cache_case_assets", False))
        self.test_gpu_preprocess_adjoint = bool(options.get("test_gpu_preprocess_adjoint", False))
        self.test_gpu_preprocess_fe_ifft = bool(options.get("test_gpu_preprocess_fe_ifft", False))
        self.profile_preprocess = bool(options.get("profile_preprocess", False))
        self._test_case_cache_key = None
        self._test_case_cache = None
        mode = options["mode"]
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be one of ['train','val','test'], got {mode}")

        if mode in ("val", "test"):
            self.D_size = -1

        if self.loss not in ("ssdu", "supervised"):
            raise ValueError("loss must be either 'ssdu' or 'supervised'")

        self.test_usrate = options.get("usrate", None)
        if mode == "test":
            if self.test_usrate is None:
                raise ValueError("In test mode you must pass --usrate")
            if isinstance(self.test_usrate, int):
                self.test_usrate = [int(self.test_usrate)]
            else:
                self.test_usrate = [int(u) for u in self.test_usrate]

        self.filename = []

        def add_case(case_dir):
            case_dir = str(case_dir)
            out_dir = _compute_out_dir(case_dir, self.in_base_dir, self.out_base_dir)
            if mode == "test":
                Nv = 4
                for u in self.test_usrate:
                    mask_ok = (Path(case_dir) / f"usmask_ktGaussian{int(u)}.mat").is_file()
                    k_ok = (Path(case_dir) / f"kdata_ktGaussian{int(u)}.mat").is_file()
                    if not (mask_ok and k_ok):
                        continue
                    for seg_i in range(Nv):
                        self.filename.append([case_dir, 0, int(u), int(seg_i), out_dir])
                return

            kdata = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
            # Challenge order: (Nv, Nt, Nc, SPE, PE, FE)
            # Keep FlowVN slicing convention for all networks.
            Nx = int(kdata.shape[-1])

            slice_starts = [0] if mode == "val" else (
                list(range(0, Nx - self.D_size + 1)) if self.D_size != -1 else [0]
            )

            for i in slice_starts:
                if mode == "train":
                    self.filename.append([case_dir, int(i), None, None, out_dir])
                else:
                    Nv = 4
                    for u in self.usrate_list:
                        for seg_i in range(Nv):
                            self.filename.append([case_dir, int(i), int(u), int(seg_i), out_dir])

        if self.input is not None and str(self.input) != "":
            case_dir = Path(self.input)
            if not case_dir.exists():
                raise FileNotFoundError(f"Input path does not exist: {case_dir}")
            add_case(str(case_dir))
        else:
            split_json = options.get(f"{mode}_split_json", None)
            split_base = options.get("split_base_dir", None) or options.get("in_base_dir", None)
            if split_json is not None and str(split_json) != "":
                subjects = load_split_cases(split_json, split_base)
            else:
                roots = options.get(f"{mode}_roots", [])
                if mode == "test":
                    # Validation/test sets may not include kdata_full.mat, so anchor on ktGaussian files.
                    req_files = ("coilmap.mat", "segmask.mat", "params.csv")
                    subjects = find_valid_cases(roots, required_files=req_files, anchor="kdata_ktGaussian*.mat")
                else:
                    subjects = find_valid_cases(roots)
            for patient_dir in subjects:
                add_case(patient_dir)

        if mode != "train":
            self.filename.sort(key=lambda x: (str(x[0]), int(x[2]), int(x[1]), int(x[3])))

    def group_fn_from_filename(dataset, idx):
        case_dir, slice_start, usrate, seg_idx, out_dir = dataset.filename[idx]
        return (str(case_dir), str(out_dir), int(usrate), int(slice_start))

    def __len__(self):
        return len(self.filename)

    def _load_test_case_assets(self, case_dir, stored_usrate, slice_start, slice_end, Nt, SPE, PE, FE, profile):
        cache_key = (
            str(case_dir),
            int(stored_usrate),
            int(slice_start),
            -1 if slice_end is None else int(slice_end),
            int(Nt),
            int(SPE),
            int(PE),
            int(FE),
            str(self.network),
        )
        if self.test_cache_case_assets and self._test_case_cache_key == cache_key and self._test_case_cache is not None:
            if profile is not None:
                profile["case_asset_cache_hit"] = True
            return dict(self._test_case_cache)

        start = time.perf_counter()
        c_raw = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")
        c_raw = c_raw[..., slice_start:slice_end].astype("complex64")
        _add_profile_ms(profile, "load_coilmap_ms", start)

        start = time.perf_counter()
        s_raw = load_mat(str(Path(case_dir) / "segmask.mat"), "segmask")
        s_raw = s_raw[..., slice_start:slice_end]
        _add_profile_ms(profile, "load_segmask_ms", start)

        start = time.perf_counter()
        params = read_params_csv(str(Path(case_dir) / "params.csv"))
        _add_profile_ms(profile, "read_params_csv_ms", start)

        start = time.perf_counter()
        mask = load_usmask_ktGaussian(case_dir, usrate=int(stored_usrate), Nt=Nt, SPE=SPE, PE=PE)
        _add_profile_ms(profile, "load_mask_ms", start)

        start = time.perf_counter()
        c = rearrange(c_raw, "nc spe pe fe -> nc fe pe spe")
        if self.network == "SDUM":
            c_denom = np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True))
            c = c / np.maximum(c_denom, 1e-8)
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            c = c / np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True))
        s = rearrange(s_raw, "spe pe fe -> fe pe spe").astype(bool)
        _add_profile_ms(profile, "mask_rearrange_normalize_ms", start)

        cached = {
            "coil_sens": c,
            "segmentation": s,
            "mask": mask,
            "usrate_true": np.float32(1.0 / np.mean(mask)),
            "VENC": np.array(params["VENC"]),
        }
        if self.test_cache_case_assets:
            self._test_case_cache_key = cache_key
            self._test_case_cache = cached

        returned = dict(cached)
        returned["_coil_sens_raw_for_gt"] = c_raw
        return returned

    def __getitem__(self, idx):
        mode = self.options["mode"]
        profile = _new_preprocess_profile(self.profile_preprocess)
        if mode != "train":
            np.random.seed(0)
            random.seed(0)

        case_dir = str(self.filename[idx][0])
        slice_start = int(self.filename[idx][1])
        stored_usrate = self.filename[idx][2]
        stored_seg_idx = self.filename[idx][3]
        out_dir = str(self.filename[idx][4])
        c = None
        s = None
        params = None
        use_test_asset_cache = mode == "test" and self.test_cache_case_assets
        defer_test_adjoint = mode in ("test", "val") and self.network == "FlowVN" and self.test_gpu_preprocess_adjoint
        defer_test_fe_ifft = defer_test_adjoint and self.test_gpu_preprocess_fe_ifft

        if not use_test_asset_cache:
            start = time.perf_counter()
            c = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")
            _add_profile_ms(profile, "load_coilmap_ms", start)

            start = time.perf_counter()
            s = load_mat(str(Path(case_dir) / "segmask.mat"), "segmask")
            _add_profile_ms(profile, "load_segmask_ms", start)

            start = time.perf_counter()
            params = read_params_csv(str(Path(case_dir) / "params.csv"))
            _add_profile_ms(profile, "read_params_csv_ms", start)

        if stored_seg_idx is None:
            if mode == "train":
                f_tmp = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
                Nv_tmp = int(f_tmp.shape[0])
                seg_idx = np.random.randint(Nv_tmp)
                del f_tmp
            else:
                seg_idx = 0
        else:
            seg_idx = int(stored_seg_idx)

        slice_end = None if self.D_size == -1 else slice_start + self.D_size

        start = time.perf_counter()
        if mode == "test":
            usrate = int(stored_usrate)
            kpath = Path(case_dir) / f"kdata_ktGaussian{usrate}.mat"
            if not kpath.is_file():
                raise FileNotFoundError(f"K-space not found: {kpath}")
            f = load_mat(str(kpath), "kdata_ktGaussian")
        else:
            f = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
            usrate = int(stored_usrate) if stored_usrate is not None else -1

        if self.network == "SDUM" and self.sdum_debug_shapes and self._sdum_debug_print_count < self.sdum_debug_max_prints:
            c_shape = "cached" if c is None else tuple(c.shape)
            s_shape = "cached" if s is None else tuple(s.shape)
            print(
                f"[SDUM DATA DEBUG][raw] case={case_dir} f_raw={tuple(f.shape)} "
                f"coilmap_raw={c_shape} segmask_raw={s_shape} slice_start={slice_start} D_size={self.D_size}"
            )

        Nv, Nt, Nc, SPE, PE, FE = f.shape

        seg_idx = int(np.clip(seg_idx, 0, Nv - 1))

        if self.T_size == -1:
            cardiac_bins = list(range(Nt))
        else:
            if mode == "train":
                first_bin = random.randint(-self.T_size + 1, Nt - self.T_size)
                cardiac_bins = list(range(first_bin, first_bin + self.T_size))
            elif mode == "val" and self.network == "SDUM":
                # SDUM model expects fixed temporal length; keep validation window length equal to T_size.
                cardiac_bins = list(range(self.T_size))
            else:
                cardiac_bins = list(range(Nt))

        bins = np.mod(cardiac_bins, Nt).astype(np.int64)

        f = sorted_read_then_gather(
            f, axis=1, idx=bins, fixed_slices={0: slice(seg_idx, seg_idx + 1)}
        )
        _add_profile_ms(profile, "load_kdata_ms", start)

        # FE is fully sampled; benchmark mode can defer this IFFT to the GPU finalizer.
        f_for_gt = None
        if defer_test_fe_ifft and mode == "val":
            start = time.perf_counter()
            f_for_gt = k2i_numpy(f, ax=[-1])
            _add_profile_ms(profile, "fe_ifft_ms", start)
        if defer_test_fe_ifft and slice_end is not None:
            raise RuntimeError("test_gpu_preprocess_fe_ifft only supports full-depth FlowVN test items")
        if not defer_test_fe_ifft:
            start = time.perf_counter()
            f = k2i_numpy(f, ax=[-1])
            _add_profile_ms(profile, "fe_ifft_ms", start)

        f = f[..., slice_start:slice_end]
        if f_for_gt is not None:
            f_for_gt = f_for_gt[..., slice_start:slice_end]
        Nv, Nt, Nc, SPE, PE, FE = f.shape

        prepared_test_assets = None
        c_for_gt = None
        if use_test_asset_cache:
            prepared_test_assets = self._load_test_case_assets(
                case_dir=case_dir,
                stored_usrate=stored_usrate,
                slice_start=slice_start,
                slice_end=slice_end,
                Nt=Nt,
                SPE=SPE,
                PE=PE,
                FE=FE,
                profile=profile,
            )
            c = prepared_test_assets["coil_sens"]
            s = prepared_test_assets["segmentation"]
            mask = prepared_test_assets["mask"]
            usrate_true = prepared_test_assets["usrate_true"]
            VENC = prepared_test_assets["VENC"]
            c_for_gt = prepared_test_assets.pop("_coil_sens_raw_for_gt", None)
        else:
            start = time.perf_counter()
            c = c[..., slice_start:slice_end].astype("complex64")
            _add_profile_ms(profile, "load_coilmap_ms", start)
            start = time.perf_counter()
            s = s[..., slice_start:slice_end]
            _add_profile_ms(profile, "load_segmask_ms", start)
            VENC = np.array(params["VENC"])

        skip_gt = (mode == "test" and self.test_skip_gt_precompute) or (mode == "test" and defer_test_adjoint)
        im = None
        if skip_gt:
            if profile is not None:
                profile["gt_skipped"] = True
        else:
            if prepared_test_assets is not None:
                if c_for_gt is None:
                    start = time.perf_counter()
                    c_for_gt = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")
                    c_for_gt = c_for_gt[..., slice_start:slice_end].astype("complex64")
                    _add_profile_ms(profile, "load_coilmap_ms", start)
                gt_coil = c_for_gt
            else:
                gt_coil = c
            start = time.perf_counter()
            gt_source = f if f_for_gt is None else f_for_gt
            im = np.sum(k2i_numpy(gt_source, ax=[-2, -3]) * np.conj(gt_coil), axis=-4)
            _add_profile_ms(profile, "gt_image_ms", start)

        if mode == "train":
            start = time.perf_counter()
            usrate = random.choice(self.usrate_list)
            total_points = (PE * SPE) // usrate
            masks_spe_pe_t = fun_mask_gen_2d(
                mask_size=(PE, SPE),
                center_radius_x=0.5,
                center_radius_y=0.5,
                total_points=total_points,
                pattern_num=Nt,
                sigma_x=PE / 5,
                sigma_y=SPE / 5,
                min_dist_factor=3,
                rep_decay_factor=0.5,
            )
            mask = rearrange(masks_spe_pe_t, "spe pe t -> 1 1 t 1 pe spe").astype(np.float32)
            _add_profile_ms(profile, "load_mask_ms", start)
        elif mode == "val":
            start = time.perf_counter()
            if self.val_on_the_fly_mask:
                val_seed = _stable_u32_seed(
                    self.val_mask_seed,
                    case_dir,
                    int(slice_start),
                    int(seg_idx),
                    int(stored_usrate),
                    int(Nt),
                    int(SPE),
                    int(PE),
                )
                mask = generate_ktgaussian_mask_deterministic(
                    usrate=int(stored_usrate),
                    Nt=Nt,
                    SPE=SPE,
                    PE=PE,
                    seed=val_seed,
                )
            else:
                mask = load_usmask_ktGaussian(case_dir, usrate=int(stored_usrate), Nt=Nt, SPE=SPE, PE=PE)
            _add_profile_ms(profile, "load_mask_ms", start)
        elif not use_test_asset_cache:
            start = time.perf_counter()
            mask = load_usmask_ktGaussian(case_dir, usrate=int(stored_usrate), Nt=Nt, SPE=SPE, PE=PE)
            _add_profile_ms(profile, "load_mask_ms", start)

        start = time.perf_counter()
        f = rearrange(f, "nv nt nc spe pe fe -> nv nc nt fe pe spe")
        if prepared_test_assets is None:
            c = rearrange(c, "nc spe pe fe -> nc fe pe spe")
            if self.network == "SDUM":
                c_denom = np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True))
                c = c / np.maximum(c_denom, 1e-8)
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                c = c / np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True))
            s = rearrange(s, "spe pe fe -> fe pe spe")

        f *= mask
        if im is not None:
            im = rearrange(im, "nv nt spe pe fe -> nv nt fe pe spe")
        _add_profile_ms(profile, "mask_rearrange_normalize_ms", start)

        start = time.perf_counter()
        imdata_p1 = None
        if not defer_test_adjoint:
            imdata_p1 = mriAdjointOp(f, c[np.newaxis, :, np.newaxis, :, :, :], mask).astype(np.complex64)
        _add_profile_ms(profile, "mri_adjoint_ms", start)

        start = time.perf_counter()
        norm = None
        if not defer_test_adjoint:
            denom = np.linalg.norm(np.abs(f) != 0)
            norm = np.linalg.norm(f) / (denom if denom != 0 else 1.0)
            if self.network == "SDUM":
                norm = max(float(norm), 1e-8)

            imdata_p1 /= norm
            if im is not None:
                im /= norm
            f /= norm

        shape_ref = im if im is not None else (imdata_p1 if imdata_p1 is not None else f[:, :, 0, :, :, :])
        _, _, depth_dim, pe_dim, fe_dim = shape_ref.shape

        if self.network == "SDUM" and self.sdum_debug_shapes and self._sdum_debug_print_count < self.sdum_debug_max_prints:
            im_shape = None if im is None else tuple(im.shape)
            print(
                f"[SDUM DATA DEBUG][proc] f={tuple(f.shape)} im={im_shape} imdata_p1={tuple(imdata_p1.shape)} "
                f"coil_sens={tuple(c.shape)} depth={depth_dim} PE={pe_dim} FE={fe_dim} seg_idx={seg_idx} usrate={usrate}"
            )
            self._sdum_debug_print_count += 1

        out = {
            "coil_sens": c,
            "segmentation": s.astype(bool),
            "case_dir": case_dir,
            "subj": Path(case_dir).name,
            "slice_start": int(slice_start),
            "seg_idx": int(seg_idx),
            "usrate": int(usrate),
            "usrate_true": (1 / np.mean(mask)).astype("float32"),
            "bins": bins.astype(np.int64),
            "Nt": int(Nt),
            "SPE": int(SPE),
            "PE": int(PE),
            "FE": int(FE),
            "VENC": VENC,
            "out_dir": out_dir,
        }
        if defer_test_adjoint:
            out["kdata_p1_unnorm"] = f
            out["mask"] = mask
            out["deferred_gpu_preprocess_adjoint"] = np.array(True)
            out["deferred_gpu_preprocess_fe_ifft"] = np.array(bool(defer_test_fe_ifft))
        else:
            out["imdata_p1"] = imdata_p1
            out["kdata_p1"] = f
            out["norm"] = norm
        if im is not None:
            out["gt"] = im
        if profile is not None:
            out["preprocess_profile"] = profile
        _add_profile_ms(profile, "final_cast_packaging_ms", start)
        return out
