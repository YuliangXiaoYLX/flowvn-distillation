# Data and tensor conventions

Obtain data through the [CMRx4DFlow organizers](https://cmrx.chihucloud.com/2026/).
This repository supplies no challenge data, participant download links, actual
split files, or reconstructions. Use only data you are authorized to access.

## Labeled training and validation

Each case directory contains:

| File | Dataset / content | Shape read by Python |
| --- | --- | --- |
| `kdata_full.mat` | `kdata_full`, complex k-space | `(Nv, Nt, Nc, SPE, PE, FE)` |
| `coilmap.mat` | `coilmap`, complex sensitivities | `(Nc, SPE, PE, FE)` |
| `segmask.mat` | `segmask`, reconstruction region | `(SPE, PE, FE)` |
| `params.csv` | Acquisition metadata | One row |

The MAT reader expects HDF5 / MATLAB v7.3, including compound `real` / `imag`
datasets. Shapes above describe the arrays returned by h5py, not MATLAB's
display order. `Nv=4` comprises one reference and three flow encodings. `Nt`
counts cardiac phases and `Nc` counts receiver coils; `FE` is the fully sampled
readout direction.

Use this layout, replacing the fictional case names with your authorized local
identifiers:

```text
data/
  TrainSet/Aorta/
    site_a/scanner_a/case_a/
      kdata_full.mat
      coilmap.mat
      segmask.mat
      params.csv
  splits/
    train.json
    val.json
```

Split files are JSON lists of case-directory paths relative to `split_base_dir`:

```json
["site_a/scanner_a/case_a", "site_b/scanner_b/case_b"]
```

Patients must not overlap between training and validation. The paper used 16
held-out cases; those exact split contents are not included. The loader checks
that cases and required files exist; study owners remain responsible for
patient-level split independence.

`params.csv` uses semicolon-separated values for vector fields. In particular,
`VENC` must contain the three encoding-specific velocity limits in **cm/s**,
in the same order as the three flow encodings. Do not substitute another unit
or infer direction order from filenames. The loader also reads `resolution`,
`FOV`, `matrix_size`, `spatial_order`, and `venc_order` when present.

## Inference on stored undersampled data

For each requested acceleration `R`, supply `kdata_ktGaussianR.mat` and
`usmask_ktGaussianR.mat`, with dataset names `kdata_ktGaussian` and
`usmask_ktGaussian`. The mask shape is `(1, Nt, 1, SPE, PE, 1)`. Supply the same
coilmap, segmentation, and parameter files. Full k-space is not needed for
`mode: test`.

Update `test_roots`, `in_base_dir`, and `out_base_dir` in `configs/inference.yaml`.
The output keeps paths relative to `in_base_dir`. Missing acceleration files
are skipped by the original loader, so check that the number of outputs equals
your requested case/acceleration combinations.

The output `img_ktGaussianR.npz` is a sparse COO representation of the complex,
segmentation-masked image, in `(Nv, Nt, SPE, PE, FE)` order. Load it with
`utils.utils_datasl.load_coo_npz(path, as_dense=True)`.

## Masks and relocation

Paper training and on-the-fly validation use the organizer mask package. Put
your authorized copy at `external/CMRx4DFlowMaskGeneration.zip`; select it with
the two environment variables shown in the README. The adapter verifies both
the archive and its inner source before importing it. The local generator is
available for development and is not interchangeable with the paper backend.

The original deterministic validation seed includes the **case-directory
string**, acceleration, encoding, slice, and shape. Moving a dataset changes
generated validation masks even if `val_mask_seed` stays `12345`. For exact
mask replay, preserve the original seed inputs or obtain an authorized fixed
mask/seed mapping. Setting `val_on_the_fly_mask: false` reads stored masks, but
ordinary organizer masks are not automatically the same as the masks used in
the paper. This release does not bundle that case-specific mapping.

## Model and metric conventions

Model images use `(B, Nv, Nt, FE, PE, SPE)` complex tensors. K-space uses
`(B, Nv, Nc, Nt, FE, PE, SPE)`, and sensitivities use `(B, Nc, FE, PE, SPE)`.
The loader applies the readout inverse FFT, coil normalization, sampling mask,
and adjoint normalization. Preserve these steps when adapting another dataset.

Flow phase is relative to the reference encoding. Physical velocity uses
`angle(x_flow * conj(x_reference)) * VENC / pi`; error wraps phase differences
on the circle before conversion. Grouped evaluation combines all four
encodings before computing velocity metrics and releases each completed group.
Background phase correction is retained, with its separate license noted in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Use deidentified case-directory names. Generated metric IDs retain the case
basename plus a hash; this is not a general anonymization mechanism. Keep
generated metadata and all patient-derived outputs private.
