# Third-party notices

The root [MIT license](LICENSE) applies to the project's own contributions.
Third-party code retains its upstream terms and is excluded from any blanket
MIT grant. Original attribution comments and available license texts are
preserved below.

| Component | Source and terms | Included notice |
| --- | --- | --- |
| FlowVN demo lineage, loader, MRI utilities | [Organizer FlowVN subtree](https://github.com/CmrxRecon/CMRx4DFlow2026/tree/f6f835f34b86464256e3ce4362e7831325f32590/CMRx4DFlowReconDemo/FlowVN), adapted from FlowMRI-Net; MIT, © 2024 ljacobs | [MIT](licenses/FlowMRI-Net-MIT.txt) |
| Variational-network layers | [rixez/pytorch_mri_variationalnetwork](https://github.com/rixez/pytorch_mri_variationalnetwork); MIT, © 2020 rixez | [MIT](licenses/rixez-MIT.txt) |
| Kernel constraint/projection | [VLOGroup/mri-variationalnetwork](https://github.com/VLOGroup/mri-variationalnetwork); MIT, © 2018 Vision, Learning and Optimization Group | [MIT](licenses/VLOGroup-MIT.txt) |
| Option printing in `utils/misc_utils.py` | [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2a7afba2895d52556dd5dfe07e8555ef657ced6f/options/base_options.py); BSD-style terms | [Complete upstream license](licenses/CycleGAN-BSD.txt) |
| SSIM implementation | [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) by Po-Hsun (Evan) Su and [pytorch-ssim-3D](https://github.com/jinh0park/pytorch-ssim-3D) | Both upstream [2D](licenses/pytorch-ssim.txt) and [3D](licenses/pytorch-ssim-3D.txt) license files contain only `MIT`; no copyright line is supplied there. |
| Background phase correction in `utils/utils_bgc.py` | [PCMRI-MSAC](https://github.com/lolacaro/PCMRI-MSAC/blob/f1157894979b47d0d0aad9e2965278b19400fc9a/README.md#licence), credited to Lola Caro; upstream states `CC-BY-NC` without a version | This Python adaptation retains the stated noncommercial terms. No specific Creative Commons version is inferred, and this module is not relabeled MIT. |
| Shared data, flow, metric, and SSIM utilities | [Organizer shared Utils](https://github.com/CmrxRecon/CMRx4DFlow2026/tree/f6f835f34b86464256e3ce4362e7831325f32590/CMRx4DFlowReconDemo/Utils) | No separate software-license grant was found for this sibling directory in the inspected revision. This project does not relicense those upstream portions. |

The shared-utility scope concerns `utils/utils_datasl.py`,
`utils/utils_flow.py`, `utils/utils_metrics.py`, and organizer-specific changes
to `utils/pytorch_ssim.py`. Local changes add data-loading support, efficiency
options, stage distillation, streaming evaluation, and VENC-scaled metrics;
original attribution comments are retained.

The SSDU-derived `partitioning.py` is excluded. Its import in the loader was
unused and was removed; the matched supervised/KD workflow does not need it.
The organizer mask archive is also excluded and must be supplied separately
under its applicable terms.

These upstream declarations were checked on 19 September 2026. Attribution
does not supply a missing upstream permission; consult the respective owners
for uses that require clarification of these terms. No license in this
directory authorizes redistribution of challenge data, trained weights, the
accepted manuscript, or participant-specific results.
