# Third-party code and redistribution status

This candidate preserves the implementation used for the paper. It is being
prepared privately while the authors clarify the permissions below. A license
for original contributions has not been selected, and the whole tree is not
being offered under a blanket MIT license.

| Component | Source and terms | Included notice |
| --- | --- | --- |
| FlowVN demo lineage, loader, MRI utilities | [Organizer FlowVN subtree](https://github.com/CmrxRecon/CMRx4DFlow2026/tree/f6f835f34b86464256e3ce4362e7831325f32590/CMRx4DFlowReconDemo/FlowVN), adapted from FlowMRI-Net; MIT, © 2024 ljacobs | [MIT](licenses/FlowMRI-Net-MIT.txt) |
| Variational-network layers | [rixez/pytorch_mri_variationalnetwork](https://github.com/rixez/pytorch_mri_variationalnetwork); MIT, © 2020 rixez | [MIT](licenses/rixez-MIT.txt) |
| Kernel constraint/projection | [VLOGroup/mri-variationalnetwork](https://github.com/VLOGroup/mri-variationalnetwork); MIT, © 2018 Vision, Learning and Optimization Group | [MIT](licenses/VLOGroup-MIT.txt) |
| Option printing in `utils/misc_utils.py` | [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2a7afba2895d52556dd5dfe07e8555ef657ced6f/options/base_options.py); BSD-style terms | [Complete upstream license](licenses/CycleGAN-BSD.txt) |
| SSIM implementation | [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) by Po-Hsun (Evan) Su and [pytorch-ssim-3D](https://github.com/jinh0park/pytorch-ssim-3D) | Both upstream [2D](licenses/pytorch-ssim.txt) and [3D](licenses/pytorch-ssim-3D.txt) license files contain only `MIT`; no copyright line is supplied there. |
| Background phase correction in `utils/utils_bgc.py` | [PCMRI-MSAC](https://github.com/lolacaro/PCMRI-MSAC#licence), credited to Lola Caro; upstream states `CC-BY-NC` without a version | The applicable version and redistribution terms need confirmation. This module is a Python adaptation; it is not relabeled MIT. |
| Shared data, flow, metric, and SSIM utilities | [Organizer shared Utils](https://github.com/CmrxRecon/CMRx4DFlow2026/tree/f6f835f34b86464256e3ce4362e7831325f32590/CMRx4DFlowReconDemo/Utils) | No explicit redistribution grant was found for this sibling directory. The nested FlowVN MIT notice does not settle its scope. |

The shared-utility clarification concerns `utils/utils_datasl.py`,
`utils/utils_flow.py`, `utils/utils_metrics.py`, and organizer-specific changes
to `utils/pytorch_ssim.py`. Local changes add data-loading support, efficiency
options, stage distillation, streaming evaluation, and VENC-scaled metrics;
original attribution comments are retained.

The SSDU-derived `partitioning.py` is excluded. Its import in the loader was
unused and was removed; the matched supervised/KD workflow does not need it.
The organizer mask archive is also excluded and must be supplied separately
under its applicable terms.

These observations were checked on 19 September 2026. The authors must resolve
the identified scope/version questions before public redistribution. No
license in this directory authorizes redistribution of challenge data, trained
weights, the accepted manuscript, or participant-specific results.
