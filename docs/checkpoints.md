# Checkpoints

Weights are not distributed with this release candidate. Training requires an
authorized S16 teacher checkpoint; inference requires an authorized checkpoint
of the configured depth. Contact the authors about availability after release
permissions are resolved. No download URL is implied by a filename below.

| Local filename | Purpose |
| --- | --- |
| `checkpoints/s16_teacher.ckpt` | S16 teacher and initialization for every matched S8 arm |
| `checkpoints/s8_full_seed12345.ckpt` | Example S8 inference/evaluation checkpoint |

For the paper's S16 teacher, the recorded SHA-256 is:

```text
22da52a5ba2a06d9ab04538654a4004c1ad3a090e53ee244f973a22e3dc51ab8
```

S8 copies teacher stages 2, 4, ..., 16 (zero-based indices 1, 3, ..., 15).
The supervised control uses exactly the same copied initialization. A different
teacher creates a new experiment, even when its architecture matches.

The primary student checkpoint is the end of epoch 10 (zero-based epoch 9),
for both seeds. Do not select each arm's best validation checkpoint. Student
checkpoints exclude teacher parameters. The loader handles the compiler's
`_orig_mod` key wrappers without changing tensor values.

Use `--resume_from_checkpoint` to resume optimizer, scheduler, epoch, and step
state. `--ckpt_path` is for evaluation or weight loading and is not a substitute
for a full training resume. Only load checkpoints from a trusted source.

When weights are cleared for distribution, record a stable URL, SHA-256,
architecture, seed, source config, epoch, teacher lineage, and permitted use for
each file. Keep them outside Git history, for example as versioned release
assets. Until then, the synthetic model check needs no weights.
