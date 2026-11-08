# Configs

| File | Resolution | Purpose |
|---|---|---|
| `clarity_s16_224.yaml` | 224 px | CLARiTy-S-16 at 224 px, DINO init, 200 epochs, paper `p=5` |
| `clarity_s16_512.yaml` | 512 px | CLARiTy-S-16 at 512 px, warm-started from the 224 checkpoint, paper `p=3` |
| `teacher_convnextv2b_224.yaml` | teacher 224 | ConvNeXtV2-B classification teacher, 224 px |
| `teacher_convnextv2b_512.yaml` | teacher 512 | ConvNeXtV2-B classification teacher, 512 px |

The teacher configs are classification-only. Run them with `-vl false`; they
have no attention sweep and no localization validation.

Paths in the YAML are placeholders or relative paths. Prefer passing
`--data-root`, `--models-root`, `--weights` and a `--splits-dir` manifest on
the command line; explicit CLI values override the config.
