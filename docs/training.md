# Training recipes

All recipes are YAML files under `configs/`. `clarity-train` accepts a config
and lets explicit CLI flags override it.

## CLARiTy-S-16 at 224 px

`configs/clarity_s16_224.yaml` is the 224 px recipe:

- DINO initialisation, 224 px input, 200 epochs;
- batch size 64 with accumulation factor 16;
- AdamW, learning rate `1e-4`, weight decay `0.05`, warmup 5;
- dropout `0.1` for attention pooling, MLP, self-attention and stochastic
  depth;
- the paper's combined loss with weights `alpha_C`, `alpha_P`, `alpha_f`,
  `beta` and `gamma`;
- `classify-70` split, attention-layer sweep 1-12; the paper selects `p=5`;
- OneBox with `joint_argmax` during training.

Run:

```bash
uv run clarity-train --config configs/clarity_s16_224.yaml
```

## CLARiTy-S-16 at 512 px

`configs/clarity_s16_512.yaml` is the 512 recipe. It is warm-started from the
224 model (`weights_img_size: 224`), uses 512 px, batch size 16 with
accumulation 64, and the same combined-loss configuration. Set `model.weights` to your
224 px checkpoint or pass `-wg <checkpoint>`. The paper
selects `p=3` for the 512 model.

## ConvNeXtV2-B teachers

`configs/teacher_convnextv2b_224.yaml` and `_512.yaml` train the
classification-only teacher with WBCE:

- 100 epochs, batch size 8 with accumulation factor 128;
- AdamW, learning rate `2.5e-5`, weight decay `0.05`, warmup 3;
- ImageNet initialisation;
- no localization validation and no attention sweep.

Because the teacher is classification-only, pass `-vl false` (and leave the
localization loader unset) when running these configs:

```bash
uv run clarity-train --config configs/teacher_convnextv2b_224.yaml -vl false
```

## Attention-layer sweep

CLARiTy can evaluate every attention-layer count after each epoch. Enable it
with `-las 1 2 3 4 5 6 7 8 9 10 11 12` and choose the primary count with
`-lasp 5` (224) or `-lasp 3` (512). Sweep selection and the
`best-selection.json` manifest are handled by the training loop. The sweep is
opt-in; a plain run uses the model's `n_attn_layers_avg` (the paper's `p`).

## Loss and model options

The public loss registry contains only the losses used in the paper:
`combined`, `wbce`, `bas`, `dice` and `mlsm` (the classification-loss
ablation). The combined loss uses the paper's weight symbols:

```text
L = alpha_C * L_CLS^C + alpha_P * L_CLS^P + alpha_f * L_CLS^f
    + beta * L_BAS + gamma * L_Dice
```

Relevant model/loss knobs include:

- `alpha_C`, `alpha_P`, `alpha_f`, `beta`, `gamma`;
- `cls_loss` (`wbce` in the reported model, `mlsm` in the ablation);
- `use_segment_cam`;
- `class_token_pool`;
- `n_attn_layers_avg` (the paper's `p`).


## Run outputs

A run folder contains `model_checkpoint.pth`, `model_best.pth`,
`hyperparams.json` and `log.pk.xz` performance curves. See
`docs/artifacts.md` for the full layout.
