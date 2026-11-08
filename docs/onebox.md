# OneBox localization

OneBox is the localization method used by the reported models. It is a
calibration-and-box-extraction procedure, not a learned detection head.

## Parameters

Each finding has five parameters:

- `t_lo`, `t_hi`: hysteresis thresholds on the score-normalised heatmap;
- `s`: the box scale (fraction of the gated component's span);
- `alpha`: the component score exponent;
- `sigma` (or `sigma_rel`): Gaussian smoothing before gating.

## Search grid

The validation fit evaluates the exact 0.05 score grid:

```text
(t_lo, t_hi) with t_lo < t_hi, both in {0.05, 0.10, ..., 0.95}   (171 pairs)
s in {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}
alpha in {0.9, 1.0, 1.1}
sigma in {0, 1, 2, 3, 4} (relative to the 224 px reference)
```

This is 5 x 171 x 6 x 3 = 15,390 configurations per finding. The selection metric is the
validation 5-threshold mean IoU accuracy; `joint_argmax` selects the
configuration with the best joint objective. The selected parameters are
frozen for the test split.

## Box extraction

For a given threshold pair, the heatmap is gated between `t_lo` and `t_hi`,
smoothed, and thresholded. Each 4-connected component produces one box: the
middle row span and middle column span of the component, scaled by `s`. The
component score is weighted by `alpha`. This is why the method is called
OneBox: one box per connected component, with no NMS and no per-component
regression.

## IoU and IoBB

The evaluator reports both IoU and IoBB families from the same calibration.
IoBB is reported for the paper's comparison tables but is never used to select
parameters; selection is always the IoU validation objective. The exact
method block is written into every evaluation pickle so a reader can verify
which family and thresholds produced the reported numbers.
