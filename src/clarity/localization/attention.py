"""Attention-map extraction for CLARiTy's localization evaluation.

Public map path used by OneBox evaluation and sweep fitting.  Legacy grad-CAM
and other extractors live in :mod:`clarity.localization.legacy`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn

from clarity.models.layers.upsample import resample
from clarity.utils.runtime import get_autocast_context
from clarity.localization.sweep import (
    SWEEP_CAPABILITY_ATTRIBUTE,
    has_sweep_capability,
    unwrap_model,
    variant_key,
)


def _single_count_sweep_map(
    variant_maps: th.Tensor,
    batch_size: int,
) -> th.Tensor:
    """Reduce a single-count sweep output ``[V, B, C, H, W]`` to ``[B, C, H, W]``.

    With one requested variant the ordinary (single-GPU) sweep forward returns
    ``[1, B, C, H, W]``. Under ``nn.DataParallel``, each replica returns
    ``[1, b, C, H, W]`` which gathers along dim 0 into ``[k, B/k, C, H, W]``;
    that gather is un-done here (the same re-arrangement used by
    :func:`get_model_attention_sweep`) so the variant map is never confused
    with a batch slice.
    """
    if not th.is_tensor(variant_maps) or variant_maps.dim() != 5:
        got = tuple(variant_maps.shape) if th.is_tensor(variant_maps) else type(variant_maps)
        raise RuntimeError(
            f"expected a single-count sweep output of shape [1, B, C, H, W], " f"got {got}"
        )
    V, B, C, h_map, w_map = variant_maps.shape
    if V == 1:
        if B != batch_size:
            raise RuntimeError(
                f"expected B={batch_size} variant map rows (matching the "
                f"input batch), got B={B}"
            )
        return variant_maps[0]
    # ``nn.DataParallel`` gather of V=1 outputs: [k, B/k, C, H, W]
    k = V
    if batch_size % k != 0 or B != batch_size // k:
        raise RuntimeError(
            f"expected a single-count sweep output [1, {batch_size}, C, H, W] "
            f"(or a DataParallel gather thereof), got {tuple(variant_maps.shape)}"
        )
    return (
        variant_maps.reshape(k, 1, B, C, h_map, w_map)
        .permute(1, 0, 2, 3, 4, 5)
        .reshape(1, batch_size, C, h_map, w_map)[0]
    )


def get_model_attention(
    model: nn.Module,
    images: th.Tensor,
    labels: th.Tensor,
    filenames: tuple[str],
    inference_attention_type: str = "attention",
    attention_layer_count: int | None = None,
    interpolation_mode: str = "bilinear",
    use_amp: bool = False,
    n_attn_layers_avg: int | None = None,
    **kwargs,
) -> dict[str, dict[int, np.ndarray]]:
    """Extract gradient-free model-attention localization maps.

    The historical path uses the model's configured
    ``n_attn_layers_avg``/``inference_attention_type`` attributes (temporarily
    overridden and restored around the forward). The call-level API
    instead selects a *single* attention-layer count for this call via
    ``attention_layer_count`` (or the equivalent ``n_attn_layers_avg``
    localization kwarg): the model is invoked through its sweep forward
    with a one-element count tuple, so ``model.n_attn_layers_avg`` and
    ``model.inference_attention_type`` are never read or assigned. The
    call-level path requires a sweep-capable model (see
    ``localization.sweep.SWEEP_CAPABILITY_ATTRIBUTE``) and fails
    clearly otherwise; an explicit ``attention_layer_count`` on an unsupported
    model is an error, while a passed-through ``n_attn_layers_avg`` kwarg
    keeps its historical (ignored) behavior for non-sweep models.
    """
    filenames = np.array(filenames, dtype=str)
    cam_dict = defaultdict(dict)
    _B, _C, H, W = images.shape

    ## Resolve the call-level attention-layer count 
    # ``attention_layer_count`` is the explicit call-level API; the canonical
    # ``n_attn_layers_avg`` localization kwarg is accepted as an equivalent
    # alias. A conflicting pair is rejected; the alias is only honoured for
    # sweep-capable models so historical non-CLARiTy invocations that passed
    # it through free-form kwargs keep their previous (ignored) behavior.
    if (
        attention_layer_count is not None
        and n_attn_layers_avg is not None
        and attention_layer_count != n_attn_layers_avg
    ):
        raise ValueError(
            f"conflicting attention-layer counts: attention_layer_count="
            f"{attention_layer_count!r} vs n_attn_layers_avg="
            f"{n_attn_layers_avg!r}"
        )
    if (
        attention_layer_count is None
        and n_attn_layers_avg is not None
        and has_sweep_capability(model)
    ):
        attention_layer_count = n_attn_layers_avg

    call_level = attention_layer_count is not None
    if call_level:
        unwrapped_for_check = unwrap_model(model)
        if not has_sweep_capability(model):
            fullname = getattr(unwrapped_for_check, "fullname", None)
            raise ValueError(
                "an explicit attention_layer_count requires a model that "
                "supports the multi-variant localization attention sweep; "
                f"model {type(model).__name__} (inner type "
                f"{type(unwrapped_for_check).__name__}, fullname={fullname!r}) "
                f"does not advertise the capability via a truthy "
                f"{SWEEP_CAPABILITY_ATTRIBUTE!r} class attribute"
            )
        if isinstance(attention_layer_count, bool) or not isinstance(attention_layer_count, int):
            raise TypeError(
                "attention_layer_count must be an int, got "
                f"{type(attention_layer_count).__name__}"
            )
        blocks = getattr(unwrapped_for_check, "blocks", None)
        if blocks is not None:
            depth = len(blocks)
            if attention_layer_count < 1 or attention_layer_count > depth:
                raise ValueError(
                    f"attention layer count {attention_layer_count} is "
                    f"outside the valid range [1, {depth}]"
                )

    if isinstance(model, nn.DataParallel):
        inner = model.module
    else:
        inner = model

    # set model inference mode. The call-level path never touches
    # ``inference_attention_type`` (the sweep forward takes a single-call
    # override) and never assigns ``n_attn_layers_avg``; ``only_classify`` is
    # temporarily disabled exactly like the historical path.
    prev_only_classify = inner.only_classify
    prev_inference_attention_type = None
    if not call_level:
        prev_inference_attention_type = inner.inference_attention_type
        inner.inference_attention_type = inference_attention_type
    inner.only_classify = False

    try:
        # get model predictions
        model.eval()
        with th.no_grad(), get_autocast_context(images.device, enabled=use_amp):
            if call_level:
                logits, variant_maps = model(
                    images,
                    localization_layer_counts=(attention_layer_count,),
                    inference_attention_type_override=inference_attention_type,
                )
                attention = _single_count_sweep_map(variant_maps, _B)
            else:
                logits, attention = model(images)
            probas = th.sigmoid(logits)
            preds = (probas > 0.5).float()
    finally:
        # reset model inference mode (even when the forward raised)
        if not call_level:
            inner.inference_attention_type = prev_inference_attention_type
        inner.only_classify = prev_only_classify

    # Keep interpolation/post-processing in float32 after the autocast model
    # forward; this also avoids unsupported CPU bfloat16 interpolation paths.
    attention = attention.float()

    # upsample attention to match image (skipped when already at image
    # resolution, e.g. image-res maps from SegmentCAM's fused upsampling head)
    if attention.shape[-2:] != (H, W):
        attention = resample(x=attention, size=(H, W), mode=interpolation_mode)

    # Detach torch vars and send to CPU as NumPy. NumPy does not support
    # bfloat16 directly, so cast autocast outputs back to float32 first.
    probas = probas.detach().float().cpu().numpy()
    preds = preds.detach().float().cpu().numpy()
    labels = labels.detach().float().cpu().numpy()
    attention = attention.detach().float().cpu().numpy()

    # add predictions to output dict
    for filename, img_labels, img_probas, img_preds in zip(filenames, labels, probas, preds):
        filename_key = str(filename)
        cam_dict[filename_key]["labels"] = img_labels
        cam_dict[filename_key]["probas"] = img_probas
        cam_dict[filename_key]["preds"] = img_preds

    # add model attention for positive ground truth labels
    for filename, img_labels, img_attention in zip(filenames, labels, attention):
        filename_key = str(filename)
        for label_idx in range(img_labels.shape[0]):
            if img_labels[label_idx].item() == 1:
                cam_dict[filename_key][label_idx] = img_attention[label_idx]

    return cam_dict


def get_model_attention_sweep(
    model: nn.Module,
    images: th.Tensor,
    labels: th.Tensor,
    filenames: tuple[str],
    attention_layer_counts: tuple[int, ...],
    inference_attention_type: str = "attention",
    interpolation_mode: str = "bilinear",
    use_amp: bool = False,
) -> dict[str, dict[str, dict[int | str, np.ndarray]]]:
    """Extract localization maps for several suffix-layer counts in one pass.

    This is the sweep counterpart of :func:`get_model_attention` for models
    advertising the sweep capability (see
    ``localization.sweep.SWEEP_CAPABILITY_ATTRIBUTE``). It consumes the
    fixed-shape ``[V, B, C, H, W]`` sweep output of the CLARiTy
    multi-variant forward and produces evaluator-ready maps for every
    requested variant without mutating model state: unlike
    :func:`get_model_attention`, no ``inference_attention_type`` or
    ``only_classify`` attribute is ever (temporarily) assigned. The model must
    already be in evaluation mode.

    Exactly one model call is performed for any number of requested variants;
    labels, sigmoid probabilities and thresholded predictions are computed
    once and shared across all variant records.

    The model is always invoked through the given (possibly ``DataParallel``
    wrapped) ``model`` object so ``DataParallel`` continues to scatter/gather
    normally; the wrapper is unwrapped only for capability inspection and
    depth validation.

    Args:
        model: Sweep-capable model (e.g. ``CLARiTy``), optionally wrapped in
            ``nn.DataParallel``. Must be in evaluation mode with
            ``only_classify=False``.
        images: Input batch ``[B, C, img_H, img_W]``.
        labels: Ground-truth multi-hot labels ``[B, n_classes]``.
        filenames: Per-image filenames (normalized to ``str`` keys).
        attention_layer_counts: Tuple of suffix-layer counts (1-based, each in
            ``[1, model depth]``), in the output order.
        inference_attention_type: Map fusion type forwarded as the single-call
            ``inference_attention_type_override`` of the model's sweep forward
            (one of the model's supported types; the standard path is
            ``mask``).
        interpolation_mode: Interpolation mode used to bring variant maps that
            are not already at image resolution (e.g. patch-resolution
            attention) up to ``[img_H, img_W]``, matching
            :func:`get_model_attention`.
        use_amp: Optional bfloat16 autocast around the model forward. All
            post-processing is performed in float32 before NumPy transfer.

    Returns:
        A dict keyed by the canonical variant keys produced by
        ``localization.sweep.variant_key`` (``n_attn_layers_avg=2``),
        then by normalized filename string, holding the same per-file record
        schema as :func:`get_model_attention`::

            {
              "n_attn_layers_avg=2": {
                filename: {
                  "labels": np.ndarray [n_classes],
                  "probas": np.ndarray [n_classes],
                  "preds":  np.ndarray [n_classes],
                  <abnormality_index int>: np.ndarray [img_H, img_W],
                }
              },
              ...
            }

        The ``'labels'``/``'probas'``/``'preds'`` arrays are shared (identical
        objects) across all variants of a call; abnormality-index entries hold
        each variant's own map **as an independent copy**, not a view into the
        shared ``[V, B, C, H, W]`` batch tensor, so callers may retain maps
        across batches (e.g. the OneBox sweep's fit pool) without pinning the
        whole batch array. Maps are included only for positive ground-truth
        labels, matching :func:`get_model_attention`.
    """
    filenames = np.array(filenames, dtype=str)
    _B, _C, H, W = images.shape
    n_counts = len(attention_layer_counts)

    ## Capability and configuration checks (inspect the unwrapped module only;
    ## the wrapped model object is what actually gets called)
    unwrapped = unwrap_model(model)
    if not has_sweep_capability(model):
        fullname = getattr(unwrapped, "fullname", None)
        raise ValueError(
            f"model {type(model).__name__} (inner type "
            f"{type(unwrapped).__name__}, fullname={fullname!r}) does not "
            f"support the multi-variant localization attention sweep; it must "
            f"advertise the capability via a truthy "
            f"{SWEEP_CAPABILITY_ATTRIBUTE!r} class attribute"
        )

    if not isinstance(attention_layer_counts, tuple):
        raise TypeError(
            "attention_layer_counts must be a tuple of ints, got "
            f"{type(attention_layer_counts).__name__}"
        )
    if n_counts == 0:
        raise ValueError("attention_layer_counts must not be empty")

    # validate the count tuple against the model depth when the model exposes
    # its block sequence (the sweep forward itself re-validates regardless)
    blocks = getattr(unwrapped, "blocks", None)
    if blocks is not None:
        depth = len(blocks)
        for count in attention_layer_counts:
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("attention_layer_counts must contain only ints, got " f"{count!r}")
            if count < 1 or count > depth:
                raise ValueError(
                    f"attention layer count {count} is outside the valid " f"range [1, {depth}]"
                )

    ## One forward pass for every variant
    with th.no_grad(), get_autocast_context(images.device, enabled=use_amp):
        logits, variant_maps = model(
            images,
            localization_layer_counts=attention_layer_counts,
            inference_attention_type_override=inference_attention_type,
        )
        probas = th.sigmoid(logits)
        preds = (probas > 0.5).float()

    ## Validate the fixed-shape sweep output [V, B, C, H', W']
    if not th.is_tensor(logits) or logits.dim() != 2 or logits.shape[0] != _B:
        got = tuple(logits.shape) if th.is_tensor(logits) else type(logits)
        raise RuntimeError(
            f"expected sweep logits of shape [B, n_classes] with B={_B}, " f"got {got}"
        )
    if not th.is_tensor(variant_maps) or variant_maps.dim() != 5:
        got = tuple(variant_maps.shape) if th.is_tensor(variant_maps) else type(variant_maps)
        raise RuntimeError(f"expected sweep output maps of shape [V, B, C, H, W], got {got}")
    V, B, C, h_map, w_map = variant_maps.shape
    if V != n_counts:
        # ``nn.DataParallel`` gathers tuple outputs elementwise along dim 0,
        # so k replicas each returning [V, b, C, H, W] (b = B / k) arrive here
        # as [V * k, B / k, C, H, W]. Un-gather back to [V, B, C, H, W]
        # (variant-major, batch order matching the input) when the shapes are
        # consistent with that gather; otherwise fail loudly.
        if n_counts == 0 or V % n_counts != 0:
            raise RuntimeError(
                f"expected V={n_counts} variant maps (one per requested " f"count), got V={V}"
            )
        k = V // n_counts
        if k < 1 or _B % k != 0 or B != _B // k:
            raise RuntimeError(
                f"expected sweep output maps of shape "
                f"[{n_counts}, {_B}, C, H, W] (or a DataParallel gather "
                f"thereof), got {tuple(variant_maps.shape)}"
            )
        # [k, V, b, C, H, W] -> [V, k, b, C, H, W] -> [V, B, C, H, W]
        variant_maps = (
            variant_maps.reshape(k, n_counts, B, C, h_map, w_map)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(n_counts, _B, C, h_map, w_map)
        )
        V, B = n_counts, _B
    if B != _B:
        raise RuntimeError(
            f"expected B={_B} variant map rows (matching the input batch), " f"got B={B}"
        )
    if labels.dim() != 2 or C != labels.shape[1]:
        raise RuntimeError(
            f"expected C={labels.shape[1] if labels.dim() == 2 else '?'} "
            f"variant map planes (matching labels), got C={C}"
        )

    ## Post-processing
    # Keep interpolation/post-processing in float32 after the autocast model
    # forward; this also avoids unsupported CPU bfloat16 interpolation paths.
    variant_maps = variant_maps.float()

    # upsample attention to image resolution when not already there (e.g.
    # patch-resolution attention), vectorized over [V*B, C, H, W]
    if (h_map, w_map) != (H, W):
        variant_maps = resample(
            x=variant_maps.reshape(V * B, C, h_map, w_map), size=(H, W), mode=interpolation_mode
        )
        variant_maps = variant_maps.reshape(V, B, C, H, W)

    # Detach torch vars and send to CPU as NumPy. NumPy does not support
    # bfloat16 directly, so cast autocast outputs back to float32 first.
    probas = probas.detach().float().cpu().numpy()
    preds = preds.detach().float().cpu().numpy()
    labels = labels.detach().float().cpu().numpy()
    attention = variant_maps.detach().float().cpu().numpy()

    ## Build the per-variant records
    # Per-variant independent normalization is already performed by the model;
    # no further normalization is applied here, matching the ordinary
    # extractor's behaviour after the model return.
    sweep_dict: dict[str, dict[str, dict[int | str, np.ndarray]]] = {}
    for variant_idx, count in enumerate(attention_layer_counts):
        cam_dict = defaultdict(dict)

        # shared labels/probabilities/predictions (identical objects across
        # variants; computed exactly once above)
        for filename, img_labels, img_probas, img_preds in zip(filenames, labels, probas, preds):
            filename_key = str(filename)
            cam_dict[filename_key]["labels"] = img_labels
            cam_dict[filename_key]["probas"] = img_probas
            cam_dict[filename_key]["preds"] = img_preds

        # add this variant's model attention for positive ground truth labels
        variant_attention = attention[variant_idx]
        for image_idx, (filename, img_labels) in enumerate(zip(filenames, labels)):
            filename_key = str(filename)
            for label_idx in range(img_labels.shape[0]):
                if img_labels[label_idx].item() == 1:
                    # Explicit copy: this slice is a view into the shared
                    # [V, B, C, H, W] batch tensor (~1.4 GiB at 512).
                    # Consumers that retain maps across batches (the OneBox
                    # sweep's fit pool) would otherwise pin one whole batch
                    # array per live map and blow host memory (a
                    # diagnosed regression). NB:
                    # np.ascontiguousarray() would NOT copy here -- the 2-D
                    # slice is already C-contiguous -- so only an explicit
                    # copy breaks the aliasing.
                    cam_dict[filename_key][label_idx] = variant_attention[
                        image_idx, label_idx
                    ].copy()

        sweep_dict[variant_key(count)] = cam_dict

    return sweep_dict


def refine_cls_attentions(cls_attentions, patch_attn, w_featmap, h_featmap):
    """
    Refines cls_attentions using patch_attn. Align dimensions to ensure compatibility.
    """
    num_heads, batch_size, seq_len, _ = patch_attn.shape  # [12, batch, 196, 196]
    assert seq_len == w_featmap * h_featmap, "Mismatch in spatial dimensions."

    # Reshape cls_attentions to align with patch_attn
    C = cls_attentions.shape[1]  # number of classes
    cls_attentions = cls_attentions.view(batch_size, C, -1, 1)  # [batch, C, 196, 1]

    # Average across attention heads in patch_attn
    patch_attn = patch_attn.mean(dim=0)  # [batch, 196, 196]

    # Perform matmul: refine cls_attentions using patch_attn
    refined_attn = th.matmul(
        patch_attn.unsqueeze(1),  # [batch, 1, 196, 196]
        cls_attentions,  # [batch, C, 196, 1]
    )  # Result: [batch, C, 196, 1]

    # Reshape to original dimensions
    refined_attn = refined_attn.view(batch_size, C, w_featmap, h_featmap)

    return refined_attn


def interpolate_cls_attentions(cls_attentions, w_orig, h_orig, interp_method="bilinear"):
    # skip interpolation when the map is already at the target resolution
    # (e.g. image-res maps from SegmentCAM's fused upsampling head)
    if cls_attentions.shape[-2:] == (w_orig, h_orig):
        return cls_attentions
    return F.interpolate(
        cls_attentions,
        size=(w_orig, h_orig),
        mode=interp_method,
        align_corners=False,
    )


def process_attention(
    cls_attentions, patch_attn, images, patch_attn_refine, patch_size, interp_method="bilinear"
):
    """
    take in the class attentions for a batch of outputs from the transformer, and process them to the full image size
    """

    # fix dimensionality issues
    if images.dim() == 3:
        images = images.unsqueeze(0)
    batch_size = images.shape[0]

    # this maths is done to make the overlay of the attention map on the image look better
    # Keep host-side image conversion in float32 for bfloat16-safe NumPy
    # handling, even though dataset images are normally already float32.
    img_temp = images.permute(0, 2, 3, 1).detach().float().cpu().numpy()
    orig_images = np.zeros_like(img_temp)
    orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.0
    orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.0
    orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.0

    w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

    # here we pass the data through the model, extracting and refining the attention maps
    # Attention post-processing is explicitly float32 and outside the
    # model-forward autocast context on both CPU and CUDA.
    with get_autocast_context(images.device, enabled=False):
        # need to do the processing one-by-one because each of the images is a different size
        # don't need to worry about this here because the images are resized to a constant size for training

        # Model attention tensors may be bfloat16 when autocast is enabled;
        # post-processing below is intentionally performed in float32.
        cls_attentions = cls_attentions.float()
        patch_attn = patch_attn.float()

        image = images[0]
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # determine the effective width and height of the attention map
        w, h = (
            image.shape[2] - image.shape[2] % patch_size,
            image.shape[3] - image.shape[3] % patch_size,
        )
        w_featmap = w // patch_size
        h_featmap = h // patch_size

        if patch_attn_refine:
            # refinement mixes patch attention with class attention; only
            # valid when both are at patch resolution (skipped for image-res
            # maps from SegmentCAM's fused upsampling head)
            if cls_attentions.shape[-2:] == (h_featmap, w_featmap):
                cls_attentions = refine_cls_attentions(
                    cls_attentions, patch_attn, w_featmap, h_featmap
                )

        # interpolate the attention maps to the size of the original image
        cls_attentions = interpolate_cls_attentions(cls_attentions, w_orig, h_orig, interp_method)


        # normalise attention
        temp = cls_attentions.view(
            cls_attentions.shape[0], cls_attentions.shape[1], -1
        )  # put the spatial dimensions into one
        min_vals = temp.min(2, keepdim=True).values
        max_vals = temp.max(2, keepdim=True).values
        temp = (temp - min_vals) / (max_vals - min_vals + 1e-8)
        cls_attentions = temp.view(cls_attentions.size())

        # we did not apply sigmoid in the model itself so we could use mixed-precision loss functions
        # but we need output probabilities here, so we apply sigmoid
        # output = torch.sigmoid(output)

        # if args.out_crf is not None:
        #     for t in [args.low_alpha, args.high_alpha]:
        #         orig_image = orig_images[b].astype(np.uint8).copy(order="C")
        #         crf = _crf_with_alpha(cam_dict, t, orig_image)
        #         folder = args.out_crf + ("_%s" % t)
        #         if not os.path.exists(folder):
        #             os.makedirs(folder)
        #         np.save(os.path.join(folder, img_name + ".npy"), crf)

    return cls_attentions
