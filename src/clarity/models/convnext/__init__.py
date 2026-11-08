"""ConvNeXt / ConvNeXtV2 models.

Public surface: the ConvNeXtV2-B teacher.  Other ConvNeXt variants
are not part of this release and are never imported from this package init.
"""

from __future__ import annotations

from clarity.models.convnext.convnextv2 import ConvNeXtV2, ConvNeXtV2_B, get_convnextv2_b

__all__ = ["ConvNeXtV2", "ConvNeXtV2_B", "get_convnextv2_b"]
