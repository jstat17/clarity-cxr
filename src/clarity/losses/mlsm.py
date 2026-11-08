from torch import nn
import textwrap

from clarity.losses.registry import register_loss


@register_loss("mlsm")
class MultiLabelSoftMarginLoss(nn.MultiLabelSoftMarginLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean", **kwargs):
        self._size_average = size_average
        self._reduce = reduce
        super().__init__(weight, size_average, reduce, reduction)

    @staticmethod
    def _json_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        return repr(value)

    def __str__(self) -> str:
        return textwrap.dedent(f"""\
            MultiLabelSoftMarginLoss(
                weight={self.weight},
                size_average={self._size_average},
                reduce={self._reduce},
                reduction='{self.reduction}'
            )""")

    def dict_repr(self) -> dict[str, dict]:
        return {
            "MultiLabelSoftMarginLoss": {
                "weight": self._json_value(self.weight),
                "size_average": self._size_average,
                "reduce": self._reduce,
                "reduction": self.reduction,
            }
        }
