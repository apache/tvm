#pylint: disable=unused-argument
"""Internal module for quantization."""
from __future__ import absolute_import
import topi
from tvm._ffi.function import _init_api
from ..op import op as _reg


@_reg.register_compute("simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type, target):
    """Compiler for simulated_quantize."""
    assert len(inputs) == 4
    assert attrs.sign
    assert attrs.rounding == "round"

    data, scale, clip_min, clip_max = inputs

    # simulate rounding error
    scaled_data = topi.divide(data, scale)
    clipped_data = topi.maximum(topi.minimum(scaled_data, clip_max), clip_min)
    round_data = topi.round(clipped_data)

    # recover data
    rdata = topi.multiply(round_data, scale)
    return [rdata]


_reg.register_schedule("simulated_quantize", _reg.schedule_injective)
_reg.register_pattern("simulated_quantize", _reg.OpPattern.OPAQUE)

_init_api("relay._quantize", __name__)
