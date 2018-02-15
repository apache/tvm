"""External function interface to IMAGE libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin

def image_decode(input_buf, output_buf):
    """Create an extern op that decode jpeg compressed stream input_buf to output_buf

    Parameters
    ----------
    input_buf : Tensor
        tensor holding input jpeg stream

    Returns
    -------
    output_buf : Tensor
        tensor to place the decided jpeg stream
    """
    return _api.extern(
        output_buf.shape, [input_buf, output_buf],
        lambda input_buf, output_buf: _intrin.call_packed(
            "tvm.contrib.image.decode",
            input_buf[0], output_buf[0]), name="decode", dtype='uint8')
