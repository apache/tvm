"""TVM operator flatten compute."""
from __future__ import absolute_import
import tvm
from .. import tag

@tvm.tag_scope(tag=tag.INJECTIVE)
def flatten(data):
    """Flattens the input array into a 2-D array by collapsing the higher dimensions.

    Parameters
    ----------
    data : tvm.Tensor
        Input array.

    Returns
    -------
    output : tvm.Tensor
        2-D array with collapsed higher dimensions.
    """
    ishape = data.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = [ishape[0], dim]

    def unwrap(idx, shape):
        index = []
        for s in reversed(shape):
            index.append(idx % s)
            idx = idx / s
        return list(reversed(index))

    return tvm.compute(oshape, lambda i, j: data(i, *unwrap(j, ishape[1:])))
