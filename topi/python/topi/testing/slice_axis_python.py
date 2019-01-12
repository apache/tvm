"""Slice axis in python"""

def slice_axis_python(data, axis, begin, end=None):
    """Slice input array along specific axis.

    Parameters
    ----------
    data : numpy.ndarray
        The source array to be sliced.

    axis : int
        Axis to be sliced.

    begin: int
        The index to begin with in the slicing.

    end: int, optional
        The index indicating end of the slice.

    Returns
    -------
    ret : numpy.ndarray
        The computed result.
    """
    dshape = data.shape
    if axis < 0:
        axis += len(dshape)
    if begin < 0:
        begin += dshape[axis]
    if end <= 0:
        end += dshape[axis]
    slc = [slice(None)] * len(dshape)
    slc[axis] = slice(begin, end)
    return data[tuple(slc)]
