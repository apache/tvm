"""gather_nd in python"""

def strided_slice_python(data, begin, end, strides):
    """Python version of strided slice operator.

    Parameters
    ----------
    data : numpy.ndarray
        Input data

    begin : list
        Begining of the slices.

    end : list
        End of the slices.

    strides : list
        The stride of each slice.

    Returns
    -------
    result : numpy.ndarray
        The sliced result.
    """
    strides = [] if strides is None else strides
    slices = []
    for i in range(len(data.shape)):
        slices.append(slice(
            begin[i] if i < len(begin) else None,
            end[i] if i < len(end) else None,
            strides[i] if i < len(strides) else None))
    return data[tuple(slices)]
