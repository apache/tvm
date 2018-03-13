"""Local response normalization in python"""
import mxnet as mx

def lrn_nchw_python(a_np, size, bias, alpha, beta, b_np):
    """Local response norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    size : int
        normalisation window size

    bias : float
        offset to avoid dividing by 0. constant value

    alpha : float
        contant valie

    beta : float
        exponent constant value


    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return mx.ndarray.LRN(mx.nd.array(a_np), alpha, beta, bias, size, mx.nd.array(b_np))
