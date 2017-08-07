"""External function interface to CuDNN v7 library."""
from collections import namedtuple
from .. import api as _api
from .. import intrin as _intrin
from .. import ndarray as _nd
from .. import get_global_func as _get_global_func


# algos can be read from cudnn.h
_FWD_ALGOS = [
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
]

_BWD_FILTER_ALGOS = [
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
    # non-deterministic
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
    # non-deterministic, algo0 with workspaceS
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
    # not implemented
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
]

_BWD_DATA_ALGOS = [
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
    # non-deterministic
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",
]

_ALGO_TYPE = [
    "fwd",
    "bwd_filter",
    "bwd_data"
]

def algo_to_tensor(algo_type, algo_name):
    """Return a tensor represents the algorithm, which can be used in
    calling CuDNN function

    Parameters
    ----------
        algo_type : str
            ["fwd", "bwd_filter", "bwd_data]
        algo_name : str
            algorithm name in cudnn definition
            fwd = [
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
            ]
            bwd_filter = [
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
                # non-deterministic
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
                # non-deterministic, algo0 with workspaceS
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
                # not implemented
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
            ]
            bwd_data = [
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
                # non-deterministic
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",
            ]
    Returns
    -------
        algo: idx
            algorithm index in int

    """
    idx = -1
    if algo_name == "fwd":
        idx = _FWD_ALGOS.index(algo_name)
    elif algo_name == "bwd_filter":
        idx = _BWD_FILTER_ALGOS.index(algo_name)
    elif algo_name == "bwd_data":
        idx = _BWD_DATA_ALGOS.index(algo_name)
    assert idx >= 0
    return idx

ConvParam = namedtuple('ConvParam',
                       [input_filter,
                        output_filter,
                        kernel,
                        pad,
                        stride,
                        dilation,
                        tensor_format,
                        conv_mode,
                        conv_dim])

def conv_output_shape(param, input_shape):
    """Return the output shape by given convolution param and input shape

    Parameters
    ----------
    param: ConvParam
        parameters for CuDNN convolution
    input_shape: NDArray with int32 type
        input data shape

    Returns
    -------
    oshape: NDArray
        output feature map shape
    """
    assert isinstance(param, ConvParam) == True
    assert isinstance(input_shape, _nd.array) == True
    assert len(param.kernel) == param.conv_dim
    assert len(param.pad) == param.conv_dim
    assert len(param.stride) == param.conv_dim
    assert len(param.dilation) == param.conv_dim
    

