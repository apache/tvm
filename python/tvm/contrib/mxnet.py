"""MXNet bridge wrap Function MXNet's async function."""
from __future__ import absolute_import as _abs

from .. import api, _api_internal, ndarray
from ..module import Module

# pylint: disable=invalid-name
_wrap_async = None


def to_mxnet_func(func, const_loc=None):
    """Wrap a TVM function as MXNet function

    MXNet function runs asynchrously via its engine.

    Parameters
    ----------
    func : Function
        A TVM function that can take positional arguments

    const_loc : list of int
        List of integers indicating the argument position
        of read only NDArray argument.
        The NDArray argument location that are not annotated
        will be viewed as mutable arrays in MXNet's engine.

    Returns
    -------
    async_func : Function
        A function that can take MXNet NDArray as argument
        in places that used to expect TVM NDArray.
        Run asynchrously in MXNet's async engine.
    """
    # only import mxnet when wrap get called.
    # pylint: disable=import-self
    import mxnet
    if isinstance(func, Module):
        func = func.entry_func

    def _get_bridge_func():
        """Get MXNet bridge function"""
        if not mxnet.base._LIB.MXTVMBridge:
            raise RuntimeError(
                "MXTVMBridge not exist in mxnet package,"
                " please update to latest version")

        fdict = api.extract_ext_funcs(mxnet.base._LIB.MXTVMBridge)
        ret = fdict["WrapAsyncCall"]
        ret.is_global = True
        return ret
    global _wrap_async

    if _wrap_async is None:
        # Register extension type in first time
        _wrap_async = _get_bridge_func()
        ndarray.register_extension(mxnet.nd.NDArray)

    const_loc = const_loc if const_loc else []
    return _wrap_async(func, _api_internal._TVMSetStream, len(const_loc), *const_loc)
