"""Helper utility to save parameter dict"""
import tvm

_save_param_dict = tvm.get_global_func("nnvm.compiler._save_param_dict")
_load_param_dict = tvm.get_global_func("nnvm.compiler._load_param_dict")

def save_param_dict(params):
    """Save parameter dictionary to binary bytes.

    The result binary bytes can be loaded by the
    GraphModule with API "load_params".

    Parameters
    ----------
    params : dict of str to NDArray
        The parameter dictionary.

    Returns
    -------
    param_bytes: bytearray
        Serialized parameters.

    Examples
    --------
    .. code-block:: python

       # compile and save the modules to file.
       graph, lib, params = nnvm.compiler.build(
          graph, target, shape={"data", data_shape}, params=params)
       module = graph_runtime.create(graph, lib, tvm.gpu(0))
       # save the parameters as byte array
       param_bytes = nnvm.compiler.save_param_dict(params)
       # We can serialize the param_bytes and load it back later.
       # Pass in byte array to module to directly set parameters
       module["load_params"](param_bytes)
    """
    args = []
    for k, v in params.items():
        args.append(k)
        args.append(tvm.nd.array(v))
    return _save_param_dict(*args)


def load_param_dict(param_bytes):
    """Load parameter dictionary to binary bytes.

    Parameters
    ----------
    param_bytes: bytearray
        Serialized parameters.

    Returns
    -------
    params : dict of str to NDArray
        The parameter dictionary.
    """
    if isinstance(param_bytes, (bytes, str)):
        param_bytes = bytearray(param_bytes)
    load_mod = _load_param_dict(param_bytes)
    size = load_mod(0)
    param_dict = {}
    for i in range(size):
        key = load_mod(1, i)
        dltensor_handle = load_mod(2, i)
        param_dict[key] = tvm.nd.NDArray(dltensor_handle, False)
    return param_dict
