# pylint: disable=invalid-name
"""Helper utility to save parameter dicts."""
import tvm
from tvm.relay.backend.interpreter import TensorValue


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
    # Wrap the `NDArray`s in a `TensorValue`, because raw `NDArray`s are not
    # serializable, and `TensorValue`s are.
    json_str = tvm.save_json({key: TensorValue(val)
                              for (key, val) in params.items()})
    return bytearray(json_str.encode('utf-8'))


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
    param_json = tvm.load_json(param_bytes.decode('utf-8'))
    # Pull the `data` field from each of the `TensorValue`s to get `NDArray`s.
    return {key: val.data for (key, val) in param_json.items()}
