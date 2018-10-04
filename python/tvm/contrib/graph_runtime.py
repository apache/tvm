"""Minimum graph runtime that executes graph containing TVM PackedFunc."""
import numpy as np

from .._ffi.base import string_types
from .._ffi.function import get_global_func
from .._ffi.runtime_ctypes import TVMContext
from ..rpc import base as rpc_base

def create(graph_json_str, libmod, ctx):
    """Create a runtime executor module given a graph and module.
    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.
    libmod : tvm.Module
        The module of the corresponding function
    ctx : TVMContext or list of TVMContext
        The context to deploy the module. It can be local or remote when there
        is only one TVMContext. Otherwise, the first context in the list will
        be used as this purpose. All context should be given for heterogeneous
        execution.
    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.
    """
    if not isinstance(graph_json_str, string_types):
        try:
            graph_json_str = graph_json_str._tvm_graph_json()
        except AttributeError:
            raise ValueError("Type %s is not supported" % type(graph_json_str))

    ctx, num_rpc_ctx, device_type_id = get_device_ctx(libmod, ctx)

    if num_rpc_ctx == len(ctx):
        hmod = rpc_base._ModuleHandle(libmod)
        fcreate = ctx[0]._rpc_sess.get_function("tvm.graph_runtime.remote_create")
        return GraphModule(fcreate(graph_json_str, hmod, *device_type_id))

    fcreate = get_global_func("tvm.graph_runtime.create")
    return GraphModule(fcreate(graph_json_str, libmod, *device_type_id))

def get_device_ctx(libmod, ctx):
    """Parse and validate all the device context(s).
    Parameters
    ----------
    libmod : tvm.Module
        The module of the corresponding function
    ctx : TVMContext or list of TVMContext
    Returns
    -------
    ctx : list of TVMContext
    num_rpc_ctx : Number of rpc contexts
    device_type_id : List of device type and device id
    """

    if isinstance(ctx, TVMContext):
        ctx = [ctx]
    elif not isinstance(ctx, (list, tuple)):
        raise ValueError("ctx has to be the type of TVMContext or a list of "
                         "TVMCTVMContext")
    for cur_ctx in ctx:
        if not isinstance(cur_ctx, TVMContext):
            raise ValueError("ctx has to be the type of TVMContext or a list "
                             "of TVMContext")

    # device_type_id[0], device_type_id[1] are used as the primary/fallback
    # context type and id. All other ones are used as device context for
    # heterogeneous execution.
    num_rpc_ctx = 0
    device_type_id = []
    for cur_ctx in ctx:
        device_type = cur_ctx.device_type
        if device_type >= rpc_base.RPC_SESS_MASK:
            assert libmod.type_key == "rpc"
            assert rpc_base._SessTableIndex(
                libmod) == cur_ctx._rpc_sess._tbl_index
            num_rpc_ctx += 1
            device_type = cur_ctx.device_type % rpc_base.RPC_SESS_MASK
        device_type_id.append(device_type)
        device_type_id.append(cur_ctx.device_id)

    if 0 < num_rpc_ctx < len(ctx):
        raise ValueError("Either all or none of the contexts should be rpc.")
    return ctx, num_rpc_ctx, device_type_id


class GraphModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.

    Attributes
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.
    """

    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        try:
            self._debug_get_output = module["debug_get_output"]
        except AttributeError:
            pass
        self._load_params = module["load_params"]

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additonal arguments
        """
        if key:
            self._get_input(key).copyfrom(value)

        if params:
            # upload big arrays first to avoid memory issue in rpc mode
            keys = list(params.keys())
            keys.sort(key=lambda x: -np.prod(params[x].shape))
            for k in keys:
                self._get_input(k).copyfrom(params[k])

    def run(self, **input_dict):
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()

    def get_num_outputs(self):
        """Get the number of outputs from the graph

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_input(self, index, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(index).copyto(out)
            return out

        return self._get_input(index)

    def get_output(self, index, out=None):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_output(index, out)
            return out

        return self._get_output(index)

    def debug_get_output(self, node, out):
        """Run graph upto node and get the output to out

        Parameters
        ----------
        node : int / str
            The node index or name

        out : NDArray
            The output array container
        """
        if hasattr(self, '_debug_get_output'):
            self._debug_get_output(node, out)
        else:
            raise RuntimeError(
                "Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0")
        return out

    def load_params(self, params_bytes):
        """Load parameters from serialized byte array of parameter dict.

        Parameters
        ----------
        params_bytes : bytearray
            The serialized parameter dict.
        """
        self._load_params(bytearray(params_bytes))

    def __getitem__(self, key):
        """Get internal module function

        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.module[key]
