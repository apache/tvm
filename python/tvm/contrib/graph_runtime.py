"""Minimum graph runtime that executes graph containing TVM PackedFunc."""
from .._ffi.base import string_types
from .._ffi.function import get_global_func
from .._ffi._ctypes.function import ModuleHandle
from ..rpc import base as rpc_base
from .. import ndarray as nd
from tvm import module
from nnvm._base import ctypes, c_array



def _create_homogeneous(graph_json_str, libmod, ctx):
    """Create a homogeneous runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    libmod : tvm.Module
        The module of the corresponding function.

    ctx : TVMContext
        The context to deploy the module, can be local or remote.

    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.
    """
    device_type = ctx.device_type
    device_id = ctx.device_id
    if device_type >= rpc_base.RPC_SESS_MASK:
        assert libmod.type_key == "rpc"
        assert rpc_base._SessTableIndex(libmod) == ctx._rpc_sess._tbl_index
        hmod = rpc_base._ModuleHandle(libmod)
        fcreate = ctx._rpc_sess.get_function("tvm.graph_runtime.remote_create")
        device_type = device_type % rpc_base.RPC_SESS_MASK
        return GraphModule(fcreate(graph_json_str, hmod, device_type,
                                   device_id), ctx)

    fcreate = get_global_func("tvm.graph_runtime.create")
    return GraphModule(fcreate(graph_json_str, libmod, device_type,
                               device_id), ctx)


def _create_heterogeneous(graph_json_str, libmod_ctx, host_ctx):
    """Create a heterogeneous runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain tvm_op and device_copy_op that
        point to the name of PackedFunc in the one of the compiled module libs.

    libmod_ctx : tvm.Module to TVMContext dict
        The module and context pair of the corresponding function.

    host_ctx : TVMContext
        The local context to deploy the module.

    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.
    """
    if host_ctx.device_type >= rpc_base.RPC_SESS_MASK:
        raise RuntimeError(
            "rpc is not supported for heterogeneous execution yet.")

    # Fallback to use the homogeneous execution if there is only one context.
    if len(libmod_ctx) == 1:
        return _create_homogeneous(graph_json_str, list(libmod_ctx.keys())[0],
                                   list(libmod_ctx.values())[0])

    libs, device_types, device_ids = [], [], []
    # CPU is always used as the master device. Its device type is 1 as
    # defined in TVMContext and dlpack.h. The libmod_ctx is sorted according
    # to the device type field in TVMContext. It is used to guarantee that the
    # first lib and device in the array belong to CPU.
    for lib, ctx in sorted(libmod_ctx.items(), key=lambda x: x[1].device_type):
        if ctx.device_type >= rpc_base.RPC_SESS_MASK:
            raise RuntimeError(
                "rpc is not supported for heterogeneous execution yet.")
        libs.append(lib.handle)
        device_types.append(ctx.device_type)
        device_ids.append(ctx.device_id)

    lib_arr = c_array(ModuleHandle, libs)
    device_type_arr = c_array(ctypes.c_int, device_types)
    device_id_arr = c_array(ctypes.c_int, device_ids)
    void_lib_arr = ctypes.cast(lib_arr, ctypes.c_void_p)
    void_dt_arr = ctypes.cast(device_type_arr, ctypes.c_void_p)
    void_di_arr = ctypes.cast(device_id_arr, ctypes.c_void_p)

    fcreate = get_global_func("tvm.graph_runtime.create_heterogeneous")
    return GraphModule(fcreate(graph_json_str, void_lib_arr, void_dt_arr,
                                void_di_arr, len(libs)), host_ctx)


def create(graph_json_str, libmod, ctx):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    libmod : tvm.Module or dict of tvm.Module to TVMContext.
        The module of the corresponding function

    ctx : TVMContext
        The context to deploy the module, can be local or remote.

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
    
    if isinstance(libmod, module.Module):
        return _create_homogeneous(graph_json_str, libmod, ctx)
    elif (libmod, dict):
        return _create_heterogeneous(graph_json_str, libmod, ctx)
    else:
        raise ValueError("Expected type of libmod is tvm.Module or a dict of "
                         "tvm.Module to TVMContext, the input type is %s" %
                         type(libmod_ctx))


class GraphModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.

    ctx : TVMContext
        The context this module is under

    Attributes
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.

    ctx : TVMContext
        The context this module is under
    """

    def __init__(self, module, ctx):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        try:
            self._debug_get_output = module["debug_get_output"]
        except AttributeError:
            pass
        self._load_params = module["load_params"]
        self.ctx = ctx

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
            self._set_input(key, nd.array(value, ctx=self.ctx))
        for k, v in params.items():
            self._set_input(k, nd.array(v, ctx=self.ctx))
        return self

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

    def get_input(self, index, out):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        self._get_input(index, out)
        return out

    def get_output(self, index, out):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        self._get_output(index, out)
        return out

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
