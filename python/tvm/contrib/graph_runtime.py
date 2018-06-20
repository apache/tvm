"""Minimum graph runtime that executes graph containing TVM PackedFunc."""
from .._ffi.base import string_types
from .._ffi.function import get_global_func
from .rpc import base as rpc_base
from .. import ndarray as nd
from ..tools.debug.runtime import debugruntime
from ..tools.debug.util import common

def create(graph_json_str, libmod, ctx, debug=False):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    libmod : tvm.Module
        The module of the corresponding function

    ctx : TVMContext
        The context to deploy the module, can be local or remote.

    debug : bool
        To enable or disable the debugging

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
    device_type = ctx.device_type
    device_id = ctx.device_id
    if device_type >= rpc_base.RPC_SESS_MASK:
        assert libmod.type_key == "rpc"
        assert rpc_base._SessTableIndex(libmod) == ctx._rpc_sess._tbl_index
        hmod = rpc_base._ModuleHandle(libmod)
        fcreate = ctx._rpc_sess.get_function("tvm.graph_runtime.remote_create")
        device_type = device_type % rpc_base.RPC_SESS_MASK
        func_obj = fcreate(graph_json_str, hmod, device_type, device_id)
        return GraphModule(func_obj, ctx, graph_json_str, debug)
    fcreate = get_global_func("tvm.graph_runtime.create")
    func_obj = fcreate(graph_json_str, libmod, device_type, device_id)
    return GraphModule(func_obj, ctx, graph_json_str, debug)


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
    def __init__(self, module, ctx, graph_json_str, debug):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._set_debug_buffer = module["set_debug_buffer"]
        self._debug_run = module["debug_run"]
        self._load_params = module["load_params"]
        self.ctx = ctx
        self.debug = debug
        if self.debug:
            self.dbgobj = debugruntime.create(self, graph_json_str, ctx)

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
        if self.debug:
            self.dbgobj.set_input(key, value, **params)
        return self

    def set_debug_buffer(self):
        """Set the debug out buffers for each tvm nodes

        Parameters
        ----------
        None
        """

        for eid in range(self.dbgobj.get_debug_buffer_count()):
            self._set_debug_buffer(self.dbgobj.get_debug_buffer(eid))

    def _debug_cli_run(self):
        """Invoke cli and when user execute any command get_run_start_resp will return response"""
        cli_command = self.dbgobj.get_run_command()
        run_start_resp = cli_command.get_run_start_resp()
        retvals = True
        if run_start_resp.action == common.CLIRunStartAction.DEBUG_RUN:
            self.set_debug_buffer()
            retvals = self._debug_run()
            self.dbgobj.dump_output()
            self.dbgobj.run_end(cli_command, retvals)

        elif run_start_resp.action == common.CLIRunStartAction.NON_DEBUG_RUN:
            retvals = self._run()
            self.dbgobj.run_end(cli_command, retvals)

    def run(self, **input_dict):
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)

        if not self.debug:
            self._run()
        else:
            self._debug_cli_run()

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
