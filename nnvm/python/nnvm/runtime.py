"""Runtime environment for nnvm relies on TVM."""
import tvm
from tvm.contrib import rpc

class Module(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    tvm_module : tvm.Module
        The interal tvm module
    """
    def __init__(self, tvm_module):
        self.tvm_module = tvm_module
        self._set_input = tvm_module["set_input"]
        self._run = tvm_module["run"]
        self._get_output = tvm_module["get_output"]

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
            self._set_input(key, tvm.nd.array(value))
        for k, v in params.items():
            self._set_input(k, tvm.nd.array(v))
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

    def get_output(self, index, out):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The input index

        out : tvm.NDArray
            The output array container
        """
        self._get_output(index, out)
        return out

    def __getitem__(self, key):
        """Get internal module function

        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.tvm_module[key]



def create(graph, libmod, ctx):
    """Create a runtime executor module given the graph and module.

    Parameters
    ----------
    graph : The graph to be deployed
        The graph to be loaded.

    libmod : tvm.Module
        The module of the corresponding function

    ctx : TVMContext
        The context to deploy the module, can be local or remote.

    Returns
    -------
    graph_module : tvm.Module
        Runtime graph module to execute the graph.
    """
    json_str = graph if isinstance(graph, str) else graph.apply("SaveJSON").json_attr("json")
    device_type = ctx.device_type
    device_id = ctx.device_id
    if device_type >= rpc.RPC_SESS_MASK:
        assert libmod.type_key == "rpc"
        assert rpc._SessTableIndex(libmod) == ctx._rpc_sess._tbl_index
        hmod = rpc._ModuleHandle(libmod)
        fcreate = ctx._rpc_sess.get_function("nnvm.runtime.remote_create")
        device_type = device_type % rpc.RPC_SESS_MASK
        return Module(fcreate(json_str, hmod, device_type, device_id))
    fcreate = tvm.get_global_func("nnvm.runtime.create")
    return Module(fcreate(json_str, libmod, device_type, device_id))
