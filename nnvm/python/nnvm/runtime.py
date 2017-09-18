"""Runtime environment for nnvm relies on TVM."""
import tvm
from tvm.contrib import rpc

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
        return fcreate(json_str, hmod, device_type, device_id)

    fcreate = tvm.get_global_func("nnvm.runtime.create")
    return fcreate(json_str, libmod, device_type, device_id)
