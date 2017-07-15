"""Logics related to build."""
import nnvm.graph as graph
import tvm
import json

DTYPE_DICT = {
    "float32": 0
}

_create_exec = tvm.get_global_func("tvm_graph._create_executor")

def build(sym, target, shape, dtype="float32"):
    # Do shape inference in python.
    g = graph.create(sym)
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    list_shape = [[]] * jnode_row_ptr[-1]
    list_dtype = [DTYPE_DICT[dtype]] * jnode_row_ptr[-1]
    for k, v in shape.items():
        list_shape[jnode_row_ptr[nindex[k]]] = v
    g._set_json_attr("shape", list_shape, 'list_shape')
    g._set_json_attr("dtype", list_dtype, 'list_int')
    g._set_json_attr("target", target, 'str')
    g = g.apply("InferShape").apply("InferType")
    g = g.apply("GraphPartition").apply("GraphFuse")
    return g


def bind(g, ctx):
    m = _create_exec(g.handle, ctx.device_type, ctx.device_id)
    return m

_get_module = tvm.get_global_func("tvm_graph._get_module_from_graph")

def compile_graph(lib_fname, sym, target, shape, dtype="float32"):
    g = build(sym, target, shape, dtype)
    m = _get_module(g.handle)
    m.save(lib_fname)
    json_str = g.apply('SaveJSON').json_attr('json')
    return json_str

@tvm.register_func("tvm_graph.lower")
def _lower(sch, inputs, func_name):
    f = tvm.lower(sch, inputs, name=func_name)
    return f if isinstance(
        f, (tvm.container.Array, tuple, list)) else [f]


@tvm.register_func("tvm_graph.build_target")
def _build(funcs, target):
    return tvm.build(funcs, target=target)


_save_param_dict = tvm.get_global_func("tvm_graph._save_param_dict")

def save_params(fname, params):
    args = []
    args.append(fname)
    args.append(len(params))
    for kv in params.items():
        args.append(kv[0])
        args.append(kv[1])
    _save_param_dict(*args)


def remote_load_exec(sess, sym_json, remote_module_name, param_blob, ctx):
    """Load a remote graph executor, with the local files.
    Parameters
    ----------
    sym_json : str
        The symbol json file.

    remote_module_fname : str
        The relative library location to remote temp folder. The
        library need to be uploaded first.

    param_blob : bytes or bytearray
        The binary file to the local parameters.

    Returns
    -------
    exec : GraphExecutor
        The remote graph executor containing remote function.
    """
    if "load_executor" not in sess._remote_funcs:
        sess._remote_funcs["load_executor"] = sess.get_function("tvm_graph._load_executor")
    assert ctx.device_type / tvm.contrib.rpc.RPC_SESS_MASK == sess._tbl_index + 1
    device_type = ctx.device_type % tvm.contrib.rpc.RPC_SESS_MASK
    return sess._remote_funcs["load_executor"](sym_json,
                                               remote_module_name,
                                               bytearray(param_blob),
                                               device_type,
                                               ctx.device_id)
