"""Remote session API to support remote and distributed auto-tuning."""
import tvm

from tvm.contrib import graph_runtime, util

import time

def run_remote_module(session, graph, lib, params, input_data,
                      remote_dev_type="cpu", remote_dev_id=0,
                      run_times=10, export_lib_format=".o"):
    """
    Run a graph module in a remote RPC session.

    Parameters
    ----------
    session : RPCSession
        Remote RPC session.

    graph : Graph
        The execution graph.

    lib : tvm.Module
        The module that comes with the execution graph.

    params : dict of str to tvm.ndarray
        Parameter dictionary

    input_data : dict of str to tvm.ndarray
        Input data dictionary

    remote_dev_type : optional, int or str
        Remote device type

    remote_dev_id : optional, int
        Remote device id

    run_times : optional, int
        Total execution times.

    export_lib_format : optional, str
        Remote lib format. Currently ".o", ".so"
        and ".tar" are supported.

    Returns
    -------
    out : float
        Average execution time in second.
    """
    if not export_lib_format.endswith(".o") and \
            not export_lib_format.endswith(".so") and \
            not export_lib_format.endswith(".tar"):
        raise RuntimeError("export_lib_format %s is not supported."
                           % export_lib_format)

    lib_name = "remote_lib%s" % export_lib_format
    temp = util.tempdir()
    path = temp.relpath(lib_name)
    if export_lib_format == ".o":
        lib.save(path)
    else:
        lib.export_library(path)
    session.upload(path)
    ctx = session.context(remote_dev_type, remote_dev_id)
    rlib = session.load_module(lib_name)
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
    rdata = {k: tvm.nd.array(v, ctx) for k, v in input_data.items()}
    module = graph_runtime.create(graph, rlib, ctx)
    module.set_input(**rparams)
    module.set_input(**rdata)
    time_eval = module.module.time_evaluator("run", ctx, number=run_times)
    return time_eval().mean
