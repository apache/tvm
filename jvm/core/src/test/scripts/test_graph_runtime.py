import os

import tvm
import json
from tvm.contrib import graph_runtime

def dump_graph_lib(target_dir):
    dim = 4
    A = tvm.placeholder((dim,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    sched = tvm.create_schedule(B.op)

    node0 = {"op": "null", "name": "x", "inputs": []}
    node1 = {"op": "tvm_op", "name": "add",
             "inputs": [[0, 0, 0]],
             "attrs": {"func_name": "myadd",
                       "flatten_data": "1",
                       "num_inputs" : "1",
                    "num_outputs" : "1"}}
    nodes = [node0, node1]
    arg_nodes = [0]
    node_row_ptr = [0, 1, 2]
    outputs = [[1, 0, 0]]
    shape = (4,)
    attrs = {
        "shape" : ["list_shape", [shape, shape]],
        "dltype" : ["list_str", ["float32", "float32"]],
        "storage_id" : ["list_int", [0, 1]],
    }
    graph = {"nodes": nodes,
             "arg_nodes": arg_nodes,
             "node_row_ptr": node_row_ptr,
             "heads": outputs,
             "attrs": attrs}

    graph = json.dumps(graph)
    mlib = tvm.build(sched, [A, B], "llvm", name="myadd")

    mlib.export_library(os.path.join(target_dir, "graph_addone_lib.so"))
    with open(os.path.join(target_dir, "graph_addone.json"), "w") as fo:
        fo.write(graph)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    dump_graph_lib(sys.argv[1])
