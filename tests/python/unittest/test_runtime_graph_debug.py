import os
import tvm
import numpy as np
import json
from tvm.contrib.debugger import debug_runtime as graph_runtime

def test_graph_simple():
    n = 4
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)

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

    def check_verify():
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        try:
            mod = graph_runtime.create(graph, mlib, tvm.cpu(0))
        except ValueError:
            return

        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.set_input(x=a)

        #verify dumproot created
        directory = mod._dump_path
        assert(os.path.exists(directory))

        #verify graph is there
        GRAPH_DUMP_FILE_NAME = '_tvmdbg_graph_dump.json'
        assert(len(os.listdir(directory)) == 1)

        #verify the file name is proper
        assert(os.path.exists(os.path.join(directory, GRAPH_DUMP_FILE_NAME)))

        mod.run()
        #Verify the tensors are dumped
        assert(len(os.listdir(directory)) > 1)

        #verify the output is correct
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.asnumpy(), a + 1)

        mod.exit()
        #verify dump root delete after cleanup
        assert(not os.path.exists(directory))

    check_verify()

if __name__ == "__main__":
    test_graph_simple()
