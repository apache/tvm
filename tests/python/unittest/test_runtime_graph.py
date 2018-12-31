import tvm
import numpy as np
import json
from tvm import rpc
from tvm import relay
from tvm.contrib import util, graph_runtime

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
        mod = graph_runtime.create(graph, mlib, tvm.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=a)
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.asnumpy(), a + 1)

    def check_remote():
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        server = rpc.Server("localhost")
        remote = rpc.connect(server.host, server.port)
        temp = util.tempdir()
        ctx = remote.cpu(0)
        path_dso = temp.relpath("dev_lib.so")
        mlib.export_library(path_dso)
        remote.upload(path_dso)
        mlib = remote.load_module("dev_lib.so")
        mod = graph_runtime.create(graph, mlib, remote.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=tvm.nd.array(a, ctx))
        out = tvm.nd.empty((n,), ctx=ctx)
        out = mod.get_output(0, out)
        np.testing.assert_equal(out.asnumpy(), a + 1)

    check_verify()
    check_remote()

def test_graph_symbolic_shape():
    b1 = tvm.var("n1")
    b2 = tvm.var("n2")
    b3 = tvm.var("n3")

    shape_var_bounds = {
        b1 : 128,
        b2 : 32,
        b3 : 12
    }

    x = relay.var("x",
              shape=[b1,
                     4,
                     2, 3], dtype="float32")




    y = relay.var("y",
              shape=[b2,
                     4,
                     2, 3], dtype="float32")

    z = relay.op.tensor.concatenate([x, y], axis=0)


    a = relay.var("a",
              shape=[b3,
                     4,
                     2, 3], dtype="float32")
    b = relay.var("b",
              shape=[27,
                     4,
                     2, 3], dtype="float32")
    c = relay.op.tensor.concatenate([a, b], axis=0)

    out = relay.op.tensor.concatenate([z, c], axis=0)

    func = relay.Function([x, y, a, b], out)
    graph, mod, param = relay.build_module.build(func, target="llvm", shape_var_bounds=shape_var_bounds)
    rt = tvm.contrib.graph_runtime.create(graph, mod, tvm.cpu())
    #### Test
    def test(n1, n2, n3):
       import numpy as np
       vars = {
              "n1": n1,
              "n2": n2,
              "n3": n3
       }
       shapes = {
              "x": (n1, 4, 2, 3),
              "y": (n2, 4, 2, 3),
              "a": (n3, 4, 2, 3),
              "b": (27, 4, 2, 3)
       }
       arrays = {
              "x": np.random.uniform(-1, 1, shapes["x"]).astype("float32"),
              "y": np.random.uniform(-1, 1, shapes["y"]).astype("float32"),
              "a": np.random.uniform(-1, 1, shapes["a"]).astype("float32"),
              "b": np.random.uniform(-1, 1, shapes["b"]).astype("float32"),
       }
       inputs = {k:tvm.nd.array(v) for k, v in arrays.items()}
 
       z = np.vstack((arrays["x"], arrays["y"]))
       c = np.vstack((arrays["a"], arrays["b"]))
       ans = np.vstack((z, c))
       rt.set_shape_variable(vars)
       rt.set_input(**inputs)
       rt.run()
       out = rt.get_output(0).asnumpy()
       np.testing.assert_allclose(ans, out, rtol=1e-5, atol=1e-5)

    test(1, 2, 3)
    test(30, 20, 10)
    test(5, 7, 9)
    test(1, 2, 3)
    test(5, 7, 9)
    test(30, 20, 10)

if __name__ == "__main__":
    test_graph_simple()
    test_graph_symbolic_shape()
