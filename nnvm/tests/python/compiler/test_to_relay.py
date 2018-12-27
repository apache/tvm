import nnvm
from nnvm import testing
from nnvm import to_relay
import tvm
from tvm.relay import ir_pass
from tvm.relay import create_executor
from tvm.contrib import graph_runtime
import numpy as np

def check_model(sym, shapes, dtypes, params):
    net = nnvm.graph.create(sym)
    graph_json, mod, params = nnvm.compiler.build(
        net,
        'llvm',
        shape=shapes,
        dtype=dtypes,
        params=params)
    nnvm_rts = graph_runtime.create(graph_json, mod, tvm.cpu(0))
    inputs = {}
    for name in shapes:
        np_array = np.random.rand(*shapes[name]).astype('float32')
        inputs[name] = tvm.nd.array(np_array)

    nnvm_rts.set_input(**params)
    nnvm_rts.run(**inputs)
    nnvm_out = nnvm_rts.get_output(0)
    relay_model, params = to_relay.to_relay(net, shapes, dtypes, params)
    relay_model = ir_pass.infer_type(relay_model)
    relay_rts = create_executor(kind='graph', ctx=tvm.cpu(0), target='llvm')
    inputs.update(params)
    relay_out = relay_rts.evaluate(relay_model)(*list(inputs.values()))
    np.testing.assert_allclose(nnvm_out.asnumpy(), relay_out.asnumpy())

# def test_mlp():
#     mlp, params = testing.mlp.get_workload(1)
#     shapes =  { "data": (10, 3, 224, 224) }
#     dtypes =  { "data": 'float32' }
#     check_model(mlp, shapes, dtypes, params)

if __name__ == "__main__":
    test_mlp()
