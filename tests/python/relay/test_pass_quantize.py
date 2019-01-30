import math
import numpy as np
import tvm
from tvm import relay
from tvm.relay import quantize as qtz


def make_dataset(graph, size=100):
    args = relay.ir_pass.infer_type(graph).params
    def create_arr(var):
        ttype = var.type_annotation
        np_arr = np.random.uniform(-1.0, 1.0, size=ttype.concrete_shape).astype(ttype.dtype)
        return tvm.ndarray.array(np_arr)

    params = {}
    for arg in args:
        if arg.name_hint == 'data':
            dataset = [{'data': create_arr(arg)} for _ in range(size)]
        else:
            params[arg.name_hint] = create_arr(arg)
    return dataset, params


def test_simulated_quantize():
    data = relay.var("data", relay.ty.TensorType((3, 4, 5, 6), "float32"))
    out = qtz._annotate.attach_simulated_quantize(data, 1)
    out = relay.ir_pass.infer_type(out)
    assert out.checked_type == out.args[0].checked_type
    assert out.args[1].checked_type == relay.ty.TensorType(tuple(), "float32")
    assert out.args[2].checked_type == relay.ty.TensorType(tuple(), "float32")
    assert out.args[3].checked_type == relay.ty.TensorType(tuple(), "float32")


def test_quantize_pass():
    def quantize_weight(arr):
        maximum = np.amax(np.abs(arr.asnumpy()))
        scale = 2**math.ceil(math.log(maximum, 2))
        out = np.around(arr.asnumpy() / scale * 128).astype('int8')
        out = np.clip(out, -127, 127)
        return relay.const(out, 'int8')

    n, c, h, w = 1, 3, 224, 224
    def make_graph(data):
        weight = relay.var("conv_weight")
        out = relay.nn.conv2d(data, weight, kernel_size=(3, 3), padding=(1, 1), channels=c)
        out = relay.Function(relay.ir_pass.free_vars(out), out)
        return out

    def make_qgraph(data, weight):
        out = data * relay.const(32.0)
        out = relay.round(out)
        out = relay.clip(out, a_min=-127, a_max=127)
        out = out.astype('int8')

        out = relay.nn.conv2d(out, weight, kernel_size=(3, 3),
                              padding=(1, 1), channels=c, out_dtype='int32')
        out = out.astype('float32')
        out = relay.multiply(out, relay.const(0.00024414062))
        out = relay.Function(relay.ir_pass.free_vars(out), out)
        return out

    data = relay.var("data", relay.TensorType((n, c, h, w), "float32"))
    graph = make_graph(data)
    dataset, params = make_dataset(graph, 10)

    with qtz.qconfig(skip_k_conv=0, global_scale=4.0,
                     round_for_shift=False, store_lowbit_output=False):
        qgraph0 = qtz.quantize(graph, params)
        qgraph0 = relay.ir_pass.infer_type(qgraph0)

    conv_weight = quantize_weight(params['conv_weight'])
    qgraph1 = make_qgraph(data, conv_weight)
    qgraph1 = relay.ir_pass.infer_type(qgraph1)

    graph = relay.create_executor('graph')
    res0 = graph.evaluate(qgraph0)(dataset[0]['data'])
    res1 = graph.evaluate(qgraph1)(dataset[0]['data'])
    tvm.testing.assert_allclose(res0.asnumpy(), res1.asnumpy())


if __name__ == "__main__":
    np.random.seed(42)
    test_simulated_quantize()
    test_quantize_pass()
