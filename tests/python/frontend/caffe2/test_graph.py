"""Test graph equality of caffe2 models."""
from tvm import relay
from model_zoo import c2_squeezenet, relay_squeezenet


def compare_graph(f1, f2):
    f1 = relay.ir_pass.infer_type(f1)
    f2 = relay.ir_pass.infer_type(f2)
    assert relay.ir_pass.alpha_equal(f1, f2)


def test_squeeze_net():
    shape_dict = {'data': (1, 3, 224, 224)}
    dtype_dict = {'data': 'float32'}
    from_c2_func, _ = relay.frontend.from_caffe2(c2_squeezenet.init_net, c2_squeezenet.predict_net, shape_dict, dtype_dict)
    relay_func, _ = relay_squeezenet()
    compare_graph(from_c2_func, relay_func)


if __name__ == '__main__':
    test_squeeze_net()
