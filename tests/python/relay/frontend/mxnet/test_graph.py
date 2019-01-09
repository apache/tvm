import mxnet as mx
import tvm
from tvm import relay
import model_zoo
from model_zoo import _batch

def test_mlp():
    mx_sym = model_zoo.mx_mlp
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'data': (_batch, 1, 28, 28)})
    from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
    relay_sym = model_zoo.relay_mlp
    assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

def test_vgg():
    for n in [11, 13, 16, 19]:
        mx_sym = model_zoo.mx_vgg[n]
        from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'data': (_batch, 3, 224, 224)})
        from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
        relay_sym = model_zoo.relay_vgg[n]
        assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

def test_resnet():
    for n in [18, 34, 50, 101, 152, 200, 269]:
        mx_sym = model_zoo.mx_resnet[n]
        from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'data': (_batch, 3, 224, 224)})
        from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
        relay_sym = model_zoo.relay_resnet[n]
        assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

def test_squeezenet():
    for version in ['1.0', '1.1']:
        mx_sym = model_zoo.mx_squeezenet[version]
        from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'data': (_batch, 3, 224, 224)})
        from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
        relay_sym = model_zoo.relay_squeezenet[version]
        assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

def test_inception_v3():
    mx_sym = model_zoo.mx_inception_v3
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'data': (_batch, 3, 299, 299)})
    from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
    relay_sym = model_zoo.relay_inception_v3
    assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

def test_dqn():
    mx_sym = model_zoo.mx_dqn
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'data': (_batch, 4, 84, 84)})
    from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
    relay_sym = model_zoo.relay_dqn
    assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

def test_dcgan():
    mx_sym = model_zoo.mx_dcgan
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'data': (_batch, 100)})
    from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
    relay_sym = model_zoo.relay_dcgan
    assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

def test_multi_outputs():
    def compose_mxnet(**kwargs):
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        z = mx.sym.split(x, **kwargs)
        return mx.sym.broadcast_sub(mx.sym.broadcast_add(z[0], z[2]), y)
    def compose_relay(**kwargs):
        x = relay.var("x", shape=(_batch, 3, 224, 224))
        y = relay.var("y", shape=(1,))
        z = relay.split(x, **kwargs)
        ret = z[0] + z[2] - y
        args = relay.ir_pass.free_vars(ret)
        return relay.Function(args, ret)
    mx_sym = compose_mxnet(num_outputs=3, axis=1)
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, {'x': (_batch, 3, 224, 224), 'y': (1,)})
    from_mx_sym = relay.ir_pass.infer_type(from_mx_sym)
    relay_sym = compose_relay(indices_or_sections=3, axis=1)
    relay_sym = relay.ir_pass.infer_type(relay_sym)
    assert relay.ir_pass.alpha_equal(from_mx_sym, relay_sym)

if __name__ == '__main__':
    test_mlp()
    test_vgg()
    test_resnet()
    test_squeezenet()
    test_inception_v3()
    test_dqn()
    test_dcgan()
    test_multi_outputs()
