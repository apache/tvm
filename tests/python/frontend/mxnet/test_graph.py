import mxnet as mx
from tvm import relay
import model_zoo

def compare_graph(f1, f2):
    f1 = relay.ir_pass.infer_type(f1)
    f2 = relay.ir_pass.infer_type(f2)
    assert relay.ir_pass.alpha_equal(f1, f2)

def test_mlp():
    shape = {"data": (1, 1, 28, 28)}
    mx_fun = model_zoo.mx_mlp()
    from_mx_fun, _ = relay.frontend.from_mxnet(mx_fun, shape=shape)
    relay_fun = model_zoo.relay_mlp()
    compare_graph(from_mx_fun, relay_fun)


def test_vgg():
    shape = {"data": (1, 3, 224, 224)}
    for n in [11, 13, 16, 19]:
        mx_sym = model_zoo.mx_vgg(n)
        from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, shape=shape)
        relay_sym = model_zoo.relay_vgg(n)
        compare_graph(from_mx_sym, relay_sym)


def test_resnet():
    shape = {"data": (1, 3, 224, 224)}
    for n in [18, 34, 50, 101]:
        mx_sym = model_zoo.mx_resnet(n)
        from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, shape=shape)
        relay_sym = model_zoo.relay_resnet(n)
        compare_graph(from_mx_sym, relay_sym)


def test_squeezenet():
    shape = {"data": (1, 3, 224, 224)}
    for version in ['1.0', '1.1']:
        mx_sym = model_zoo.mx_squeezenet(version)
        from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, shape)
        relay_sym = model_zoo.relay_squeezenet(version)
        compare_graph(from_mx_sym, relay_sym)


def test_inception_v3():
    shape = {"data": (1, 3, 299, 299)}
    mx_sym = model_zoo.mx_inception_v3()
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, shape)
    relay_sym = model_zoo.relay_inception_v3()
    compare_graph(from_mx_sym, relay_sym)


def test_dqn():
    shape = {"data": (1, 4, 84, 84)}
    mx_sym = model_zoo.mx_dqn()
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, shape)
    relay_sym = model_zoo.relay_dqn()
    compare_graph(from_mx_sym, relay_sym)


def test_dcgan():
    shape = {"data": (2, 100)}
    mx_sym = model_zoo.mx_dcgan()
    from_mx_sym, _ = relay.frontend.from_mxnet(mx_sym, shape)
    relay_sym = model_zoo.relay_dcgan(batch_size=2)
    compare_graph(from_mx_sym, relay_sym)


def test_multi_outputs():
    xshape = (10, 27)
    yshape = (10, 9)

    def mx_compose(F, **kwargs):
        x = F.sym.Variable("x")
        y = F.sym.Variable("y")
        z = F.sym.split(x, **kwargs)
        return F.sym.broadcast_sub(F.sym.broadcast_add(z[0], z[2]), y)

    def relay_compose(F, **kwargs):
        x = F.var("x", shape=xshape)
        y = F.var("y", shape=yshape)
        z = F.split(x, **kwargs)
        z = F.subtract(F.add(z[0], z[2]), y)
        return relay.Function(relay.ir_pass.free_vars(z), z)

    mx_sym = mx_compose(mx, num_outputs=3, axis=1)
    from_mx_sym, _ = relay.frontend.from_mxnet(
        mx_sym, shape={"x":xshape, "y":yshape})
    relay_sym = relay_compose(relay, indices_or_sections=3, axis=1)
    compare_graph(from_mx_sym, relay_sym)


if __name__ == "__main__":
    test_mlp()
    test_resnet()
    test_vgg()
    test_multi_outputs()
    test_dqn()
    test_dcgan()
    test_squeezenet()
    test_inception_v3()
