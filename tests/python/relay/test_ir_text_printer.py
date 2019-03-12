import tvm
import tvm.relay.testing
import numpy as np
from tvm import relay


do_print = [False]

def show(text):
    if do_print[0]:
        print("---------------------------")
        print(text)

def test_func():
    x = relay.var("x", shape=(3, 2))
    y = relay.var("y")
    one = relay.const(10e10, dtype="float32")
    z = relay.add(x, one)
    z = relay.add(z, z)
    f = relay.Function([x, y], z)
    show(z.astext())
    show(f.astext())


def test_env():
    x = relay.var("x", "float32")
    y = relay.var("y", "float32")
    z = relay.add(x, y)
    z = relay.add(z, z)
    f = relay.Function([x, y], z)
    env = relay.Module()
    env["myf"] = f
    text = env.astext()
    assert "def @myf" in text
    assert "def @myf" in str(env)
    assert "%1 = add(%0, %0) # ty=float32" in text
    assert "%1 = add(%0, %0) # ty=float32" in str(env)
    show(env.astext(annotate=lambda x: str(x.checked_type.dtype)))
    show(text)


def test_meta_data():
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = relay.var("x", shape=(n, c, h, w))
    w = relay.var("w")
    z = relay.nn.conv2d(x, w,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        channels=2)
    f = relay.Function([x, w], z)
    text = f.astext()
    text_no_meta = str(f)
    assert "channels=2" in text
    assert "channels=2" in text_no_meta
    assert "meta[Variable][0]" in text
    assert "meta[Variable][0]" in text_no_meta
    assert "type_key" in text
    assert "type_key" not in text_no_meta
    show(text)
    show(f)

    text = relay.const([1,2,3]).astext()
    assert "meta[relay.Constant][0]" in text
    show(text)


def test_call_attrs():
    x = relay.var("x")
    # non default args
    z = relay.nn.softmax(x, axis=2)
    assert "axis=2" in z.astext()
    # default args
    z = relay.nn.softmax(x)
    assert "softmax(%x)" in z.astext()
    # non default args
    z = relay.expand_dims(x, axis=2, num_newaxis=2)
    assert "num_newaxis=2" in z.astext()


def test_let_if_scope():
    x = relay.var("x", "float32")
    y = relay.var("y", "float32")
    cond = relay.var("cond", "bool")

    sb = relay.ScopeBuilder()
    with sb.if_scope(cond):
        v1 = sb.let("v", relay.const(1, "float32"))
        v2 = sb.let("v", x)
        sb.ret(relay.subtract(v1, v2))
    with sb.else_scope():
        v3 = relay.var("v")
        let2 = relay.Let(v3, y, v3)
        sb.ret(relay.add(let2, let2))
    result = sb.get()

    f = relay.Function([x, y, cond], result)
    text = f.astext()
    assert text.count("{") == 4
    assert "%cond: bool" in text
    show(f.astext())


def test_variable_name():
    # avoid pure number even if the namehint is pure number
    v1 = relay.var("1")
    assert "%v1" in v1.astext()


def test_mlp():
    net, params = tvm.relay.testing.mlp.get_workload(batch_size=1)
    net.astext()


def test_resnet():
    net, params = tvm.relay.testing.resnet.get_workload(batch_size=1)
    net.astext()


def test_mobilenet():
    net, params = tvm.relay.testing.mobilenet.get_workload(batch_size=1)
    net.astext()


def test_dqn():
    net, params = tvm.relay.testing.dqn.get_workload(batch_size=1)
    net.astext()


def test_dcgan():
    net, params = tvm.relay.testing.dcgan.get_workload(batch_size=1)
    net.astext()


def test_lstm():
    net, params = tvm.relay.testing.lstm.get_workload(4, 4)
    net.astext()

def test_inception_v3():
    net, params = tvm.relay.testing.inception_v3.get_workload(batch_size=1)
    net.astext()

def test_squeezenet():
    for version in ['1.0', '1.1']:
        net, params = tvm.relay.testing.squeezenet.get_workload(batch_size=1, version=version)
        net.astext()

def test_vgg():
    net, params = tvm.relay.testing.vgg.get_workload(batch_size=1)
    net.astext()

def test_densenet():
    net, params = tvm.relay.testing.densenet.get_workload(batch_size=1)
    net.astext()


if __name__ == "__main__":
    do_print[0] = True
    test_resnet()
    test_mobilenet()
    test_mlp()
    test_dqn()
    test_dcgan()
    test_squeezenet()
    test_inception_v3()
    test_vgg()
    test_densenet()
    test_func()
    test_env()
    test_meta_data()
    test_call_attrs()
    test_let_if_scope()
    test_variable_name()
