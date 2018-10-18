import tvm
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
    env = relay.Environment()
    env.add("myf", f)
    text = env.astext()
    assert "def @myf" in text
    assert "%1 = add(%0, %0) # ty=float32" in text
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
    assert "channels=2" in text
    assert "meta.Variable(id=0)" in text
    show(text)

    text = relay.const([1,2,3]).astext()
    assert "meta.relay.Constant(id=0)" in text
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
    v1 = relay.var("v")
    v2 = relay.var("v", "float32")
    then_branch = relay.Let(
        v1, relay.const(1, "float32"),
        relay.Let(v2, x, relay.subtract(v1, v2)))
    v3 = relay.var("v")
    let2 = relay.Let(v3, y, v3)
    else_branch = relay.add(let2, let2)
    result = relay.If(cond, then_branch, else_branch)
    f = relay.Function([x, y, cond], result)
    text = f.astext()
    assert text.count("{") == 4
    assert "%cond: bool" in text
    show(f.astext())


if __name__ == "__main__":
    do_print[0] = True
    test_let_if_scope()
    test_func()
    test_env()
    test_meta_data()
    test_call_attrs()
