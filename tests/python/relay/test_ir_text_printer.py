# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import relay
import tvm.relay.testing
import numpy as np
from tvm.relay import Expr
from tvm.relay.analysis import alpha_equal, assert_alpha_equal, assert_graph_equal, free_vars

do_print = [False]

SEMVER = "v0.0.4\n"

def astext(p, unify_free_vars=False):
    txt = p.astext()
    if isinstance(p, Expr) and free_vars(p):
        return txt
    x = relay.fromtext(txt)
    if unify_free_vars:
        assert_graph_equal(x, p)
    else:
        assert_alpha_equal(x, p)
    return txt

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
    show(astext(z))
    show(astext(f))


def test_env():
    x = relay.var("x", "float32")
    y = relay.var("y", "float32")
    z = relay.add(x, y)
    z = relay.add(z, z)
    f = relay.Function([x, y], z)
    env = relay.Module()
    env["myf"] = f
    text = astext(env)
    assert "def @myf" in text
    assert "def @myf" in str(env)
    assert "add(%0, %0) /* ty=float32 */" in text
    assert "add(%0, %0) /* ty=float32 */" in str(env)
    show(env.astext(annotate=lambda x: str(x.checked_type.dtype) if type(x) == relay.Call else ""))
    show(text)


def test_meta_data():
    n, c, h, w = tvm.size_var("n"), 10, 224, 224
    x = relay.var("x", shape=(n, c, h, w))
    w = relay.var("w")
    z = relay.nn.conv2d(x, w,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        channels=2)
    f = relay.Function([x, w], z)
    text = astext(f, unify_free_vars=True)
    text_no_meta = str(f)
    assert "channels=2" in text
    assert "channels=2" in text_no_meta
    assert "meta[SizeVar][0]" in text
    assert "meta[SizeVar][0]" in text_no_meta
    assert "type_key" in text
    assert "type_key" not in text_no_meta

    text = astext(relay.const([1,2,3]))
    assert "meta[relay.Constant][0]" in text


def test_call_attrs():
    x = relay.var("x")
    # non default args
    z = relay.nn.softmax(x, axis=2)
    assert "axis=2" in astext(z)
    # default args
    z = relay.nn.softmax(x)
    assert "softmax(%x)" in astext(z)
    # non default args
    z = relay.expand_dims(x, axis=2, num_newaxis=2)
    assert "num_newaxis=2" in astext(z)


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
    text = astext(f)
    assert text.count("{") == 3
    assert "%cond: bool" in text
    show(astext(f))


def test_variable_name():
    # avoid pure number even if the namehint is pure number
    v1 = relay.var("1")
    assert "%v1" in astext(v1)


def test_mlp():
    net, params = tvm.relay.testing.mlp.get_workload(batch_size=1)
    astext(net)


def test_resnet():
    net, params = tvm.relay.testing.resnet.get_workload(batch_size=1)
    astext(net)


def test_mobilenet():
    net, params = tvm.relay.testing.mobilenet.get_workload(batch_size=1)
    astext(net)


def test_dqn():
    net, params = tvm.relay.testing.dqn.get_workload(batch_size=1)
    astext(net)


def test_dcgan():
    net, params = tvm.relay.testing.dcgan.get_workload(batch_size=1)
    astext(net)


def test_lstm():
    net, params = tvm.relay.testing.lstm.get_workload(1, 1)
    astext(net)

    net, params = tvm.relay.testing.lstm.get_workload(4, 4)
    astext(net)

def test_inception_v3():
    net, params = tvm.relay.testing.inception_v3.get_workload(batch_size=1)
    astext(net)


def test_squeezenet():
    for version in ['1.0', '1.1']:
        net, params = tvm.relay.testing.squeezenet.get_workload(batch_size=1, version=version)
        astext(net)


def test_vgg():
    net, params = tvm.relay.testing.vgg.get_workload(batch_size=1)
    astext(net)


def test_densenet():
    net, params = tvm.relay.testing.densenet.get_workload(batch_size=1)
    astext(net)


def test_call_node_order():
    x = relay.var("x")
    y = relay.var("y")
    prog = relay.Call(relay.Function([x], x), [relay.Call(relay.Function([y], y), [relay.const(1)])])
    assert astext(prog) == SEMVER + \
        ("%0 = fn (%y) {\n"
         "  %y\n"
         "};\n"
         "%1 = %0(1);\n"
         "%2 = fn (%x) {\n"
         "  %x\n"
         "};\n"
         "%2(%1)")


def test_let_inlining():
    tup = relay.Tuple([relay.const(0), relay.const(0)])
    x = relay.var("x")
    assert astext(relay.Let(x, tup, tup)) == SEMVER + \
        ("%0 = (0, 0);\n"
         "let %x = %0;\n"
         "%0")

    assert astext(relay.Let(x, tup, x)) == SEMVER + \
        ("let %x = (0, 0);\n"
         "%x")


def test_zeros():
    x = relay.op.zeros([], "float32")
    astext(x)


def test_unapplied_constructor():
    type_def_str = r"""
type List[A] {
  Cons(A, List[A]),
  Nil,
}
    """
    main_def_str = r"""
def @main[A]() -> fn (A, List[A]) -> List[A] {
  Cons
}
    """
    mod = relay.fromtext(SEMVER + type_def_str + main_def_str)
    mod_str = str(mod)
    # ensure constructors are printed correctly in type definitions (with their
    # signature) and as exprs (without their signature)
    assert type_def_str.strip() in mod_str
    assert main_def_str.strip() in mod_str


if __name__ == "__main__":
    do_print[0] = True
    test_lstm()
    test_zeros()
    test_meta_data()
    test_let_inlining()
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
    test_call_attrs()
    test_let_if_scope()
    test_variable_name()
    test_call_node_order()
    test_unapplied_constructor()
