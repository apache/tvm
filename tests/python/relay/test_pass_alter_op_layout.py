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
"""Test alter op layout pass"""
import tvm

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay import transform, analysis


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = relay.Module.from_expr(expr)
    seq = transform.Sequential(passes)
    with transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_alter_op():
    """Test directly replacing an operator with a new one"""
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=100)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, relay.multiply(weight, relay.const(2.0, "float32")),
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    a = before()
    a = run_opt_pass(a, transform.AlterOpLayout())
    b = run_opt_pass(expected(), transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


def test_alter_return_none():
    """Test doing nothing by returning 'None' """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.global_max_pool2d(x)
        y = relay.Function([x], y)
        return y

    called = [False]

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.global_max_pool2d", level=101)
    def alter_conv2d(attrs, inputs, tinfos):
        called[0] = True
        return None

    a = before()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = before()
    b = run_opt_pass(b, transform.InferType())
    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)
    assert(called[0])


def test_alter_layout():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias")
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.bias_add(y, bias)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.cast(y, 'int32')
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=102)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        new_attrs['kernel_layout'] = 'OIHW16i'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(64, 64, 3, 3))

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        w = relay.layout_transform(weight, "OIHW", "OIHW16i")
        y = relay.nn.conv2d(y, w,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            kernel_layout="OIHW16i",
                            data_layout="NCHW16c")
        b = relay.expand_dims(bias, axis=1, num_newaxis=2)
        b = relay.expand_dims(b, axis=0, num_newaxis=1)
        b = relay.layout_transform(b, "NCHW", "NCHW16c")
        y = relay.add(y, b)

        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NCHW16c")
        y = relay.cast(y, 'int32')
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.CanonicalizeOps(),
                         transform.AlterOpLayout()])

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_dual_path():
    """
    Test alternating the layout with two outputs.
    One path continues to use the new layout while one path fall backs to old layout.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.batch_flatten(y)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=103)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NCHW16c')
        y1 = relay.nn.relu(y1)
        y1 = relay.layout_transform(y1, "NCHW16c", "NCHW")
        y2 = relay.layout_transform(y, "NCHW16c", "NCHW")
        y2 = relay.nn.batch_flatten(y2)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

def test_alter_layout_resnet():
    """Test alternating the layout of a residual block
    This also tests the elimination of duplicated transformation.
    If a same transformation applies to a same node twice, only one transformation will be created.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2,
                             channels=32,
                             kernel_size=(1, 1))
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y)
        return relay.Function(analysis.free_vars(y), y)

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=104)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2,
                             channels=32,
                             kernel_size=(1, 1),
                             data_layout='NCHW16c')
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y, layout="NCHW16c")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        return relay.Function(analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_broadcast_op():
    """Test boradcast operators """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        scale = relay.var("scale", shape=(64, 1, 1))
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.bias_add(y, bias) # test broadcasting to lhs
        y = relay.multiply(scale, y)         # test broadcasting to rhs
        y = relay.Function(analysis.free_vars(y), y)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=105)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        scale = relay.var("scale", shape=(64, 1, 1))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        bias = relay.expand_dims(bias, 1, 2)
        bias = relay.expand_dims(bias, 0, 1)
        bias = relay.layout_transform(bias, "NCHW", "NCHW16c")
        scale = relay.expand_dims(scale, 0, 1)
        scale = relay.layout_transform(scale, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.add(y, bias)          # test broadcasting to lhs
        y = relay.multiply(scale, y)      # test broadcasting to rhs
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.CanonicalizeOps(),
                         transform.AlterOpLayout()])

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

def test_alter_layout_scalar():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.add(y, relay.const(1, "float32"))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=106)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        w = relay.var("weight")

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, w,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.add(y, relay.const(1.0, "float32"))

        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.CanonicalizeOps(),
                         transform.AlterOpLayout()])

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_concatenate():
    """ NCHW, NHWC and corner case concatenate layout transform."""
    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=107)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    # NCHW layout transformation.
    def before_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        ret = relay.concatenate([y, y1], axis=1)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NCHW16c')
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before_nchw()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected_nchw()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

    # NHWC layout transformation.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC')
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NHWC')
        ret = relay.concatenate([y, y1], axis=3)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1')
        weight2 = relay.var('weight2')
        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NCHW16c')
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW16c", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before_nhwc()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected_nhwc()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_nchw_upsamping_op():
    """Test upsamping operators """
    def before():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var('weight', shape=(32, 32, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.upsampling(y, scale=2)
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=108)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.nn.upsampling(y, scale=2, layout="NCHW16c")
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2), layout='NCHW16c')
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.CanonicalizeOps(),
                         transform.AlterOpLayout()])

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_strided_slice():
    """Test rewriting strided_slice during alter_iop_layout"""
    def before():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var('weight', shape=(32, 32, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.strided_slice(y, begin=[0, 16], end=[None, None])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=109)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW4c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW4c")
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1),
                            data_layout="NCHW4c")
        y = relay.strided_slice(y, begin=[0, 4], end=[None, 8])
        y = relay.layout_transform(y, "NCHW4c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.CanonicalizeOps(),
                         transform.AlterOpLayout()])

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

def test_alter_layout_depthwise_conv2d():
    """Test depthwise_conv2d operator"""
    def before():
        x = relay.var("x", shape=(1, 32, 56, 56))
        w = relay.var("w", shape=(32, 1, 3, 3))
        y = relay.nn.conv2d(x, w, padding=(1, 1), channels=32, kernel_size=(3, 3), groups=32)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    import topi
    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=110)
    def alter_conv2d(attrs, inputs, tinfos):
        with tvm.target.create("llvm"):
            return topi.nn.conv2d_alter_layout(attrs, inputs, tinfos, relay)

    def expected():
        x = relay.var("x", shape=(1, 32, 56, 56))
        w = relay.var("w", shape=(32, 1, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NCHW8c")
        w = relay.layout_transform(w, "OIHW", "OIHW1i8o")
        y = relay.nn.contrib_depthwise_conv2d_nchwc(x, w, padding=(1, 1), channels=32, kernel_size=(3, 3),
                                                    groups=32, data_layout="NCHW8c", kernel_layout="OIHW1i8o",
                                                    out_layout="NCHW8c")
        y = relay.layout_transform(y, "NCHW8c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.CanonicalizeOps(),
                         transform.AlterOpLayout()])

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert(analysis.alpha_equal(a, b))

def test_alter_layout_prelu():
    """Test PRelu operator"""
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight")
        alpha = relay.var("alpha", relay.IncompleteType())
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.prelu(y, alpha)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=111)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        w = relay.var("weight")
        alpha = relay.var("alpha", relay.IncompleteType())

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, w,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.nn.prelu(y, alpha)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])

    b = expected()
    b = run_opt_pass(b, transform.InferType())

    assert(analysis.alpha_equal(a, b))


def test_alter_layout_pool():
    """ Check NCHW, NHWC pool layout conversion"""
    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=113)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    # Check NCHW conversion.
    def before_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1))
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1), layout='NCHW16c')
        ret = relay.layout_transform(ret, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before_nchw()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected_nchw()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

    # Check NHWC conversion.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC')
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1), layout='NHWC')
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1')
        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1), layout='NCHW16c')
        ret = relay.layout_transform(ret, "NCHW16c", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before_nhwc()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected_nhwc()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_sum():
    """ Check NCHW, NHWC sum layout conversion"""
    # Register alter op layout. "level" is used to override the previously registered functions.
    @register_alter_op_layout("nn.conv2d", level=114)
    def alter_conv2d(attrs, inputs, tinfos):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = 'NCHW16c'
        return relay.nn.conv2d(data, weight, **new_attrs)

    # Check NCHW conversion.
    def before_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        ret = relay.sum(y, axis=1, keepdims=True)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var('weight1')
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        ret = relay.layout_transform(y, "NCHW16c", "NCHW")
        ret = relay.sum(ret, axis=[1], keepdims=True)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before_nchw()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected_nchw()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

    # Check NHWC conversion.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1')
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC')
        ret = relay.sum(y, axis=3, keepdims=True)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1')
        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout="NCHW16c")
        ret = relay.layout_transform(y, "NCHW16c", "NCHW")
        ret = relay.sum(ret, axis=[1], keepdims=True)
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before_nhwc()
    a = run_opt_pass(a, transform.AlterOpLayout())

    b = expected_nhwc()
    b = run_opt_pass(b, transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)


if __name__ == "__main__":
    test_alter_op()
    test_alter_return_none()
    test_alter_layout()
    test_alter_layout_dual_path()
    test_alter_layout_resnet()
    test_alter_layout_broadcast_op()
    test_alter_layout_scalar()
    test_alter_layout_concatenate()
    test_alter_layout_nchw_upsamping_op()
    test_alter_layout_strided_slice()
    test_alter_layout_depthwise_conv2d()
    test_alter_layout_prelu()
    test_alter_layout_pool()
    test_alter_layout_sum()
