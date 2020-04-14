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
from tvm import te

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay import transform, analysis


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_no_convert_layout():
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

    def expected():
        return before()

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var('weight', shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(x, weight,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC',
                            kernel_layout='HWIO')
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var('weight', shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, 'NHWC', 'NCHW')
        weight = relay.layout_transform(weight, 'HWIO', 'OIHW')
        y = relay.nn.conv2d(x, weight,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, 'NCHW', 'NHWC')
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_bias_pool_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1),
                            data_layout='NHWC', kernel_layout='HWIO')
        y = relay.nn.bias_add(y, bias, axis=3)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout='NHWC')
        y = relay.cast(y, 'int32')
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, 'NHWC', 'NCHW')
        weight = relay.layout_transform(weight, 'HWIO', 'OIHW')
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))

        bias = relay.expand_dims(bias, axis=0, num_newaxis=3)
        bias = relay.layout_transform(bias, 'NHWC', 'NCHW')
        y = relay.add(y, bias)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.cast(y, 'int32')
        y = relay.layout_transform(y, 'NCHW', 'NHWC')
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_concat_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1', shape=(3, 3, 64, 64))
        weight2 = relay.var('weight2', shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(x, weight1,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC',
                            kernel_layout='HWIO')
        y1 = relay.nn.conv2d(y, weight2,
                             channels=64,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NHWC',
                             kernel_layout='HWIO')
        ret = relay.concatenate([y, y1], axis=3)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1', shape=(3, 3, 64, 64))
        weight2 = relay.var('weight2', shape=(3, 3, 64, 64))
        weight1 = relay.layout_transform(weight1, 'HWIO', 'OIHW')
        weight2 = relay.layout_transform(weight2, 'HWIO', 'OIHW')
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(y, weight1,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y1 = relay.nn.conv2d(y, weight2,
                             channels=64,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_dual_path_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1', shape=(3, 3, 64, 32))
        weight2 = relay.var('weight2', shape=(3, 3, 32, 32))
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC',
                            kernel_layout='HWIO')
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             data_layout='NHWC',
                             kernel_layout='HWIO')
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.batch_flatten(y)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1', shape=(3, 3, 64, 32))
        weight2 = relay.var('weight2', shape=(3, 3, 32, 32))
        weight1 = relay.layout_transform(weight1, 'HWIO', 'OIHW')
        weight2 = relay.layout_transform(weight2, 'HWIO', 'OIHW')
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(y, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2,
                             channels=32,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y1 = relay.layout_transform(y1, "NCHW", "NHWC")
        y2 = relay.layout_transform(y, "NCHW", "NHWC")
        y2 = relay.nn.batch_flatten(y2)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_bn_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1', shape=(3, 3, 64, 32))
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC',
                            kernel_layout='HWIO')
        gamma = relay.var("gamma")
        beta = relay.var("beta")
        mean = relay.var("mean")
        variance = relay.var("variance")
        y, _, _ = relay.nn.batch_norm(y , gamma, beta, mean, variance, axis=3)
        return relay.Function(analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))

    # Check that there is only 1 NHWC to NCHW transform.
    has_lt = list()
    find_op = lambda x : \
            has_lt.append(isinstance(x, tvm.relay.expr.Call) and x.op.name == "layout_transform" \
            and x.attrs.src_layout == 'NCHW' and x.attrs.dst_layout == 'NHWC')
    relay.analysis.post_order_visit(a, find_op)
    has_lt = list(filter(lambda x: x, has_lt))
    assert len(has_lt) == 1


def test_resnet_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var('weight1', shape=(3, 3, 64, 32))
        weight2 = relay.var('weight2', shape=(1, 1, 64, 32))
        y = relay.nn.conv2d(x, weight1,
                            channels=32,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            data_layout='NHWC',
                            kernel_layout='HWIO')
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2,
                             channels=32,
                             kernel_size=(1, 1),
                             data_layout='NHWC',
                             kernel_layout='HWIO')
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y, layout='NHWC')
        return relay.Function(analysis.free_vars(y), y)

    def expected():
        x = relay.var("x", shape=(1,56, 56, 64))
        weight1 = relay.var('weight1', shape=(3, 3, 64, 32))
        weight2 = relay.var('weight2', shape=(1, 1, 64, 32))
        weight1 = relay.layout_transform(weight1, 'HWIO', 'OIHW')
        weight2 = relay.layout_transform(weight2, 'HWIO', 'OIHW')
        x = relay.layout_transform(x, "NHWC", "NCHW")
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
        y = relay.layout_transform(y, "NCHW", "NHWC")
        return relay.Function(analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_scalar_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1),
                            data_layout='NHWC', kernel_layout='HWIO')
        y = relay.add(y, relay.const(1, "float32"))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, 'NHWC', 'NCHW')
        w = relay.layout_transform(w, 'HWIO', 'OIHW')
        y = relay.nn.conv2d(x, w,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.add(y, relay.const(1.0, "float32"))

        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_bn_convert_layout():
    """ Check that layout transforms are propagated through bn. """
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1),
                            data_layout='NHWC', kernel_layout='HWIO')

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var, axis=3)
        y = relay.nn.relu(y[0])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, 'NHWC', 'NCHW')
        w = relay.layout_transform(w, 'HWIO', 'OIHW')
        y = relay.nn.conv2d(x, w,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var, axis=1)
        y = relay.nn.relu(y[0])
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_conv_requantize_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype='int8')
        weight = relay.var('weight', shape=(3, 3, 64, 64), dtype='int8')
        y = relay.qnn.op.conv2d(x, weight,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                data_layout='NHWC',
                                kernel_layout='HWIO')
        y = relay.qnn.op.requantize(y,
                                    relay.const(1, 'float32'),
                                    relay.const(1, 'int32'),
                                    relay.const(1, 'float32'),
                                    relay.const(1, 'int32'),
                                    out_dtype='int32')
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype='int8')
        weight = relay.var('weight', shape=(3, 3, 64, 64), dtype='int8')
        x = relay.layout_transform(x, 'NHWC', 'NCHW')
        weight = relay.layout_transform(weight, 'HWIO', 'OIHW')
        y = relay.qnn.op.conv2d(x, weight,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        y = relay.qnn.op.requantize(y,
                                    relay.const(1, 'float32'),
                                    relay.const(1, 'int32'),
                                    relay.const(1, 'float32'),
                                    relay.const(1, 'int32'),
                                    axis=1,
                                    out_dtype='int32')
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, 'NCHW', 'NHWC')
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_conv_concat_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype='int8')
        weight1 = relay.var('weight1', shape=(3, 3, 64, 64), dtype='int8')
        weight2 = relay.var('weight2', shape=(3, 3, 64, 64), dtype='int8')
        y = relay.qnn.op.conv2d(x, weight1,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                data_layout='NHWC',
                                kernel_layout='HWIO')
        y1 = relay.qnn.op.conv2d(y, weight2,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                data_layout='NHWC',
                                kernel_layout='HWIO')
        y = relay.cast(y, 'int8')
        y1 = relay.cast(y, 'int8')
        ret = relay.qnn.op.concatenate([y, y1],
                                       [relay.const(1, 'float32'), relay.const(1, 'float32')],
                                       [relay.const(1, 'int32'), relay.const(1, 'int32')],
                                       relay.const(1, 'float32'),
                                       relay.const(1, 'int32'),
                                       axis=3)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype='int8')
        weight1 = relay.var('weight1', shape=(3, 3, 64, 64), dtype='int8')
        weight2 = relay.var('weight2', shape=(3, 3, 64, 64), dtype='int8')
        weight1 = relay.layout_transform(weight1, 'HWIO', 'OIHW')
        weight2 = relay.layout_transform(weight2, 'HWIO', 'OIHW')
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.qnn.op.conv2d(y, weight1,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        y1 = relay.qnn.op.conv2d(y, weight2,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        y = relay.cast(y, 'int8')
        y1 = relay.cast(y, 'int8')
        ret = relay.qnn.op.concatenate([y, y1],
                                      [relay.const(1, 'float32'), relay.const(1, 'float32')],
                                      [relay.const(1, 'int32'), relay.const(1, 'int32')],
                                      relay.const(1, 'float32'),
                                      relay.const(1, 'int32'),
                                      axis=1)
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_conv_add_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype='int8')
        weight1 = relay.var('weight1', shape=(3, 3, 64, 64), dtype='int8')
        weight2 = relay.var('weight2', shape=(3, 3, 64, 64), dtype='int8')
        y = relay.qnn.op.conv2d(x, weight1,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                data_layout='NHWC',
                                kernel_layout='HWIO')
        y1 = relay.qnn.op.conv2d(y, weight2,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                data_layout='NHWC',
                                kernel_layout='HWIO')
        y = relay.cast(y, 'int8')
        y1 = relay.cast(y, 'int8')
        ret = relay.qnn.op.add(y, y1,
                               relay.const(1, 'float32'),
                               relay.const(1, 'int32'),
                               relay.const(1, 'float32'),
                               relay.const(1, 'int32'),
                               relay.const(1, 'float32'),
                               relay.const(1, 'int32'))
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype='int8')
        weight1 = relay.var('weight1', shape=(3, 3, 64, 64), dtype='int8')
        weight2 = relay.var('weight2', shape=(3, 3, 64, 64), dtype='int8')
        weight1 = relay.layout_transform(weight1, 'HWIO', 'OIHW')
        weight2 = relay.layout_transform(weight2, 'HWIO', 'OIHW')
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.qnn.op.conv2d(y, weight1,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        y1 = relay.qnn.op.conv2d(y, weight2,
                                relay.const(1, 'int32'),
                                relay.const(1, 'int32'),
                                relay.const(1, 'float32'),
                                relay.const(1, 'float32'),
                                channels=64,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        y = relay.cast(y, 'int8')
        y1 = relay.cast(y, 'int8')
        ret = relay.qnn.op.add(y, y1,
                               relay.const(1, 'float32'),
                               relay.const(1, 'int32'),
                               relay.const(1, 'float32'),
                               relay.const(1, 'int32'),
                               relay.const(1, 'float32'),
                               relay.const(1, 'int32'))
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout('NCHW'))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


if __name__ == "__main__":
    test_no_convert_layout()
    test_conv_convert_layout()
    test_conv_bias_pool_convert_layout()
    test_conv_concat_convert_layout()
    test_dual_path_convert_layout()
    test_bn_convert_layout()
    test_resnet_convert_layout()
    test_scalar_convert_layout()
    test_conv_bn_convert_layout()
    test_qnn_conv_requantize_convert_layout()
    test_qnn_conv_concat_convert_layout()
    test_qnn_conv_add_convert_layout()
