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
"""Test legalize pass"""
import numpy as np
import tvm
from tvm import te

from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay import transform, analysis
from tvm.relay.testing.temp_op_attr import TempOpAttr

def alpha_equal(x, y):
    """
    Wrapper around alpha equality which ensures that
    the hash function respects equality.
    """
    x = x['main']
    y = y['main']
    return tvm.ir.structural_equal(x, y) and \
            tvm.ir.structural_hash(x) == tvm.ir.structural_hash(y)

def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def test_qnn_legalize():
    """Test directly replacing an operator with a new one"""
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype='int8')
        y = relay.qnn.op.requantize(x,
                                    input_scale=relay.const(1, 'float32'),
                                    input_zero_point=relay.const(0, 'int32'),
                                    output_scale=relay.const(1, 'float32'),
                                    output_zero_point=relay.const(0, 'int32'),
                                    out_dtype='int8')
        y = relay.Function([x], y)
        return y

    def legalize_qnn_requantize(attrs, inputs, types):
        data = inputs[0]
        data = relay.add(relay.const(0, 'int8'), data)
        y = relay.qnn.op.requantize(data,
                                    input_scale=relay.const(1, 'float32'),
                                    input_zero_point=relay.const(0, 'int32'),
                                    output_scale=relay.const(1, 'float32'),
                                    output_zero_point=relay.const(0, 'int32'),
                                    out_dtype='int8')
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype='int8')
        y = relay.add(relay.const(0, 'int8'), x)
        z = relay.qnn.op.requantize(y,
                                    input_scale=relay.const(1, 'float32'),
                                    input_zero_point=relay.const(0, 'int32'),
                                    output_scale=relay.const(1, 'float32'),
                                    output_zero_point=relay.const(0, 'int32'),
                                    out_dtype='int8')
        z = relay.Function([x], z)
        return z

    a = before()

    with TempOpAttr("qnn.requantize", "FTVMQnnLegalize", legalize_qnn_requantize):

        # Check that Relay Legalize does not change the graph.
        a = run_opt_pass(a, relay.transform.Legalize())
        b = run_opt_pass(before(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

        # Check that QNN Legalize modifies the graph.
        a = run_opt_pass(a, relay.qnn.transform.Legalize())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_legalize_qnn_conv2d():
    def _get_mod(data_dtype, kernel_dtype):
        data_shape = (1, 64, 256, 256)
        kernel_shape = (128, 64, 3, 3)
        data = relay.var("data", shape=data_shape,
                dtype=data_dtype)
        kernel = relay.var("kernel", shape=kernel_shape,
                dtype=kernel_dtype)
        func = relay.qnn.op.conv2d(
                data, kernel,
                input_zero_point=relay.const(1, 'int32'),
                kernel_zero_point=relay.const(1, 'int32'),
                input_scale=relay.const(1.0, 'float32'),
                kernel_scale=relay.const(1.0, 'float32'),
                kernel_size=(3, 3),
                channels=kernel_shape[0],
                strides=(1, 1),
                dilation=(1, 1),
                out_dtype='int32',
                data_layout='NCHW',
                kernel_layout='OIHW')

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(mod)
        return mod

    # Check uint8 x uint8 and int8 x int8 transformation
    for dtype in ('uint8', 'int8'):
        mod = _get_mod(dtype, dtype)

        #############################################################
        # Check transformations for platforms with fast Int8 support.
        #############################################################
        # Check that Intel VNNI gets picked up.
        with tvm.target.create('llvm -mcpu=skylake-avx512'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert 'cast' in legalized_mod.astext() and "qnn.conv2d" in legalized_mod.astext()

        # Since same dtype, there should not be any transformation
        with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+v8.2a,+dotprod'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert tvm.ir.structural_equal(mod, legalized_mod)

        ################################################################
        # Check transformations for platforms without fast Int8 support.
        ################################################################
        # Older Intel versions.
        with tvm.target.create('llvm'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

        # Older ARM vesions.
        with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Check uint8 x int8 transformation
    mod = _get_mod('uint8', 'int8')
    #############################################################
    # Check transformations for platforms with fast Int8 support.
    #############################################################
    # Check no transformation for Intel VNNI.
    with tvm.target.create('llvm -mcpu=skylake-avx512'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert tvm.ir.structural_equal(mod, legalized_mod)

    # ARM - so check that transformation has happened.
    with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+v8.2a,+dotprod'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn.conv2d" in legalized_mod.astext()

    ################################################################
    # Check transformations for platforms without fast Int8 support.
    ################################################################
    # Older Intel versions.
    with tvm.target.create('llvm'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Older ARM vesions.
    with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    ###########################################
    # Check transformations for CUDA platforms.
    ###########################################
    with tvm.target.create('cuda'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn" in legalized_mod.astext()


def test_qnn_legalize_qnn_dense():
    def _get_mod(data_dtype, kernel_dtype):
        data_shape = (10, 3)
        kernel_shape = (20, 3)
        data = relay.var("data", shape=data_shape,
                dtype=data_dtype)
        kernel = relay.var("kernel", shape=kernel_shape,
                dtype=kernel_dtype)
        func = relay.qnn.op.dense(
                data, kernel,
                input_zero_point=relay.const(1, 'int32'),
                kernel_zero_point=relay.const(1, 'int32'),
                input_scale=relay.const(1, 'float32'),
                kernel_scale=relay.const(1, 'float32'),
                units=kernel_shape[0],
                out_dtype='int32')

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(mod)
        return mod

    # Check uint8 x uint8 and int8 x int8 transformation
    for dtype in ('uint8', 'int8'):
        mod = _get_mod(dtype, dtype)

        #############################################################
        # Check transformations for platforms with fast Int8 support.
        #############################################################
        # Check that Intel VNNI gets picked up.
        with tvm.target.create('llvm -mcpu=skylake-avx512'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert 'cast' in legalized_mod.astext() and "qnn.dense" in legalized_mod.astext()

        # Since same dtype, there should not be any transformation
        with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+v8.2a,+dotprod'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert tvm.ir.structural_equal(mod, legalized_mod)

        ################################################################
        # Check transformations for platforms without fast Int8 support.
        ################################################################
        # Older Intel versions.
        with tvm.target.create('llvm'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

        # Older ARM vesions.
        with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu'):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Check uint8 x int8 transformation
    mod = _get_mod('uint8', 'int8')
    #############################################################
    # Check transformations for platforms with fast Int8 support.
    #############################################################
    # Check no transformation for Intel VNNI.
    with tvm.target.create('llvm -mcpu=skylake-avx512'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert tvm.ir.structural_equal(mod, legalized_mod)

    # ARM - so check that transformation has happened.
    with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+v8.2a,+dotprod'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn.dense" in legalized_mod.astext()

    ################################################################
    # Check transformations for platforms without fast Int8 support.
    ################################################################
    # Older Intel versions.
    with tvm.target.create('llvm'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Older ARM vesions.
    with tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    ###########################################
    # Check transformations for CUDA platforms.
    ###########################################
    with tvm.target.create('cuda'):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert 'cast' in legalized_mod.astext() and "qnn" in legalized_mod.astext()


if __name__ == "__main__":
    test_qnn_legalize()
    test_qnn_legalize_qnn_conv2d()
    test_qnn_legalize_qnn_dense()
