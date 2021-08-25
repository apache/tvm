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
# pylint: disable=invalid-name, unused-argument

import pytest

pytest.importorskip("ethosu.vela")
import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend.contrib import ethosu
from tvm.relay.backend.contrib.ethosu import legalize, preprocess
from tvm.relay.dataflow_pattern import *
from tvm.relay.op.contrib.ethosu import *
import relay_ir_builder


def test_split_indices_legalize():
    def create_graph(axis):
        x = relay.var("x", shape=(1, 50, 50, 3))
        x_relu = relay.nn.relu(x)
        split_o = relay.split(x_relu, [5, 20, 45], axis).tuple_value
        return relay.Function([x], split_o)

    def expected_mod_axis1():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethosu_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 5, 50, 3), float32],\
                                                               Tensor[(1, 15, 50, 3), float32],\
                                                               Tensor[(1, 25, 50, 3), float32],\
                                                               Tensor[(1, 5, 50, 3), float32]) {
          %0 = nn.relu(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 5, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 5, 50, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 5, 0, 0], end=[1, 20, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 15, 50, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 20, 0, 0], end=[1, 45, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 25, 50, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 45, 0, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 5, 50, 3), float32] */;
          (%1, %2, %3, %4)
        }
        """
        return tvm.parser.fromtext(expected_ir_string)

    def expected_mod_axis2():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethosu_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 50, 5, 3), float32],\
                                                               Tensor[(1, 50, 15, 3), float32],\
                                                               Tensor[(1, 50, 25, 3), float32],\
                                                               Tensor[(1, 50, 5, 3), float32]) {
          %0 = nn.relu(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 50, 5, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 5, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 0, 5, 0], end=[1, 50, 20, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 15, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 0, 20, 0], end=[1, 50, 45, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 25, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 0, 45, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 5, 3), float32] */;
          (%1, %2, %3, %4)
        }
        """
        return tvm.parser.fromtext(expected_ir_string)

    mod_axis1 = tvm.IRModule()
    mod_axis1["tvmgen_default_ethosu_main_0"] = create_graph(1)
    mod_axis1 = legalize.LegalizeSplit()(mod_axis1)
    expected_axis1 = expected_mod_axis1()
    tvm.ir.assert_structural_equal(mod_axis1, expected_axis1)

    mod_axis2 = tvm.IRModule()
    mod_axis2["tvmgen_default_ethosu_main_0"] = create_graph(2)
    mod_axis2 = legalize.LegalizeSplit()(mod_axis2)
    expected_axis2 = expected_mod_axis2()
    tvm.ir.assert_structural_equal(mod_axis2, expected_axis2)


def test_split_sections_legalize():
    def create_graph(axis, sections):
        x = relay.var("x", shape=(1, 50, 50, 3))
        x_abs = relay.abs(x)
        split_o = relay.split(x_abs, sections, axis).tuple_value
        outputs = list()
        for section_idx in range(sections):
            split_single_out = relay.TupleGetItem(split_o, section_idx)
            tanh = relay.tanh(split_single_out)
            outputs.append(tanh)
        tuple_out = relay.Tuple(outputs)
        return relay.Function([x], tuple_out)

    def expected_mod_axis1():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethosu_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32]) {
          %0 = abs(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 10, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 10, 0, 0], end=[1, 20, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 20, 0, 0], end=[1, 30, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 30, 0, 0], end=[1, 40, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %5 = strided_slice(%0, begin=[0, 40, 0, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %6 = (%1, %2, %3, %4, %5);
          %7 = %6.0;
          %8 = tanh(%7) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %9 = %6.1;
          %10 = tanh(%9) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %11 = %6.2;
          %12 = tanh(%11) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %13 = %6.3;
          %14 = tanh(%13) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %15 = %6.4;
          %16 = tanh(%15) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          (%8, %10, %12, %14, %16)
        }
        """
        return tvm.parser.fromtext(expected_ir_string)

    def expected_mod_axis2():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethosu_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32]) {
          %0 = abs(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 50, 10, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 0, 10, 0], end=[1, 50, 20, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 0, 20, 0], end=[1, 50, 30, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 0, 30, 0], end=[1, 50, 40, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %5 = strided_slice(%0, begin=[0, 0, 40, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %6 = (%1, %2, %3, %4, %5);
          %7 = %6.0;
          %8 = tanh(%7) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %9 = %6.1;
          %10 = tanh(%9) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %11 = %6.2;
          %12 = tanh(%11) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %13 = %6.3;
          %14 = tanh(%13) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %15 = %6.4;
          %16 = tanh(%15) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          (%8, %10, %12, %14, %16)
        }
        """
        return tvm.parser.fromtext(expected_ir_string)

    mod_axis1 = tvm.IRModule()
    mod_axis1["tvmgen_default_ethosu_main_0"] = create_graph(1, 5)
    mod_axis1 = legalize.LegalizeSplit()(mod_axis1)
    expected_axis1 = expected_mod_axis1()
    tvm.ir.assert_structural_equal(mod_axis1, expected_axis1)

    mod_axis2 = tvm.IRModule()
    mod_axis2["tvmgen_default_ethosu_main_0"] = create_graph(2, 5)
    mod_axis2 = legalize.LegalizeSplit()(mod_axis2)
    expected_axis2 = expected_mod_axis2()
    tvm.ir.assert_structural_equal(mod_axis2, expected_axis2)


def infer_type_function_pass(func):
    mod = tvm.IRModule()
    mod["test"] = func
    mod = relay.transform.InferType()(mod)
    return mod["test"]


def get_shape_expr(in_expr, out_expr):
    main_f = relay.Function([in_expr], out_expr)
    main_f = infer_type_function_pass(main_f)
    shape = [int(i) for i in main_f.body.checked_type.shape]
    return shape


INVERSE_LAYOUT_TRANSFORM_OHWI_MAP = {
    "HWIO": [1, 2, 3, 0],
    "HWOI": [1, 2, 0, 3],
    "OWHI": [0, 1, 2, 3],
}


def test_ethosu_conv2d_legalize():
    def create_graph_single(input_tensor_name, input_tensor_shape, input_tensor_dtype):
        c1_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c1_params.ifm.shape = input_tensor_shape
        c1_params.kernel.shape = (3, 3, c1_params.ifm.shape[3], 32)
        c1_params.strides = (1, 1)
        c1_params.pad = "VALID"
        c1_params.activation = "CLIP"
        c1_params.clip_min = 23
        c1_params.clip_max = 180
        input0 = relay.var(input_tensor_name, shape=c1_params.ifm.shape, dtype=c1_params.ifm.dtype)
        c1, new_params = relay_ir_builder.create_qnn_conv2d(c1_params, input0)
        c1_params.ofm.shape = get_shape_expr(input0, c1)

        f = relay.Function([input0], c1)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod, [c1_params]

    def create_graph_double(input_tensor_name, input_tensor_shape, input_tensor_dtype):
        c1_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c1_params.ifm.shape = input_tensor_shape
        c1_params.kernel.shape = (7, 7, c1_params.ifm.shape[3], 8)
        c1_params.strides = (2, 2)
        c1_params.pad = "VALID"
        c1_params.activation = "CLIP"
        c1_params.clip_min = 10
        c1_params.clip_max = 240
        input0 = relay.var(input_tensor_name, shape=c1_params.ifm.shape, dtype=c1_params.ifm.dtype)
        c1, new_params = relay_ir_builder.create_qnn_conv2d(c1_params, input0)
        c1_params.ofm.shape = get_shape_expr(input0, c1)

        c2_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c2_params.ifm.shape = c1_params.ofm.shape
        c2_params.kernel.shape = (5, 5, c2_params.ifm.shape[3], 16)
        c2_params.strides = (1, 1)
        c2_params.pad = "SAME"
        c2, new_params = relay_ir_builder.create_qnn_conv2d(c2_params, c1)
        c2_params.ofm.shape = get_shape_expr(input0, c2)

        f = relay.Function([input0], c2)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod, [c2_params, c1_params]

    def verify_tensor(tensor_type, expr):
        assert list(tensor_type.shape) == list(expr.checked_type.shape)
        assert str(tensor_type.dtype) == str(expr.checked_type.dtype)

    def verify_linear(ext_func, conv2d_params):
        op = ext_func.body
        for param in conv2d_params:
            verify_tensor(param.ifm, op.args[0])
            verify_tensor(param.ofm, op)

            # This will be in OHWI layout
            weights_ohwi = op.args[1].data.asnumpy()
            weights_layout = str(param.kernel.layout)
            weights = np.transpose(weights_ohwi, INVERSE_LAYOUT_TRANSFORM_OHWI_MAP[weights_layout])
            assert weights.shape == param.kernel.shape
            assert weights.dtype == param.kernel.dtype

            assert list(op.args[2].checked_type.shape)[0] == weights_ohwi.shape[0]

            assert float(op.attrs.ifm_scale) == float(param.ifm.sc.data.asnumpy())
            assert int(op.attrs.ifm_zero_point) == int(param.ifm.zp.data.asnumpy())
            assert int(op.attrs.weight_zero_point) == int(param.kernel.zp.data.asnumpy())
            assert float(op.attrs.ofm_scale) == float(param.ofm.sc.data.asnumpy())
            assert int(op.attrs.ofm_zero_point) == int(param.ofm.zp.data.asnumpy())
            assert int(op.attrs.ofm_channels) == int(weights_ohwi.shape[0])
            assert list(op.attrs.padding) == list(param.pad)
            assert list(op.attrs.strides) == list(param.strides)
            assert list(op.attrs.dilation) == list(param.dilation)
            assert str(op.attrs.activation) == str(param.activation)
            assert int(op.attrs.clip_min) == int(param.clip_min)
            assert int(op.attrs.clip_max) == int(param.clip_max)
            op = op.args[0]

    test_cases = [
        (create_graph_single, ["input", (1, 299, 299, 3), "uint8"]),
        (create_graph_double, ["input", (1, 128, 256, 4), "uint8"]),
    ]
    for test_case in test_cases:
        mod, conv_params = test_case[0](*test_case[1])
        mod = ethosu.partition_for_ethosu(mod)
        mod = legalize.LegalizeEthosUConv2D()(mod)
        verify_linear(mod["tvmgen_default_ethosu_main_0"], conv_params)


def test_ethosu_conv2d_legalize_errors():
    def create_graph_single_unsupported_ifm_layout(
        input_tensor_name, input_tensor_shape, input_tensor_dtype
    ):
        c1_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c1_params.ifm.shape = input_tensor_shape
        c1_params.ifm.layout = "NCHW"
        c1_params.kernel.shape = (3, 3, c1_params.ifm.shape[1], 32)
        c1_params.strides = (1, 1)
        c1_params.pad = "VALID"
        c1_params.activation = "CLIP"
        c1_params.clip_min = 23
        c1_params.clip_max = 180
        input0 = relay.var(input_tensor_name, shape=c1_params.ifm.shape, dtype=c1_params.ifm.dtype)
        c1, new_params = relay_ir_builder.create_qnn_conv2d(c1_params, input0)
        c1_params.ofm.shape = get_shape_expr(input0, c1)

        f = relay.Function([input0], c1)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod, [c1_params]

    test_cases = [
        (create_graph_single_unsupported_ifm_layout, ["input", (1, 3, 299, 299), "uint8"]),
    ]

    for test_case in test_cases:
        mod, conv_params = test_case[0](*test_case[1])
        mod = ethosu.partition_for_ethosu(mod)
        try:
            mod = legalize.LegalizeEthosUConv2D()(mod)
        except Exception as e:
            assert "EthosUCodegenError: Unsupported Layout NCHW" in e.args[0]
