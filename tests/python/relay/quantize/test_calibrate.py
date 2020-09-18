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
from tvm.data import RandomDatasetManager
from tvm.relay.transform.quantize import (
    Quantizer,
    QuantizerPattern,
    QuantizationCalibrator,
    Conv2DPattern,
    Conv2DBiasAddPattern,
    DensePattern,
    DenseBiasAddPattern,
    AddPattern,
    MultiplyPattern,
    CalibrationCallback,
    GlobalCalibrationCallback,
    AverageMaxCalibrationCallback,
    AverageMaxPerChannelConv2DPattern,
    AverageMaxPerChannelConv2DBiasAddPattern,
    AverageMaxPerChannelDensePattern,
    AverageMaxPerChannelDenseBiasAddPattern,
)
from test_quantize import (
    create_conv2d_func,
    create_q_conv2d_func,
    create_conv2d_bias_func,
    create_q_conv2d_bias_func,
    create_dense_func,
    create_q_dense_func,
    create_dense_bias_func,
    create_q_dense_bias_func,
    create_add_func,
    create_q_add_func,
    create_mul_func,
    create_q_mul_func,
)
from tvm.relay.frontend.common import infer_type

import numpy as np

# Calls all the methods of CalibrationCallback to make sure they work OK


class TestCalibrationCallback(CalibrationCallback):
    def __init__(self):
        self.scale_value = np.array(2).astype("float32")
        self.zp_value = np.array(0.5).astype("int32")

    def calibrate_pattern(self, calibration_info):
        scale_zp_values = {}

        for i in range(len(calibration_info.input_scale_zps)):
            scale_name = calibration_info.input_scale_zps[i][0].name_hint
            scale_zp_values[scale_name] = self.scale_value
            zp_name = calibration_info.input_scale_zps[i][1].name_hint
            scale_zp_values[zp_name] = self.zp_value

        inputs, _ = calibration_info.dataset_manager.get_next_batch()

        calibration_info.get_unquantized_layer_inputs(inputs)
        calibration_info.get_unquantized_layer_output(inputs)
        calibration_info.get_quantized_layer_inputs(inputs, scale_zp_values)
        calibration_info.get_quantized_layer_output(inputs, scale_zp_values)

        return scale_zp_values


def test_calibrate(quantizer, quantized_func, params, dataset_manager):
    calibrator = QuantizationCalibrator(quantizer, dataset_manager=dataset_manager)
    calibrated_func = calibrator.calibrate()

    quantized_func = relay.build_module.bind_params_by_name(
        quantized_func, calibrator.calibration_info.scale_zp_value_map
    )
    quantized_func = relay.build_module.bind_params_by_name(quantized_func, params)
    quantized_func = infer_type(quantized_func)
    calibrated_func = infer_type(calibrated_func)

    assert tvm.ir.structural_equal(quantized_func, calibrated_func)


def reset_scale_zp_counter():
    # For testing purposes, we reset the static scale counter to zero before calibrating so that our variable names
    # match up properly
    QuantizerPattern.scales_count = 0
    QuantizerPattern.zp_count = 0


def verify_conv2d(data_shape, weight_shape, attrs, cc=None, pattern_list=None):
    reset_scale_zp_counter()

    conv2d_func, data, weight = create_conv2d_func(data_shape, weight_shape, attrs)
    q_conv2d_func = create_q_conv2d_func(data, weight, weight_shape, attrs)

    if cc is None:
        cc = TestCalibrationCallback()

    params = {"weight": np.random.randn(*weight_shape).astype("float32")}
    if pattern_list is None:
        pattern_list = [Conv2DPattern(cc)]
    quantizer = Quantizer(conv2d_func, params, pattern_list, skip_first=False, skip_last=False)

    test_calibrate(
        quantizer, q_conv2d_func, params, RandomDatasetManager(data_shape, "float32", 1, 3)
    )


def verify_conv2d_bias(
    data_shape, weight_shape, bias_shape, attrs, bias_type="bias_add", cc=None, pattern_list=None
):
    reset_scale_zp_counter()

    conv2d_func, data, weight, bias = create_conv2d_bias_func(
        data_shape, weight_shape, bias_shape, attrs, bias_type
    )
    q_conv2d_func = create_q_conv2d_bias_func(data, weight, bias, weight_shape, attrs, bias_type)

    if cc is None:
        cc = TestCalibrationCallback()

    params = {"weight": np.random.randn(*weight_shape).astype("float32")}
    if pattern_list is None:
        pattern_list = [Conv2DBiasAddPattern(cc)]
    quantizer = Quantizer(conv2d_func, params, pattern_list, skip_first=False, skip_last=False)

    test_calibrate(
        quantizer, q_conv2d_func, params, RandomDatasetManager(data_shape, "float32", 1, 3)
    )


def verify_dense(data_shape, weight_shape, attrs, cc=None, pattern_list=None):
    reset_scale_zp_counter()

    dense_func, data, weight = create_dense_func(data_shape, weight_shape, attrs)
    q_dense_func = create_q_dense_func(data, weight, attrs)

    if cc is None:
        cc = TestCalibrationCallback()

    params = {"weight": np.random.randn(*weight_shape).astype("float32")}

    if pattern_list is None:
        pattern_list = [DensePattern(cc)]
    quantizer = Quantizer(dense_func, params, pattern_list, skip_first=False, skip_last=False)

    test_calibrate(
        quantizer, q_dense_func, params, RandomDatasetManager(data_shape, "float32", 1, 3)
    )


def verify_dense_bias(
    data_shape, weight_shape, bias_shape, attrs, bias_type="bias_add", cc=None, pattern_list=None
):
    reset_scale_zp_counter()

    dense_bias_func, data, weight, bias = create_dense_bias_func(
        data_shape, weight_shape, bias_shape, attrs, bias_type
    )
    q_dense_bias_func = create_q_dense_bias_func(data, weight, bias, attrs, bias_type)

    if cc is None:
        cc = TestCalibrationCallback()

    params = {"weight": np.random.randn(*weight_shape).astype("float32")}
    if pattern_list is None:
        pattern_list = [DenseBiasAddPattern(cc)]
    quantizer = Quantizer(dense_bias_func, params, pattern_list, skip_first=False, skip_last=False)

    test_calibrate(
        quantizer, q_dense_bias_func, params, RandomDatasetManager(data_shape, "float32", 1, 3)
    )


def verify_add(lhs_shape, rhs_shape, cc=None):
    reset_scale_zp_counter()

    add_func, lhs, rhs = create_add_func(lhs_shape, rhs_shape)
    q_add_func = create_q_add_func(lhs, rhs)

    if cc is None:
        cc = TestCalibrationCallback()

    params = {"weight": np.random.randn(*rhs_shape).astype("float32")}
    quantizer = Quantizer(add_func, params, [AddPattern(cc)], skip_first=False, skip_last=False)

    test_calibrate(quantizer, q_add_func, params, RandomDatasetManager(lhs_shape, "float32", 1, 3))


def verify_mul(lhs_shape, rhs_shape, cc=None):
    reset_scale_zp_counter()

    mul_func, lhs, rhs = create_mul_func(lhs_shape, rhs_shape)
    q_mul_func = create_q_mul_func(lhs, rhs)

    if cc is None:
        cc = TestCalibrationCallback()

    params = {"weight": np.random.randn(*rhs_shape).astype("float32")}
    quantizer = Quantizer(
        mul_func, params, [MultiplyPattern(cc)], skip_first=False, skip_last=False
    )

    test_calibrate(quantizer, q_mul_func, params, RandomDatasetManager(lhs_shape, "float32", 1, 3))


def verify_cc_all_patterns(cc):
    verify_conv2d(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        cc=cc,
    )
    verify_conv2d(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        cc=cc,
    )
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        cc=cc,
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        cc=cc,
    )
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (1, 32, 1, 1),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
        cc=cc,
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (1, 1, 1, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
        cc=cc,
    )
    verify_dense((1, 8), (16, 8), {"units": 16}, cc=cc)
    verify_dense((1, 4), (3, 4), {"units": 3}, cc=cc)
    verify_dense_bias((1, 8), (16, 8), (16,), {"units": 16}, cc=cc)
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3}, cc=cc)
    verify_dense_bias((1, 8), (16, 8), (16,), {"units": 16}, bias_type="normal_add", cc=cc)
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3}, bias_type="normal_add", cc=cc)
    verify_add((1, 2, 3), (1, 2, 3), cc=cc)
    verify_mul((1, 2, 3), (1, 2, 3), cc=cc)


def test_conv2d():
    verify_conv2d(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
    )
    verify_conv2d(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
    )


def test_conv2d_bias():
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
    )
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (1, 32, 1, 1),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (1, 1, 1, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
    )


def test_dense():
    verify_dense((1, 8), (16, 8), {"units": 16})
    verify_dense((1, 4), (3, 4), {"units": 3})


def test_dense_bias():
    verify_dense_bias((1, 8), (16, 8), (16,), {"units": 16})
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3})
    verify_dense_bias((1, 8), (16, 8), (16,), {"units": 16}, bias_type="normal_add")
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3}, bias_type="normal_add")


def test_add():
    verify_add((1, 2, 3), (1, 2, 3))


def test_mul():
    verify_mul((1, 2, 3), (1, 2, 3))


def test_global_cc():
    verify_cc_all_patterns(GlobalCalibrationCallback(0.05, 0.01))


def test_average_max_cc():
    verify_cc_all_patterns(AverageMaxCalibrationCallback())


def test_per_channel_average_max_cc():
    pl = [AverageMaxPerChannelConv2DPattern()]
    verify_conv2d(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        pattern_list=pl,
    )
    verify_conv2d(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        pattern_list=pl,
    )
    pl = [AverageMaxPerChannelConv2DBiasAddPattern()]
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        pattern_list=pl,
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        pattern_list=pl,
    )
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (1, 32, 1, 1),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
        pattern_list=pl,
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (1, 1, 1, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
        pattern_list=pl,
    )
    pl = [AverageMaxPerChannelDensePattern()]
    verify_dense((1, 8), (16, 8), {"units": 16}, pattern_list=pl)
    verify_dense((1, 4), (3, 4), {"units": 3}, pattern_list=pl)
    pl = [AverageMaxPerChannelDenseBiasAddPattern()]
    verify_dense_bias((1, 8), (16, 8), (16,), {"units": 16}, pattern_list=pl)
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3}, pattern_list=pl)
    verify_dense_bias(
        (1, 8), (16, 8), (16,), {"units": 16}, bias_type="normal_add", pattern_list=pl
    )
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3}, bias_type="normal_add", pattern_list=pl)


if __name__ == "__main__":
    test_conv2d()
    test_conv2d_bias()
    test_dense()
    test_dense_bias()
    test_add()
    test_mul()
    test_global_cc()
    test_average_max_cc()
    test_per_channel_average_max_cc()
