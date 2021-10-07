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
"""
This module provides infrastructure to verify the correctness of
the command stream produced.
Currently it will invoke vela to generate a vela-optimized tflite
in which the command stream is contained as a custom operator.
This class include methods to parse the custom operator to extract
the command stream and perform an equivalency check for single operator
test cases.
"""
from typing import List

import os
import struct
import numpy
import math
from enum import IntEnum
import tensorflow as tf

from ethosu.vela.register_command_stream_generator import CmdMode
from ethosu.vela.register_command_stream_generator import cmd0
from ethosu.vela.register_command_stream_generator import cmd1

import tvm
from tvm import relay
import tvm.relay.backend.contrib.ethosu.op as ethosu_ops
from tvm.topi.nn.utils import get_pad_tuple

from tests.python.relay.aot.aot_test_utils import (
    AOTCompiledTestModel,
    AOTDataLinkage,
    AOTTestModel,
    AOTTestRunner,
    compile_models,
    run_and_check,
)


class AttachType(IntEnum):
    kGroupRoot = 1
    kInline = 2
    kInlinedAlready = 3
    kScope = 4
    kScanUpdate = 5


class VelaArtifacts:
    def __init__(self):
        self.cs = dict()
        self.flash = dict()
        self.sram = dict()
        self.npu_ops = set()


def print_payload(payload):
    cmds = deserialize_command_stream(payload)
    for cmd_val in cmds:
        cmd, val = parse_cmd(cmd_val)
        s = str(cmd)
        s = s.ljust(40)
        s += str(val)
        print(s)


def parse_cmd(binary_cmd):
    code = binary_cmd[0] & 0x0000FFFF  # lower 16 bits
    param = binary_cmd[0] >> 16  # higher 16 bits
    payload_mode = CmdMode(code & CmdMode.Mask)
    if payload_mode == CmdMode.Payload32:
        command = cmd1(code & CmdMode.CmdOpMask)
        value = binary_cmd[1]
    else:
        command = cmd0(code & CmdMode.CmdOpMask)
        value = param
    return command, value


def check_cmms_equivalency(vela_cmd, vela_value, tvm_value, ignore_cmds=None):
    if ignore_cmds is None:
        ignore_cmds = []
    if vela_value != tvm_value and vela_cmd not in ignore_cmds:
        raise RuntimeError(
            "ValueMismatch :: vela={}, tvm={} for command:{}".format(
                vela_value, tvm_value, vela_cmd
            )
        )


def verify_cmms(cmms_tvm_blob, cmms_vela_blob):
    vela_cmm = deserialize_command_stream(cmms_vela_blob)
    tvm_cmm = deserialize_command_stream(cmms_tvm_blob)
    cmms_zip = zip(vela_cmm, tvm_cmm)

    first_ifm_found = False
    last_ofm_found = False

    ignore_commands = (
        cmd1.NPU_SET_DMA0_SRC,
        cmd1.NPU_SET_DMA0_DST,
        cmd1.NPU_SET_WEIGHT_BASE,
        cmd1.NPU_SET_OFM_BASE0,
        cmd1.NPU_SET_IFM_BASE0,
        cmd1.NPU_SET_SCALE_BASE,
    )

    ofm_region_params = []
    ofm_bases = []
    for vela_cmm, tvm_cmm in cmms_zip:
        vela_cmd, vela_value = parse_cmd(vela_cmm)
        tvm_cmd, tvm_value = parse_cmd(tvm_cmm)

        assert vela_cmd == tvm_cmd

        # The first IFM region could be different, but it needs to be 1 and 3.
        if vela_cmd == cmd0.NPU_SET_IFM_REGION and not first_ifm_found:
            if vela_value == 1 and tvm_value == 3:
                first_ifm_found = True
                continue

        if vela_cmd == cmd1.NPU_SET_IFM_BASE0 and not first_ifm_found:
            if tvm_value != 0:
                raise RuntimeError("ValueError :: tvm primary ifm base should be zero")
            continue

        # OFM regions should be cached to be checked later
        if vela_cmd == cmd0.NPU_SET_OFM_REGION:
            ofm_region_params.append((vela_value, tvm_value))
            continue

        # OFM bases should be cached to be checked later
        if vela_cmd == cmd1.NPU_SET_OFM_BASE0:
            ofm_bases.append((vela_value, tvm_value))
            continue

        check_cmms_equivalency(vela_cmd, vela_value, tvm_value, ignore_commands)

    # The last OFM region could be different but it should be 1 and 4.
    last_vela_ofm_region, last_tvm_ofm_region = ofm_region_params.pop(-1)
    if not (last_vela_ofm_region == 1 and last_tvm_ofm_region == 4):
        raise RuntimeError(
            "ValueMismatch :: vela={}, tvm={} for last ofm region it should be 1 and 4 respectively".format(
                last_vela_ofm_region, last_tvm_ofm_region
            )
        )

    # The rest of the OFM regions should be the same.
    for vela_value, tvm_value in ofm_region_params:
        check_cmms_equivalency(vela_cmd, vela_value, tvm_value, ignore_commands)

    # The last OFM base should be zero for tvm
    _, last_tvm_ofm_base = ofm_bases.pop(-1)
    if not last_tvm_ofm_base == 0:
        raise RuntimeError("ValueError :: tvm primary ofm base should be zero")


def deserialize_command_stream(blob):
    assert isinstance(blob, bytes)
    payload_bytes = struct.unpack("<{0}I".format(len(blob) // 4), blob)
    cmms = []
    # remove_header
    payload_bytes = payload_bytes[8:]
    idx = 0
    while idx < len(payload_bytes):
        cmd = []
        code = payload_bytes[idx]
        idx += 1
        cmd.append(code)
        payload_mode = CmdMode(code & CmdMode.Mask)
        if payload_mode == CmdMode.Payload32:
            value = payload_bytes[idx]
            idx += 1
            cmd.append(value)
        cmms.append(cmd)
    return cmms


def _create_test_runner(accel):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    test_root = os.path.join(file_dir, "reference_system")
    ethosu_macs = accel[accel.rfind("-") + 1 :]
    return AOTTestRunner(
        makefile="corstone300",
        prologue="""
        uart_init();
        EthosuInit();
        """,
        includes=["uart.h", "ethosu_55.h", "ethosu_mod.h", "hard_fault.h"],
        parameters={"ETHOSU_TEST_ROOT": test_root, "NPU_VARIANT": ethosu_macs},
        pass_config={
            "relay.ext.ethosu.options": {
                "accelerator_config": accel,
            }
        },
    )


def build_source(module, inputs, outputs, accel="ethos-u55-256", output_tolerance=0):
    test_runner = _create_test_runner(accel)
    return compile_models(
        models=AOTTestModel(
            module=module,
            inputs=inputs,
            outputs=outputs,
            output_tolerance=output_tolerance,
            extra_memory_in_bytes=16 * 1024 * 1024,
        ),
        interface_api="c",
        use_unpacked_api=True,
        workspace_byte_alignment=16,
        pass_config=test_runner.pass_config,
    )


def verify_source(
    models: List[AOTCompiledTestModel],
    accel="ethos-u55-256",
):
    """
    This method verifies the generated source from an NPU module by building it and running on an FVP.
    """
    interface_api = "c"
    test_runner = _create_test_runner(accel)
    run_and_check(
        models,
        test_runner,
        interface_api,
        workspace_byte_alignment=16,
        data_linkage=AOTDataLinkage(section="ethosu_scratch", alignment=16),
    )


def flatten_numpy_data(data):
    """Flatten the numpy tensor to be single dimensional"""
    total_elements = data.size
    reshaped_data = numpy.reshape(data, [total_elements])
    return reshaped_data


class InputGenerator:
    def __init__(self, random_state):
        self._random_state = random_state

    def generate(self, size, dtype):
        if dtype == numpy.float32:
            print("random float32")
            return self._random_state.uniform(-1, 1, size).astype(dtype)
        else:
            print("random (u)int min=%d max=%d", numpy.iinfo(dtype).min, numpy.iinfo(dtype).max)
            low = numpy.iinfo(dtype).min
            high = numpy.iinfo(dtype).max + 1
            return self._random_state.randint(low, high, size, dtype)


def generate_ref_data_tflite(model):
    """
    This method generates reference data by running the specified model on tflite with random input data.
    The random input data and generated output data are returned.
    """
    expected_output_data = {}
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize random generators with a fixed seed to get deterministic results
    seed = 0
    random_state = numpy.random.RandomState(seed)

    inputgen = InputGenerator(random_state)

    # Generate input data
    input_data = {
        input_detail["name"]: inputgen.generate(
            input_detail["shape"],
            input_detail["dtype"],
        )
        for input_detail in input_details
    }
    for index, value in enumerate(input_data.values()):
        interpreter.set_tensor(index, value)
    interpreter.invoke()

    expected_output_data = [
        interpreter.get_tensor(output_detail["index"]) for output_detail in output_details
    ]

    return input_data, expected_output_data


def generate_weights_data(shape, dtype):
    size = 1
    for dim in shape:
        size *= dim
    return (numpy.arange(size) % 255).reshape(shape).astype(dtype)


def get_convolutional_args(call, include_buffers=False, remove_constants=False):
    """A method to extract the arguments from conv2d or depthwise_conv2d extern call."""
    args = call.args
    conv_args = []
    remove_indices = [0]

    if remove_constants:
        remove_indices += [41, 42, 44, 45]

    for i, arg in enumerate(args):
        if i in remove_indices:
            continue
        elif isinstance(arg, tvm.tir.expr.IntImm) or isinstance(arg, tvm.tir.expr.FloatImm):
            conv_args.append(arg.value)
        elif isinstance(arg, tvm.tir.expr.Load) and not include_buffers:
            conv_args.append(arg.index)
        else:
            conv_args.append(arg)

    return conv_args


def compute_ofm_shape(ifm_shape, padding, kernel_shape, strides, dilation=[1, 1]):
    assert len(strides) == 2
    assert len(dilation) == 2
    assert len(kernel_shape) == 2
    if padding.lower() == "valid":
        h = math.ceil((ifm_shape[1] - (kernel_shape[0] - 1) * dilation[0]) / strides[0])
        w = math.ceil((ifm_shape[2] - (kernel_shape[1] - 1) * dilation[1]) / strides[1])
    if padding.lower() == "same":
        h = math.ceil(ifm_shape[1] / strides[0])
        w = math.ceil(ifm_shape[2] / strides[1])
    ofm_shape = [ifm_shape[0], h, w, ifm_shape[3]]
    return ofm_shape


def compute_padding_shape(ifm_shape, ofm_shape, padding, kernel_shape, strides, dilation=[1, 1]):
    assert len(strides) == 2
    assert len(dilation) == 2
    assert len(kernel_shape) == 2
    if padding.lower() == "valid":
        return [0, 0, 0, 0]
    if padding.lower() == "same":
        effective_kernel_shape = [
            dilation[0] * (kernel_shape[0] - 1) + 1,
            dilation[1] * (kernel_shape[1] - 1) + 1,
        ]
        pad_along_height = max(
            (ofm_shape[1] - 1) * strides[0] + effective_kernel_shape[0] - ifm_shape[1], 0
        )
        pad_along_width = max(
            (ofm_shape[2] - 1) * strides[1] + effective_kernel_shape[1] - ifm_shape[2], 0
        )
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return [pad_top, pad_left, pad_bottom, pad_right]


def make_ethosu_conv2d(
    ifm,
    ifm_channels,
    ofm_channels,
    kernel_shape,
    padding,
    strides,
    dilation,
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
    weight_dtype="int8",
):
    # conv params
    weight_shape = (ofm_channels, kernel_shape[0], kernel_shape[1], ifm_channels)
    padding = get_pad_tuple(padding, kernel_shape)

    scale_bias_data = generate_weights_data((weight_shape[0], 10), "uint8")
    scale_bias = relay.const(scale_bias_data, dtype="uint8")
    weight_data = generate_weights_data(weight_shape, "int8")
    weight = relay.const(weight_data, dtype=weight_dtype)
    conv = ethosu_ops.ethosu_conv2d(
        ifm,
        weight,
        scale_bias,
        lut=relay.const([], dtype="int8"),
        ifm_scale=0.5,
        ifm_zero_point=10,
        weight_zero_point=12,
        ofm_scale=0.25,
        ofm_zero_point=14,
        kernel_shape=kernel_shape,
        ofm_channels=ofm_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        activation=activation,
        clip_min=10 if activation == "CLIP" else 0,
        clip_max=100 if activation == "CLIP" else 0,
        upscale="NONE",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    return conv


def make_ethosu_depthwise_conv2d(
    ifm,
    channels,
    kernel_shape,
    padding,
    strides,
    dilation,
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
    weight_dtype="int8",
):
    # params
    weight_shape = (channels, kernel_shape[0], kernel_shape[1], 1)
    padding = get_pad_tuple(padding, kernel_shape)

    scale_bias_data = generate_weights_data((weight_shape[0], 10), "uint8")
    scale_bias = relay.const(scale_bias_data, dtype="uint8")
    weight_data = generate_weights_data(weight_shape, weight_dtype)
    weight = relay.const(weight_data, dtype=weight_dtype)
    depthwise = ethosu_ops.ethosu_depthwise_conv2d(
        ifm,
        weight,
        scale_bias,
        lut=relay.const([], dtype="int8"),
        ifm_scale=0.6,
        ifm_zero_point=11,
        weight_zero_point=13,
        ofm_scale=0.26,
        ofm_zero_point=15,
        kernel_shape=kernel_shape,
        ofm_channels=channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        activation=activation,
        clip_min=15 if activation == "CLIP" else 0,
        clip_max=105 if activation == "CLIP" else 0,
        upscale="NONE",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    return depthwise


def get_pooling_args(call, include_buffers=False):
    args = call.args
    pooling_args = []

    for i, arg in enumerate(args):
        if isinstance(arg, tvm.tir.expr.IntImm) or isinstance(arg, tvm.tir.expr.FloatImm):
            pooling_args.append(arg.value)
        elif isinstance(arg, tvm.tir.expr.Load) and not include_buffers:
            pooling_args.append(arg.index)
        else:
            pooling_args.append(arg)

    return pooling_args


def make_ethosu_pooling(
    ifm,
    pooling_type,
    pool_shape,
    ofm_channels,
    strides,
    padding,
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
):
    pooling = ethosu_ops.ethosu_pooling(
        ifm,
        lut=relay.const([], dtype="int8"),
        pooling_type=pooling_type,
        ifm_scale=1,
        ifm_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        pool_shape=pool_shape,
        ofm_channels=ofm_channels,
        strides=strides,
        padding=padding,
        activation=activation,
        clip_min=10 if activation == "CLIP" else 0,
        clip_max=100 if activation == "CLIP" else 0,
        upscale="NONE",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    return pooling


def get_binary_elementwise_args(call, include_buffers=False):
    args = call.args
    binary_elementwise_args = []

    for i, arg in enumerate(args):
        if isinstance(arg, tvm.tir.expr.IntImm) or isinstance(arg, tvm.tir.expr.FloatImm):
            binary_elementwise_args.append(arg.value)
        elif isinstance(arg, tvm.tir.expr.Load) and not include_buffers:
            binary_elementwise_args.append(arg.index)
        else:
            binary_elementwise_args.append(arg)

    return binary_elementwise_args


def make_ethosu_binary_elementwise(
    ifm,
    ifm2,
    ofm_channels,
    operator_type,
    ofm_dtype,
    reversed_operands=False,
    activation="NONE",
    ifm_layout="NHWC",
    ifm2_layout="NHWC",
    ofm_layout="NHWC",
):
    ethosu_binary_elementwise = ethosu_ops.ethosu_binary_elementwise(
        ifm=ifm,
        ifm2=ifm2,
        lut=relay.const([], dtype="int8"),
        operator_type=operator_type,
        ifm_scale=1,
        ifm_zero_point=0,
        ifm2_scale=1,
        ifm2_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        ofm_channels=ofm_channels,
        reversed_operands=reversed_operands,
        activation=activation,
        ofm_dtype=ofm_dtype,
        clip_min=10 if activation == "CLIP" else 0,
        clip_max=100 if activation == "CLIP" else 0,
        ifm_layout=ifm_layout,
        ifm2_layout=ifm2_layout,
        ofm_layout=ofm_layout,
    )
    return ethosu_binary_elementwise
