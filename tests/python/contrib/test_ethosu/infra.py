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
from enum import IntEnum

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


def parse_relay_tflite_model(tflite_model, input_tensor, input_shape, input_dtype):
    mod_, params_ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={input_tensor: input_shape},
        dtype_dict={input_tensor: input_dtype},
    )
    return mod_, params_


def parse_tflite_model(model_file):
    try:
        import tflite

        return tflite.Model.GetRootAsModel(model_file, 0)
    except AttributeError:
        import tflite.Model

        return tflite.Model.Model.GetRootAsModel(model_file, 0)


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


def generate_weights_data(shape, dtype):
    size = 1
    for dim in shape:
        size *= dim
    return (numpy.arange(size) % 255).reshape(shape).astype(dtype)


def get_convolutional_args(call, include_buffers=False, remove_constants=False):
    """A method to extract the arguments from conv2d or depthwise2d extern call."""
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
