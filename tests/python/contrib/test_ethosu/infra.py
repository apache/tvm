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
import numpy as np
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
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.backend.contrib.ethosu import preprocess
import tvm.relay.testing.tf as tf_testing
from tvm import WorkspaceMemoryPools, WorkspacePoolInfo, PoolInfoProperties

from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tvm.testing.aot import (
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


def _get_workspace_size_define_macro(pool_name: str, model_name="default") -> str:
    """This function converts pool names to compiler generated
    workspace pool size macros"""

    prefix = "TVMGEN_" + model_name.upper() + "_"
    postfix = "_WORKSPACE_POOL_SIZE"
    return prefix + pool_name.upper() + postfix


def create_test_runner(
    accel="ethos-u55-256",
    enable_usmp=True,
    enable_cascader=False,
    enable_striping=False,
    workspace_pools=None,
):

    file_dir = os.path.dirname(os.path.abspath(__file__))
    test_root = os.path.join(file_dir, "reference_system")
    _, ethosu_variant, ethosu_macs = accel.split("-")
    ethosu_variant = ethosu_variant.upper()

    prologue = """
    UartStdOutInit();
    EthosuInit();

    struct ethosu_driver* ethos_u = ethosu_reserve_driver();
    """

    if workspace_pools:
        for pool in workspace_pools.pools:
            prologue = (
                prologue
                + f"""
    #ifdef {_get_workspace_size_define_macro(pool.pool_name)}
    __attribute__((section(".bss.noinit.tvm"), aligned(16)))
    static uint8_t {pool.pool_name}[{_get_workspace_size_define_macro(pool.pool_name)}];
    #endif

            """
            )

    return AOTTestRunner(
        makefile="corstone300",
        prologue=prologue,
        epilogue="""
        ethosu_release_driver(ethos_u);
        """,
        includes=["uart_stdout.h", "ethosu_55.h", "ethosu_mod.h", "hard_fault.h"],
        parameters={
            "ETHOSU_TEST_ROOT": test_root,
            "NPU_MACS": ethosu_macs,
            "NPU_VARIANT": ethosu_variant,
        },
        pass_config={
            "relay.ext.ethos-u.options": {
                "accelerator_config": accel,
                "enable_cascader": enable_cascader,
                "enable_striping": enable_striping,
            },
            "tir.usmp.enable": enable_usmp,
            "tir.usmp.algorithm": "hill_climb",
            "tir.disable_storage_rewrite": enable_usmp,
        },
    )


def build_source(
    module,
    inputs,
    outputs,
    test_runner,
    output_tolerance=0,
    workspace_pools=None,
):
    return compile_models(
        models=AOTTestModel(
            module=module,
            inputs=inputs,
            outputs=outputs,
            output_tolerance=output_tolerance,
            extra_memory_in_bytes=0,
        ),
        interface_api="c",
        use_unpacked_api=True,
        workspace_memory_pools=workspace_pools,
        workspace_byte_alignment=16,
        pass_config=test_runner.pass_config,
    )


def verify_source(models: List[AOTCompiledTestModel], test_runner):
    """
    This method verifies the generated source from an NPU module by building it and running on an FVP.
    """
    interface_api = "c"
    run_and_check(
        models,
        test_runner,
        interface_api,
        workspace_byte_alignment=16,
        data_linkage=AOTDataLinkage(section="ethosu_scratch", alignment=16),
    )


class InputGenerator:
    def __init__(self, random_state):
        self._random_state = random_state

    def generate(self, size, dtype):
        if dtype == np.float32:
            print("random float32")
            return self._random_state.uniform(-1, 1, size).astype(dtype)
        else:
            print("random (u)int min=%d max=%d", np.iinfo(dtype).min, np.iinfo(dtype).max)
            low = np.iinfo(dtype).min
            high = np.iinfo(dtype).max + 1
            return self._random_state.randint(low, high, size, dtype)


def generate_ref_data_tflite(model):
    """
    This method generates reference data by running the specified model on tflite with random input data.
    The random input data and generated output data are returned.
    """
    expected_output_data = {}

    interpreter = tf.lite.Interpreter(
        model_content=model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize random generators with a fixed seed to get deterministic results
    seed = 0
    random_state = np.random.RandomState(seed)

    inputgen = InputGenerator(random_state)

    # Generate input data
    input_data = {
        input_detail["name"]: inputgen.generate(
            input_detail["shape"],
            input_detail["dtype"],
        )
        for input_detail in input_details
    }
    input_index = {input_detail["name"]: input_detail["index"] for input_detail in input_details}

    for input_name in input_data.keys():
        data = input_data[input_name]
        index = input_index[input_name]
        interpreter.set_tensor(index, data)
    interpreter.invoke()

    expected_output_data = {
        output_detail["name"]: interpreter.get_tensor(output_detail["index"])
        for output_detail in output_details
    }

    return input_data, expected_output_data


def get_tflite_model(model_url):
    """Get a TFLite model from URL."""
    tflite_model_file = tf_testing.get_workload_official(model_url[0], model_url[1])
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    return tflite_model_buf


def get_tflite_graph(tf_func, shapes, ranges=None):
    tensor_specs = [tf.TensorSpec(shape, dtype=tf.float32) for shape in shapes]
    if not ranges:
        ranges = [(0, 1) for _ in shapes]
    concrete_func = tf_func.get_concrete_function(*tensor_specs)

    # Convert the model
    def representative_dataset():
        for _ in range(100):
            inputs = []
            for i, shape in enumerate(shapes):
                data = np.random.uniform(
                    low=ranges[i][0], high=ranges[i][1], size=tuple(shape)
                ).astype("float32")
                inputs.append(data)

            yield inputs

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_graph = converter.convert()

    # Get TFLite model from buffer
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_graph, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(tflite_model)
    mod = partition_for_ethosu(relay_module, params)
    return mod, tflite_graph


def compare_ethosu_with_reference(
    mod,
    input_data,
    output_data,
    accel_type: str,
    output_tolerance=0,
    print_cmm=False,
    enable_cascader=None,
):
    if enable_cascader is None:
        enable_cascader = "u65" not in accel_type
    pool_name = "my_memory_pool"
    host_target = tvm.target.Target("c")
    ethosu_target = tvm.target.Target("ethos-u")
    workspace_pools = WorkspaceMemoryPools(
        [
            WorkspacePoolInfo(
                pool_name,
                [host_target, ethosu_target],
                PoolInfoProperties(
                    size_hint_bytes=2400000,
                    read_bandwidth_bytes_per_cycle=16,
                    write_bandwidth_bytes_per_cycle=16,
                    target_burst_bytes={ethosu_target: 1},
                ),
            )
        ]
    )
    test_runner = create_test_runner(
        accel_type,
        enable_usmp=True,
        enable_cascader=enable_cascader,
        enable_striping=False,
        workspace_pools=workspace_pools,
    )
    compiled_models = build_source(
        mod,
        input_data,
        output_data,
        test_runner,
        workspace_pools=workspace_pools,
        output_tolerance=output_tolerance,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    if print_cmm:
        get_artifacts = tvm._ffi.get_global_func("runtime.module.ethos-u.get_artifacts")
        compilation_artifacts = get_artifacts(ethosu_module)
        cmms = bytes.fromhex(compilation_artifacts[0].command_stream)
        print_payload(cmms)

    verify_source(compiled_models, test_runner)


def compare_tvm_with_tflite(
    tf_func,
    shapes,
    accel_type,
    ranges=None,
    output_tolerance=0,
    print_cmm=False,
    enable_cascader=None,
):
    mod, tflite_graph = get_tflite_graph(tf_func, shapes, ranges)

    # Generate reference data
    input_data, output_data = generate_ref_data_tflite(tflite_graph)

    compare_ethosu_with_reference(
        mod,
        input_data,
        output_data,
        accel_type,
        output_tolerance=output_tolerance,
        print_cmm=print_cmm,
        enable_cascader=enable_cascader,
    )


class EthosUAnnotator(ExprMutator):
    """Annotate entire graph for Ethos-U offload"""

    def __init__(self):
        super(EthosUAnnotator, self).__init__()
        self.compiler = "ethos-u"
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Var):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = compiler_end(new_call, self.compiler)
        return new_call

    def visit_constant(self, constant):
        new_constant = compiler_begin(constant, self.compiler)
        return new_constant


def create_ethosu_partition(mod):
    mod["main"] = EthosUAnnotator().visit(mod["main"])
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    mod = preprocess.preprocess_ext_io()(mod)
    return mod


def generate_weights_data(shape, dtype):
    size = 1
    for dim in shape:
        size *= dim
    return (np.arange(size) % 255).reshape(shape).astype(dtype)


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
        elif isinstance(arg, tvm.tir.expr.BufferLoad) and not include_buffers:
            conv_args.append(arg.indices[0])
        else:
            conv_args.append(arg)

    return conv_args


def compute_ofm_shape(
    ifm_shape, padding, kernel_shape, strides, dilation=[1, 1], channel_padding=[0, 0]
):
    assert len(strides) == 2
    assert len(dilation) == 2
    assert len(kernel_shape) == 2
    if isinstance(padding, tuple):
        h = (
            ifm_shape[1] - (kernel_shape[0] - 1) * dilation[0] + padding[0] + padding[2]
        ) // strides[0]
        w = (
            ifm_shape[2] - (kernel_shape[1] - 1) * dilation[1] + padding[1] + padding[3]
        ) // strides[1]
    elif padding.lower() == "valid":
        h = math.ceil((ifm_shape[1] - (kernel_shape[0] - 1) * dilation[0]) / strides[0])
        w = math.ceil((ifm_shape[2] - (kernel_shape[1] - 1) * dilation[1]) / strides[1])
    elif padding.lower() == "same":
        h = math.ceil(ifm_shape[1] / strides[0])
        w = math.ceil(ifm_shape[2] / strides[1])
    ofm_shape = [ifm_shape[0], h, w, ifm_shape[3] + channel_padding[0] + channel_padding[1]]
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
    lut=relay.const([], dtype="int8"),
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
    weight_dtype="int8",
    scale_bias_dtype="uint8",
    rounding_mode="TFL",
    upscale="NONE",
):
    # conv params
    weight_shape = (ofm_channels, kernel_shape[0], kernel_shape[1], ifm_channels)
    padding = get_pad_tuple(padding, kernel_shape)

    scale_bias_data = generate_weights_data((weight_shape[0], 10), scale_bias_dtype)
    scale_bias = relay.const(scale_bias_data, dtype=scale_bias_dtype)
    weight_data = generate_weights_data(weight_shape, weight_dtype)
    weight = relay.const(weight_data, dtype=weight_dtype)
    conv = ethosu_ops.ethosu_conv2d(
        ifm,
        weight,
        scale_bias,
        lut=lut,
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
        rounding_mode=rounding_mode,
        upscale=upscale,
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
    scale_bias_dtype="uint8",
    rounding_mode="TFL",
):
    # params
    weight_shape = (channels, kernel_shape[0], kernel_shape[1], 1)
    padding = get_pad_tuple(padding, kernel_shape)

    scale_bias_data = generate_weights_data((weight_shape[0], 10), scale_bias_dtype)
    scale_bias = relay.const(scale_bias_data, dtype=scale_bias_dtype)
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
        rounding_mode=rounding_mode,
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
        elif isinstance(arg, tvm.tir.expr.BufferLoad) and not include_buffers:
            pooling_args.append(arg.indices[0])
        else:
            pooling_args.append(arg)

    return pooling_args


def make_ethosu_pooling(
    ifm,
    pooling_type,
    pool_shape,
    ofm_channels,
    ofm_dtype,
    strides,
    padding,
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
    rounding_mode="TFL",
    upscale="NONE",
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
        ofm_dtype=ofm_dtype,
        strides=strides,
        padding=padding,
        activation=activation,
        clip_min=10 if activation == "CLIP" else 0,
        clip_max=100 if activation == "CLIP" else 0,
        rounding_mode=rounding_mode,
        upscale=upscale,
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
        elif isinstance(arg, tvm.tir.expr.BufferLoad) and not include_buffers:
            binary_elementwise_args.append(arg.indices[0])
        else:
            binary_elementwise_args.append(arg)

    return binary_elementwise_args


def make_ethosu_binary_elementwise(
    ifm,
    ifm2,
    ifm_channels,
    ifm2_channels,
    operator_type,
    ofm_dtype,
    reversed_operands=False,
    activation="NONE",
    ifm_layout="NHWC",
    ifm2_layout="NHWC",
    ofm_layout="NHWC",
    rounding_mode="TFL",
    use_rescale: bool = False,
    rescale_scale: int = 0,
    rescale_shift: int = 0,
    lut=relay.const([], dtype="int8"),
    ifm_scale: float = 1.0,
    ifm_zero_point: int = 0,
    ifm2_scale: float = 1.0,
    ifm2_zero_point: int = 0,
    ofm_scale: float = 1.0,
    ofm_zero_point: int = 0,
):
    ethosu_binary_elementwise = ethosu_ops.ethosu_binary_elementwise(
        ifm=ifm,
        ifm2=ifm2,
        lut=lut,
        operator_type=operator_type,
        ifm_scale=ifm_scale,
        ifm_zero_point=ifm_zero_point,
        ifm2_scale=ifm2_scale,
        ifm2_zero_point=ifm2_zero_point,
        ofm_scale=ofm_scale,
        ofm_zero_point=ofm_zero_point,
        ifm_channels=ifm_channels,
        ifm2_channels=ifm2_channels,
        reversed_operands=reversed_operands,
        activation=activation,
        ofm_dtype=ofm_dtype,
        clip_min=10 if activation == "CLIP" else 0,
        clip_max=100 if activation == "CLIP" else 0,
        rounding_mode=rounding_mode,
        ifm_layout=ifm_layout,
        ifm2_layout=ifm2_layout,
        ofm_layout=ofm_layout,
        use_rescale=use_rescale,
        rescale_scale=rescale_scale,
        rescale_shift=rescale_shift,
    )
    return ethosu_binary_elementwise


def make_ethosu_identity(
    ifm,
    lut=relay.const([], dtype="int8"),
    ifm_scale=1,
    ifm_zero_point=0,
    ofm_scale=1,
    ofm_zero_point=0,
    activation="NONE",
):
    identity = ethosu_ops.ethosu_identity(
        ifm,
        lut=lut,
        ifm_scale=ifm_scale,
        ifm_zero_point=ifm_zero_point,
        ofm_scale=ofm_scale,
        ofm_zero_point=ofm_zero_point,
        activation=activation,
    )
    return identity


def make_ethosu_unary_elementwise(
    ifm,
    ofm_channels,
    operator_type,
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
    rounding_mode="TFL",
):
    ethosu_unary_elementwise = ethosu_ops.ethosu_unary_elementwise(
        ifm=ifm,
        lut=relay.const([], dtype="int8"),
        operator_type=operator_type,
        ifm_scale=1,
        ifm_zero_point=0,
        ofm_scale=1,
        ofm_zero_point=0,
        ofm_channels=ofm_channels,
        activation=activation,
        clip_min=10 if activation == "CLIP" else 0,
        clip_max=100 if activation == "CLIP" else 0,
        rounding_mode=rounding_mode,
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    return ethosu_unary_elementwise
