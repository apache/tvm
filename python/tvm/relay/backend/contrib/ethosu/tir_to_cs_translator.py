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
"""This source will contain code to convert TIR, as produced by
the Relay to TIR compilation process, to Vela API calls to
generate command stream.
"""
from typing import Dict, NamedTuple, Tuple, Union
from enum import auto
from enum import Enum
import numpy as np  # type: ignore
import ethosu.vela.api as vapi  # type: ignore

import tvm
from tvm.tir import stmt_functor
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.backend.contrib.ethosu import vela_api
from tvm.relay.backend.contrib.ethosu.tir import spec


class BufferType(Enum):
    """The type of information that a buffer contains."""

    constant = auto()
    input_or_output = auto()
    scratch = auto()
    input = auto()
    output = auto()


_REGION_MAP = {
    BufferType.constant: 0,
    BufferType.scratch: 1,
    BufferType.input: 3,
    BufferType.output: 4,
}


class BufferInfo(NamedTuple):
    """A data structure to hold metadata of the buffer."""

    # If the buffer holds constants, the values will contain that otherwise None
    values: np.ndarray
    shape: tvm.ir.container.Array
    dtype: np.dtype
    btype: BufferType


def translate(tir_module, params):
    """This will take an tir module for the NPU
    and compile to command stream

    Parameters
    ----------
    tir_module : tvm.IRModule
        The TIR module containing ethosu extern calls
    params : dict
        A dictionary containing TIR primfunc argument ordering
        idx to constant NDArray map
    accel_type : ethosu.vela.api.NpuAccelerator
        the accelerator variant the tir module needs to compiled to

    Returns
    -------
    cs : str
        An hex string of the bytes of command stream
    encoded_constants : str
        An hex string of the bytes that includes concat'd
        encoded weights, encoded biases and scales.
    scratch_size : int
        The size of the scratch buffer needed.
    """

    buffer_info = extract_buffer_info(tir_module, params)
    call_extern_list = extract_call_extern_list(tir_module)
    _npu_ops = list()
    for call_extern in call_extern_list:
        _npu_ops.append(translate_ethosu_tir_call_extern(call_extern))
    _npu_ops, constant_tensor, scratch_size = assign_addresses(buffer_info, _npu_ops)
    target_accel_config = vela_api.get_accelerator_config()
    cmds = vapi.npu_generate_register_command_stream(_npu_ops, target_accel_config)
    payload = vapi.npu_create_driver_payload(cmds, target_accel_config)
    hex_value = "" if constant_tensor is None else constant_tensor.tobytes().hex()
    return payload.hex(), hex_value, scratch_size


def extract_call_extern_list(mod):
    """This function will obtain all extern
    calls from a TIR module
    Parameters
    ----------
    mod : tvm.IRModule
        The TIR Module for NPU

    Returns
    -------
    list
        of tvm.tir.Call objects
        that are tir extern calls
    """
    # There should only be a single function
    assert len(mod.functions.items()) == 1
    primfunc = mod.functions.items()[0][1]

    call_extern_list = list()

    def populate_call_extern_list(stmt):
        if isinstance(stmt, tvm.tir.Call) and stmt.op.name == "tir.call_extern":
            call_extern_list.append(stmt)

    stmt_functor.post_order_visit(primfunc.body, populate_call_extern_list)
    return call_extern_list


def extract_buffer_info(
    mod: tvm.IRModule, param_dict: Dict[int, np.ndarray]
) -> Dict[str, BufferInfo]:
    """This function is to read the tvm.IRModule that
    contains Relay to TIR compiled IRModule. Thereafter,
    this will extract the buffer information as the shape
    and constant data (if any).

    Parameters
    ----------
    mod : tvm.IRModule
        The NPU TIR IRModule.
    param_dict : Dict[int, np.ndarray]
        A dictionary containing param idx --> const numpy.NDArray

    Returns
    -------
    dict : Dict[str, BufferInfo]
        A dictionary of buffer names --> BufferInfo

    """
    buffer_info = dict()
    # There should only be a single function
    assert len(mod.functions.items()) == 1
    primfunc = mod.functions.items()[0][1]
    for idx, const_data in param_dict.items():
        param = primfunc.params[idx]
        buffer_info[primfunc.buffer_map[param].data] = BufferInfo(
            const_data, const_data.shape, const_data.dtype, BufferType.constant
        )

    for param in primfunc.params:
        if primfunc.buffer_map[param].data not in buffer_info.keys():
            buffer_info[primfunc.buffer_map[param].data] = BufferInfo(
                None,
                primfunc.buffer_map[param].shape,
                primfunc.buffer_map[param].dtype,
                BufferType.input_or_output,
            )

    def populate_allocate_buffer_info(stmt):
        if isinstance(stmt, tvm.tir.stmt.Allocate):
            allocate = stmt
            buffer_info[allocate.buffer_var] = BufferInfo(
                None,
                allocate.extents,
                allocate.dtype,
                BufferType.scratch,
            )

    tvm.tir.stmt_functor.post_order_visit(primfunc.body, populate_allocate_buffer_info)

    return buffer_info


def assign_addresses(buffer_info, npu_ops):
    """This function will assign addresses to tensors
    within two buffers : scratch and constants.
    The scratch is the buffer created to hold all intermediary data
    The constants is the buffer created via unifying all the constant data
    (post-encoding).
    Parameters
    ----------
    buffer_info : dict
        This is the dictionary obtained via calling extract_buffer_info.
        The key is the buffer name to BufferInfo
    npu_ops : list
        A list of Vela NpuOps with tir.Loads for addresses
    Returns
    -------
    npu_ops : list
        A list of Vela NpuOps with addesses within scratch and constant buffers
    constant_tensor : NDArray
        A unified constant data array of uint8 as the constant buffer
    scratch_size : int
        The size of the scratch tensor.
    """

    def replace_npu_fm_with_address(npu_fm):
        assert isinstance(npu_fm.tiles.addresses[0], tvm.tir.Load)
        # We currently does not support tiles
        # Change this when tiles are needed
        # (i.e. when using rolling buffers)
        assert npu_fm.tiles.addresses[1:] == [0, 0, 0]
        npu_fm.tiles.addresses[1:] = [0, 0, 0]
        buffer = npu_fm.tiles.addresses[0].buffer_var
        assert buffer in buffer_addresses.keys()
        address, buffer_type = buffer_addresses[buffer]
        index = npu_fm.tiles.addresses[0].index * (
            np.iinfo(np.dtype(npu_fm.tiles.addresses[0])).bits // 8
        )
        npu_fm.tiles.addresses[0] = address + int(index)
        npu_fm.region = _REGION_MAP[buffer_type]
        return npu_fm

    def replace_npu_address_range_with_address(npu_addr_range):
        assert isinstance(npu_addr_range.address, tvm.tir.Load)
        buffer = npu_addr_range.address.buffer_var
        assert buffer in buffer_addresses.keys()
        address, buffer_type = buffer_addresses[buffer]
        return vapi.NpuAddressRange(_REGION_MAP[buffer_type], address, npu_addr_range.length)

    def replace_tir_loads(npu_object):
        if isinstance(npu_object, vapi.NpuFeatureMap):
            return replace_npu_fm_with_address(npu_object)
        if isinstance(npu_object, vapi.NpuAddressRange):
            return replace_npu_address_range_with_address(npu_object)
        return npu_object

    def classify_io(buffer):
        for _npu_op in npu_ops:
            if issubclass(type(_npu_op), vapi.NpuBlockOperation):
                if _npu_op.ifm and _npu_op.ifm.tiles.addresses[0].buffer_var == buffer:
                    return BufferType.input
                if _npu_op.ifm2 and _npu_op.ifm2.tiles.addresses[0].buffer_var == buffer:
                    return BufferType.input
                if _npu_op.ofm and _npu_op.ofm.tiles.addresses[0].buffer_var == buffer:
                    return BufferType.output

        raise ValueError(f"Unused IO : {buffer} in tir module.")

    scratch_size = 0
    constant_tensor = None
    buffer_addresses = dict()
    for _buffer, info in buffer_info.items():
        if info.values is not None:
            assert np.dtype(info.dtype) == np.uint8
            assert info.btype == BufferType.constant
            assert len(info.shape) == 1
            if constant_tensor is None:
                buffer_addresses[_buffer] = (0, info.btype)
                assert info.values.dtype == np.uint8
                size_in_bytes = info.values.size
                # Every memory address the NPU access have to be 16 byte aligned
                size_in_bytes = util.round_up(size_in_bytes, 16)
                constant_tensor = np.resize(info.values, size_in_bytes)
            else:
                buffer_addresses[_buffer] = (constant_tensor.size, info.btype)
                assert info.values.dtype == np.uint8
                size_in_bytes = info.values.size
                # Every memory address the NPU access have to be 16 byte aligned
                size_in_bytes = util.round_up(size_in_bytes, 16)
                constant_tensor = np.append(constant_tensor, np.resize(info.values, size_in_bytes))
        else:
            size_in_bytes = int(
                (np.iinfo(np.dtype(info.dtype)).bits // 8) * np.prod(list(info.shape))
            )
            # Every memory address the NPU access have to be 16 byte aligned
            size_in_bytes = util.round_up(size_in_bytes, 16)
            if info.btype == BufferType.input_or_output:
                buffer_type = classify_io(_buffer)
                assert buffer_type in (BufferType.input, BufferType.output)
                address = 0
                buffer_addresses[_buffer] = (address, buffer_type)
            else:
                assert info.btype == BufferType.scratch
                address = scratch_size
                scratch_size += size_in_bytes
                buffer_addresses[_buffer] = (address, info.btype)

    for npu_op in npu_ops:
        for attr_name, attr in npu_op.__dict__.items():
            if isinstance(attr, list):
                new_attr = list()
                for attr_ in attr:
                    new_attr.append(replace_tir_loads(attr_))
                setattr(npu_op, attr_name, new_attr)
            else:
                setattr(npu_op, attr_name, replace_tir_loads(attr))

    return npu_ops, constant_tensor, scratch_size


def translate_ethosu_tir_call_extern(tir_call_extern):
    """This is a dispatcher function to dispatch
    correct translation call depending on the extern call's
    first argument"""
    supported_call_extern = {
        "ethosu_conv2d": translate_ethosu_conv2d,
        "ethosu_copy": translate_ethosu_copy,
        "ethosu_depthwise_conv2d": translate_ethosu_depthwise_conv2d,
        "ethosu_pooling": translate_ethosu_pooling,
        "ethosu_binary_elementwise": translate_ethosu_binary_elementwise,
    }
    ext_call_type = tir_call_extern.args[0].value
    assert ext_call_type in supported_call_extern.keys(), f"{ext_call_type} is not yet supported"
    npu_op = supported_call_extern[ext_call_type](tir_call_extern)
    # Some conversions return additional outputs
    # if they are needed, the caller should use the function directly
    if isinstance(npu_op, tuple):
        return npu_op[0]
    return npu_op


def translate_ethosu_copy(tir_call_extern: tvm.tir.Call) -> vapi.NpuDmaOperation:
    """This function will translate a TIR call_extern
    as produced by NPU Relay to TIR compilation.

    Parameters
    ----------
    tir_call_extern : tvm.tir.Call

    Returns
    -------
    ethosu.vela.api.NpuDmaOperation
        The vela object containing the params of ethosu_copy
    """
    # We skip the first element as it is the call_extern function name
    serial_object = spec.create_serial_object(spec.SerialCopy, tir_call_extern.args[1:])
    return _create_npu_dma_op(serial_object)


def _convert_clip_bounds(npu_op: vapi.NpuBlockOperation):
    """This function will convert the min and max value
    of clip activations to non quantized floats as
    expected by the API.

    Parameters
    ----------
    npu_op : vapi.NpuBlockOperation

    """
    clip_min_quant = npu_op.activation.min
    clip_max_quant = npu_op.activation.max
    clip_min_actual = (
        clip_min_quant - npu_op.ofm.quantization.zero_point
    ) * npu_op.ofm.quantization.scale_f32
    clip_max_actual = (
        clip_max_quant - npu_op.ofm.quantization.zero_point
    ) * npu_op.ofm.quantization.scale_f32
    npu_op.activation.min = clip_min_actual
    npu_op.activation.max = clip_max_actual


def translate_ethosu_conv2d(tir_call_extern: tvm.tir.Call) -> Tuple[vapi.NpuConv2DOperation, int]:
    """This function will translate a TIR call_extern
    as produced by NPU Relay to TIR compilation.

    Parameters
    ----------
    tir_call_extern : tvm.tir.Call
        This should be a TIR call_extern that has agreed upon ordering
        for TIR Compiler. See Serial2DConvolution in
        tvm/relay/backend/contrib/ethosu/tir/spec.py for the ordering.

    Returns
    -------
    ethosu.vela.api.NpuConv2DOperation
        The vela object containing the params of ethosu_conv2d
    weights_zero_point : int
        The zero point of the weights
    """
    # We skip the first element as it is the call_extern function name
    serial_object = spec.create_serial_object(spec.Serial2DConvolution, tir_call_extern.args[1:])
    return _create_npu_op_conv2d(serial_object)


def _create_npu_op_conv2d(
    serial_2d_convolution: spec.Serial2DConvolution,
) -> Tuple[vapi.NpuConv2DOperation, int]:
    """This is a helper function to capture a list
    of arguments to create Vela NpuConv2DOperation object.
    """
    npu_conv2d_op = vapi.NpuConv2DOperation()
    npu_conv2d_op.ifm = _create_npu_feature_map(serial_2d_convolution.ifm)
    npu_conv2d_op.ofm = _create_npu_feature_map(serial_2d_convolution.ofm)
    npu_conv2d_op.kernel = _create_npu_kernel(serial_2d_convolution.kernel)
    npu_conv2d_op.weights = [_create_npu_address_range(serial_2d_convolution.weight)]
    weights_zero_point = np.int64(serial_2d_convolution.weight_zero_point.value)
    npu_conv2d_op.biases = [_create_npu_address_range(serial_2d_convolution.scale_bias)]
    npu_conv2d_op.padding = _create_npu_padding(serial_2d_convolution.padding)

    npu_conv2d_op.activation = _create_npu_activation(serial_2d_convolution.activation)
    if (
        npu_conv2d_op.activation
        and npu_conv2d_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_conv2d_op)

    npu_conv2d_op.upscale = _create_npu_resampling_mode(serial_2d_convolution.upscale)
    accel_config = vela_api.get_accelerator_config()
    block_config = vela_api.get_optimal_block_config(npu_conv2d_op, accel_config)
    npu_conv2d_op.block_config = block_config
    weights_shape_ohwi = [
        npu_conv2d_op.ofm.shape.depth,
        npu_conv2d_op.kernel.height,
        npu_conv2d_op.kernel.width,
        npu_conv2d_op.ifm.shape.depth,
    ]
    npu_conv2d_op.block_traversal = vela_api.calculate_block_traversal_mode(
        is_depthwise=False,
        weights_shape_ohwi=weights_shape_ohwi,
        ifm_bitdepth=npu_conv2d_op.ifm.data_type.size_in_bits(),
    )
    return npu_conv2d_op, weights_zero_point


def translate_ethosu_depthwise_conv2d(
    tir_call_extern: tvm.tir.Call,
) -> Tuple[vapi.NpuConvDepthWiseOperation, int]:
    """This function will translate a TIR call_extern
    as produced by NPU Relay to TIR compilation.

    Parameters
    ----------
    tir_call_extern : tvm.tir.Call
        This should be a TIR call_extern that has agreed upon ordering
        for TIR Compiler. See Serial2DDepthwise in
        tvm/relay/backend/contrib/ethosu/tir/spec.py for the ordering.

    Returns
    -------
    ethosu.vela.api.NpuConvDepthWiseOperation
        The vela object containing the params of ethosu_depthwise_conv2d
    weights_zero_point : int
        The zero point of the weights
    """
    serial_object = spec.create_serial_object(spec.Serial2DDepthwise, tir_call_extern.args[1:])
    return _create_npu_op_depthwise_conv2d(serial_object)


def _create_npu_op_depthwise_conv2d(serial_2d_depthwise):
    npu_depthwise_conv2d_op = vapi.NpuConvDepthWiseOperation()

    npu_depthwise_conv2d_op.ifm = _create_npu_feature_map(serial_2d_depthwise.ifm)
    npu_depthwise_conv2d_op.ofm = _create_npu_feature_map(serial_2d_depthwise.ofm)
    npu_depthwise_conv2d_op.kernel = _create_npu_kernel(serial_2d_depthwise.kernel)
    npu_depthwise_conv2d_op.weights = [_create_npu_address_range(serial_2d_depthwise.weight)]
    weights_zero_point = np.int64(serial_2d_depthwise.weight_zero_point.value)
    npu_depthwise_conv2d_op.biases = [_create_npu_address_range(serial_2d_depthwise.scale_bias)]
    npu_depthwise_conv2d_op.padding = _create_npu_padding(serial_2d_depthwise.padding)

    npu_depthwise_conv2d_op.activation = _create_npu_activation(serial_2d_depthwise.activation)
    if (
        npu_depthwise_conv2d_op.activation
        and npu_depthwise_conv2d_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_depthwise_conv2d_op)

    npu_depthwise_conv2d_op.upscale = _create_npu_resampling_mode(serial_2d_depthwise.upscale)
    target_accel_config = vela_api.get_accelerator_config()
    block_config = vela_api.get_optimal_block_config(npu_depthwise_conv2d_op, target_accel_config)
    npu_depthwise_conv2d_op.block_config = block_config

    return npu_depthwise_conv2d_op, weights_zero_point


def _create_npu_feature_map(serial_feature_map: spec.SerialFeatureMap) -> vapi.NpuFeatureMap:
    """This is a helper function to capture a list
    of arguments to create Vela NpuFeatureMap object.
    """
    layout_map = {"NHWC": vapi.NpuLayout.NHWC, "NHCWB16": vapi.NpuLayout.NHCWB16}
    datatype_map = {
        "uint8": vapi.NpuDataType.UINT8,
        "int8": vapi.NpuDataType.INT8,
        "uint16": vapi.NpuDataType.UINT16,
        "int16": vapi.NpuDataType.INT16,
        "int32": vapi.NpuDataType.INT32,
    }
    layout = str(serial_feature_map.layout.value)
    data_type = str(serial_feature_map.data_type.value)
    date_type_bytes = np.iinfo(np.dtype(data_type)).bits // 8
    assert layout in layout_map.keys()
    assert data_type in datatype_map.keys()
    nfm = vapi.NpuFeatureMap()
    nfm.data_type = datatype_map[data_type]
    nfm.shape = vapi.NpuShape3D(
        int(serial_feature_map.height),
        int(serial_feature_map.width),
        int(serial_feature_map.channels),
    )
    nfm.tiles = vapi.NpuTileBox(
        int(serial_feature_map.tile_height_0),
        int(serial_feature_map.tile_height_1),
        int(serial_feature_map.tile_width_0),
        [
            serial_feature_map.tile_address_0,
            serial_feature_map.tile_address_1,
            serial_feature_map.tile_address_2,
            serial_feature_map.tile_address_3,
        ],
    )
    nfm.quantization = _create_npu_quantization(
        serial_feature_map.scale, serial_feature_map.zero_point
    )
    nfm.layout = layout_map[layout]
    nfm.strides = vapi.NpuShape3D(
        int(serial_feature_map.stride_h.value) * date_type_bytes,
        int(serial_feature_map.stride_w.value) * date_type_bytes,
        int(serial_feature_map.stride_c.value) * date_type_bytes,
    )
    return nfm


def _create_npu_kernel(serial_kernel: spec.SerialKernel) -> vapi.NpuKernel:
    """This is a helper function to capture a list
    of arguments to create Vela NpuKernel object.
    """
    nknl = vapi.NpuKernel(
        w=int(serial_kernel.width),
        h=int(serial_kernel.height),
        stride_x=int(serial_kernel.stride_w),
        stride_y=int(serial_kernel.stride_h),
        dilation_x=int(serial_kernel.dilation_w),
        dilation_y=int(serial_kernel.dilation_h),
    )
    return nknl


def _create_npu_address_range(
    serial_address_range: spec.SerialAddressRange,
) -> vapi.NpuAddressRange:
    """This is a helper function to capture a list
    of arguments to create Vela NpuAddressRange object.
    """
    addr_range = vapi.NpuAddressRange(
        # region will be updated later
        region=0,
        address=serial_address_range.address,
        length=int(serial_address_range.length),
    )
    return addr_range


def _create_npu_quantization(
    scale: Union[tvm.tir.FloatImm, float],
    zero_point: Union[tvm.tir.IntImm, int],
) -> vapi.NpuQuantization:
    """This is a helper function to capture a list
    of arguments to create Vela NpuQuantization object.
    """
    return vapi.NpuQuantization(scale_f32=float(scale), zero_point=int(zero_point))


def _create_npu_weights_zero_point(
    zero_point: Union[int, tvm.tir.IntImm],
) -> int:
    """This is a helper function to capture the weights zero point."""
    return int(zero_point)


def _create_npu_padding(serial_padding: spec.SerialPadding) -> vapi.NpuPadding:
    """This is a helper function to capture a list
    of arguments to create Vela NpuPadding object."""
    padding = vapi.NpuPadding(
        top=int(serial_padding.top),
        left=int(serial_padding.left),
        bottom=int(serial_padding.bottom),
        right=int(serial_padding.right),
    )
    return padding


def _create_npu_activation(serial_activation: spec.SerialActivation) -> vapi.NpuActivation:
    """This is a helper function to capture a list
    of arguments to create Vela NpuActivation object."""
    if serial_activation.op == "NONE":
        return None
    if (
        serial_activation.op == "CLIP"
        and serial_activation.clip_min == 0
        and serial_activation.clip_max == 0
    ):
        return None
    op_map = {
        "CLIP": vapi.NpuActivationOp.NONE_OR_RELU,
        "TANH": vapi.NpuActivationOp.TANH,
        "SIGMOID": vapi.NpuActivationOp.SIGMOID,
    }
    op = str(serial_activation.op.value)
    assert op in op_map.keys()
    act_op = vapi.NpuActivation(op_map[op])
    act_op.min = int(serial_activation.clip_min)
    act_op.max = int(serial_activation.clip_max)
    return act_op


def _create_npu_resampling_mode(
    mode: str,
) -> vapi.NpuResamplingMode:
    """This is a helper function to capture a list
    of arguments to create Vela NpuResamplingMode object."""
    mode_map = {
        "NONE": vapi.NpuResamplingMode.NONE,
        "NEAREST": vapi.NpuResamplingMode.NEAREST,
        "TRANSPOSE": vapi.NpuResamplingMode.TRANSPOSE,
    }
    mode = str(mode.value)
    assert mode in mode_map.keys()
    return mode_map[mode]


def _create_npu_dma_op(serial_copy):
    """This is a helper function to capture the list of arguments
    to create a NpuDmaOperation object"""
    src = vapi.NpuAddressRange(
        # region will be updated later
        region=0,
        address=serial_copy.read_address,
        length=int(serial_copy.length.value),
    )
    dest = vapi.NpuAddressRange(
        # region will be updated later
        region=0,
        address=serial_copy.write_address,
        length=int(serial_copy.length.value),
    )
    return vapi.NpuDmaOperation(src, dest)


def translate_ethosu_pooling(tir_call_extern: tvm.tir.Call) -> vapi.NpuPoolingOperation:
    """This function will translate a TIR call_extern
    as produced by NPU Relay to TIR compilation.

    Parameters
    ----------
    tir_call_extern : tvm.tir.Call
        This should be a TIR call_extern that has agreed upon ordering
        for TIR Compiler. See SerialPooling in
        tvm/relay/backend/contrib/ethosu/tir/spec.py for the ordering.

    Returns
    -------
    ethosu.vela.api.NpuPoolingOperation
        The vela object containing the params of ethosu_pooling
    """
    serial_object = spec.create_serial_object(spec.SerialPooling, tir_call_extern.args[1:])
    return _create_npu_op_pooling(serial_object)


def _create_npu_op_pooling(serial_pooling: spec.SerialPooling):
    pooling_type = serial_pooling.pooling_type
    if pooling_type == "AVG":
        npu_pooling_op = vapi.NpuPoolingOp.AVERAGE
    elif pooling_type == "MAX":
        npu_pooling_op = vapi.NpuPoolingOp.MAX

    npu_pooling_op = vapi.NpuPoolingOperation(npu_pooling_op)
    npu_pooling_op.ifm = _create_npu_feature_map(serial_pooling.ifm)
    npu_pooling_op.ofm = _create_npu_feature_map(serial_pooling.ofm)
    npu_pooling_op.kernel = _create_npu_kernel(serial_pooling.pool_shape)
    npu_pooling_op.padding = _create_npu_padding(serial_pooling.padding)

    npu_pooling_op.activation = _create_npu_activation(serial_pooling.activation)
    if (
        npu_pooling_op.activation
        and npu_pooling_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_pooling_op)

    npu_pooling_op.upscale = _create_npu_resampling_mode(serial_pooling.upscale)

    target_accel_config = vela_api.get_accelerator_config()
    block_config = vela_api.get_optimal_block_config(npu_pooling_op, target_accel_config)
    npu_pooling_op.block_config = block_config

    return npu_pooling_op


def translate_ethosu_binary_elementwise(
    tir_call_extern: tvm.tir.Call,
) -> vapi.NpuElementWiseOperation:
    """This function will translate a TIR call_extern
    as produced by NPU Relay to TIR compilation.

    Parameters
    ----------
    tir_call_extern : tvm.tir.Call
        This should be a TIR call_extern that has agreed upon ordering
        for TIR Compiler. See SerialBinaryElementwise in
        tvm/relay/backend/contrib/ethosu/tir/spec.py for the ordering.

    Returns
    -------
    ethosu.vela.api.NpuElementWiseOperation
        The vela object containing the params of ethosu_binary_elementwise
    """
    serial_object = spec.create_serial_object(
        spec.SerialBinaryElementwise, tir_call_extern.args[1:]
    )
    return _create_npu_op_binary_elementwise(serial_object)


def _create_npu_op_binary_elementwise(serial_binary_elementwise: spec.SerialBinaryElementwise):
    operator_type = serial_binary_elementwise.operator_type
    if operator_type == "ADD":
        op = vapi.NpuElementWiseOp.ADD
    elif operator_type == "SUB":
        op = vapi.NpuElementWiseOp.SUB
    elif operator_type == "MUL":
        op = vapi.NpuElementWiseOp.MUL
    elif operator_type == "MIN":
        op = vapi.NpuElementWiseOp.MIN
    elif operator_type == "MAX":
        op = vapi.NpuElementWiseOp.MAX
    elif operator_type == "SHR":
        op = vapi.NpuElementWiseOp.SHR
    elif operator_type == "SHL":
        op = vapi.NpuElementWiseOp.SHL

    npu_binary_elementwise_op = vapi.NpuElementWiseOperation(op)
    npu_binary_elementwise_op.ifm = _create_npu_feature_map(serial_binary_elementwise.ifm)
    npu_binary_elementwise_op.ifm2 = _create_npu_feature_map(serial_binary_elementwise.ifm2)
    npu_binary_elementwise_op.ofm = _create_npu_feature_map(serial_binary_elementwise.ofm)
    npu_binary_elementwise_op.reversed_operands = serial_binary_elementwise.reversed_operands

    npu_binary_elementwise_op.activation = _create_npu_activation(
        serial_binary_elementwise.activation
    )
    if (
        npu_binary_elementwise_op.activation
        and npu_binary_elementwise_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_binary_elementwise_op)

    target_accel_config = vela_api.get_accelerator_config()
    block_config = vela_api.get_optimal_block_config(npu_binary_elementwise_op, target_accel_config)
    npu_binary_elementwise_op.block_config = block_config

    return npu_binary_elementwise_op
