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
# pylint: disable=use-list-literal, invalid-name
"""This source will contain code to convert TIR, as produced by
the Relay to TIR compilation process, to Vela API calls to
generate command stream.
"""
from typing import Dict, NamedTuple, Tuple, Union, List
from enum import auto
from enum import Enum
import numpy as np  # type: ignore
import ethosu.vela.api as vapi  # type: ignore

import tvm
from tvm.tir import stmt_functor
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.backend.contrib.ethosu import vela_api
from tvm.relay.backend.contrib.ethosu.tir import spec
from tvm.relay.backend.contrib.ethosu.tir import utils as tir_utils


class BufferType(Enum):
    """The type of information that a buffer contains."""

    constant = auto()
    input_or_output = auto()
    scratch = auto()
    input = auto()
    output = auto()
    shram = auto()


class BufferInfo(NamedTuple):
    """A data structure to hold metadata of the buffer."""

    # If the buffer holds constants, the values will contain that otherwise None
    values: np.ndarray
    shape: tvm.ir.container.Array
    dtype: np.dtype
    btype: BufferType


class AcceleratorArchConfig:
    def __init__(self, total_shram_banks):
        self.shram_bank_size = 1024
        self.total_shram_banks = total_shram_banks
        self.shram_size_bytes = self.shram_bank_size * self.total_shram_banks
        self.lut_size_bytes = 2048
        self.lut_start_address = self.shram_size_bytes - self.lut_size_bytes


def get_accelerator_arch_config(accel_type):
    accel_config_str_map = {
        "ethos-u55-32": AcceleratorArchConfig(16),
        "ethos-u55-64": AcceleratorArchConfig(16),
        "ethos-u55-128": AcceleratorArchConfig(24),
        "ethos-u55-256": AcceleratorArchConfig(48),
        "ethos-u65-256": AcceleratorArchConfig(48),
    }
    return accel_config_str_map[accel_type]


class RegionOffset(NamedTuple):
    """A data structure to hold region and address offset corresponding to a tensor"""

    region: int
    offset: int


def analyze_scratch_memory_acesses(mod: tvm.IRModule, candidate_regions_for_scratch: List[int]):
    """
    This function analyzes the IRModule for intermediary tensors that can be resulting
    from a offset of pool variables (via Let nodes) and/or allocate nodes. The allocate
    nodes will be folded into a single TVMBackendallocWorkspace call with offsets. Ultimately
    this will produce a mapping from each such node to a RegionOffset named tuple that
    has the region and the obtained offset, as mentioned above.

    Parameters
    ----------
    mod: tvm.IRModule
        The TIR module containing ethosu extern calls
    candidate_regions_for_scratch: List[int]
        A list of region integers that could be used for scratch regions

    Returns
    -------
    scratch_region_map : Dict[tvm.tir.Var, RegionOffset]
        A map between buffer vars to scratch regions they are assigned
    tvm_backend_alloc_workspace_size : int
        The size of tvm_backend_alloc_workspace call required to service
        remaining allocate nodes if any
    tvm_backend_alloc_workspace_region : int
        The region associated with the tvm_backend_alloc_workspace
    """
    scratch_region_map = dict()
    pool_var_region_map = dict()
    # There should only be a single function
    assert len(mod.functions.items()) == 1
    primfunc = mod.functions.items()[0][1]
    if "pool_args" in primfunc.attrs.keys():
        pool_args = primfunc.attrs["pool_args"]
        for pool_arg in pool_args:
            pool_param = primfunc.params[int(pool_arg.pool_var_idx)]
            pool_var_region_map[pool_param] = candidate_regions_for_scratch.pop()
            scratch_region_map[pool_param] = RegionOffset(
                region=pool_var_region_map[pool_param], offset=None
            )

    def analyze_pool_access(stmt):
        if isinstance(stmt, tvm.tir.stmt.LetStmt):
            call_address_of = stmt.value
            load = call_address_of.args[0]
            pool_var = load.buffer.data
            scratch_region_map[stmt.var] = RegionOffset(
                region=pool_var_region_map[pool_var], offset=int(load.indices[0])
            )

    tvm.tir.stmt_functor.post_order_visit(primfunc.body, analyze_pool_access)

    dynamic_allocation_region = None
    if len(candidate_regions_for_scratch) > 0:
        dynamic_allocation_region = candidate_regions_for_scratch.pop()
        dynamic_allocation_size = 0

        # If there are tir.Allocate remaining by now, they need to be serviced via
        # dynamic_allocation calls.
        def analyze_remaining_allocates(stmt):
            nonlocal dynamic_allocation_size
            if isinstance(stmt, tvm.tir.stmt.Allocate):
                allocate = stmt
                pointer_type = allocate.buffer_var.type_annotation
                storage_scope = pointer_type.storage_scope
                if storage_scope == "global":
                    dtype_bytes = np.iinfo(np.dtype(allocate.dtype)).bits // 8
                    size_in_bytes = int(dtype_bytes * np.prod(list(allocate.extents)))
                    # Every memory address the NPU access have to be 16 byte aligned
                    size_in_bytes = util.round_up(size_in_bytes, 16)
                    address = dynamic_allocation_size
                    dynamic_allocation_size += size_in_bytes
                    scratch_region_map[allocate.buffer_var] = RegionOffset(
                        region=dynamic_allocation_region, offset=address
                    )

        tvm.tir.stmt_functor.post_order_visit(primfunc.body, analyze_remaining_allocates)

    return (
        scratch_region_map,
        dynamic_allocation_size,
        dynamic_allocation_region,
    )


def _get_region(buffer_type, var=None, scratch_region_map=None):
    """A helper to obtain regions for buffer_types and buffer vars"""
    static_regions = {
        BufferType.constant: 0,
        BufferType.input: 3,
        BufferType.output: 4,
        BufferType.shram: int((1 << 8) | (3 << 0)),
    }
    if buffer_type in static_regions.keys():
        return static_regions[buffer_type]
    assert buffer_type == BufferType.scratch
    assert var in scratch_region_map.keys(), f"{var} is not analyzed for scratch regions"
    return scratch_region_map[var].region


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
    base_addresses : List[util.BaseAddress]
        base addresses to be used by the driver
    """

    # The NPU has 6 usable regions ranging from 0-6
    # The regions 0, 3, and 4 is already used for input,
    # output and constant, respectively (See _get_regions()).
    # Thus, for scratch we are left with 5, 2 and 1.
    candidate_regions_for_scratch = [5, 2, 1]
    (
        scratch_region_map,
        dynamic_allocation_size,
        dynamic_allocation_region,
    ) = analyze_scratch_memory_acesses(tir_module, candidate_regions_for_scratch)
    buffer_info = extract_buffer_info(tir_module, params)
    call_extern_list = extract_call_extern_list(tir_module)
    _npu_ops = list()
    for call_extern in call_extern_list:
        _npu_ops.append(translate_ethosu_tir_call_extern(call_extern))
    _npu_ops, constant_data = assign_addresses(buffer_info, _npu_ops, scratch_region_map)
    base_addresses = extract_param_base_addresses(tir_module, buffer_info, scratch_region_map)
    if dynamic_allocation_size:
        base_addresses.append(
            util.BaseAddress(
                name="dynamic_allocation",
                primfunc_param_idx=None,
                region=dynamic_allocation_region,
                size=dynamic_allocation_size,
                is_runtime_allocation=True,
            )
        )
    target_accel_config = vela_api.get_accelerator_config()
    cmds = vapi.npu_generate_register_command_stream(_npu_ops, target_accel_config)
    payload = vapi.npu_create_driver_payload(cmds, target_accel_config)
    return payload.hex(), constant_data, base_addresses


def extract_param_base_addresses(mod, buffer_info, scratch_region_map) -> List[util.BaseAddress]:
    """This function extracts base addresses to be used by the driver

    Parameters
    ----------
    mod : tvm.IRModule
        The TIR Module for NPU
    buffer_info : Dict[tvm.tir.Var, BufferInfo]
        Information regarding buffer vars used in the PrimFunc

    Returns
    -------
    List[util.BaseAddress]
        base addresses to be used by the driver
    """
    # There should only be a single function
    assert len(mod.functions.items()) == 1
    primfunc = mod.functions.items()[0][1]

    buffer_map = tir_utils.collect_buffer_map(primfunc.body)

    base_addresses = list()
    idx = 0

    for param in primfunc.params:
        # constants are pooled together and handled specially
        # this will change after tir.allocate_const.
        # For now, we are skipping generating buffer addresses here
        if buffer_info[param].btype == BufferType.constant:
            continue

        if param in buffer_map:
            buffer = buffer_map[param]
            dtype = buffer.dtype
            element_size_bytes = np.iinfo(dtype).bits // 8
            size_bytes = element_size_bytes * np.prod(list(buffer.shape))
            base_addresses.append(
                util.BaseAddress(
                    param.name.replace("-", "_"),
                    idx,
                    _get_region(buffer_info[param].btype, param, scratch_region_map),
                    size_bytes,
                )
            )
        else:
            base_addresses.append(
                util.BaseAddress(
                    param.name.replace("-", "_"),
                    idx,
                    _get_region(buffer_info[param].btype, param, scratch_region_map),
                    0,
                )
            )
        idx += 1

    return base_addresses


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
    param_dict : Dict[tvm.tir.Var, np.ndarray]
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

    for param, const_data in param_dict.items():
        if isinstance(param, tvm.tir.Buffer):
            param = param.data
        buffer_info[param] = BufferInfo(
            const_data, const_data.shape, const_data.dtype, BufferType.constant
        )

    pool_param_indices = list()
    if "pool_args" in primfunc.attrs.keys():
        pool_args = primfunc.attrs["pool_args"]
        pool_param_indices = [allocated_pool_info.pool_var_idx for allocated_pool_info in pool_args]

    for idx, param in enumerate(primfunc.params):
        if param not in buffer_info.keys():
            if idx in pool_param_indices:
                btype = BufferType.scratch
            else:
                btype = BufferType.input_or_output
            buffer_info[param] = BufferInfo(
                None,
                None,
                None,
                btype,
            )

    def populate_allocate_buffer_info(stmt):
        if isinstance(stmt, tvm.tir.stmt.Allocate):
            allocate = stmt
            pointer_type = allocate.buffer_var.type_annotation
            storage_scope = pointer_type.storage_scope
            if storage_scope == "local":
                buffer_info[allocate.buffer_var] = BufferInfo(
                    None,
                    allocate.extents,
                    allocate.dtype,
                    BufferType.shram,
                )

    tvm.tir.stmt_functor.post_order_visit(primfunc.body, populate_allocate_buffer_info)
    return buffer_info


def assign_addresses(buffer_info, npu_ops, scratch_region_map):
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
        A list of Vela NpuOps with tir.BufferLoads for addresses
    scratch_region_map : Dict[tvm.tir.Var, RegionOffset]
        A buffer_var to region and offset map.
    Returns
    -------
    npu_ops : list
        A list of Vela NpuOps with addesses within scratch and constant buffers
    constant_tensor : NDArray
        A unified constant data array of uint8 as the constant buffer
    """

    def replace_npu_fm_with_address(npu_fm):
        assert isinstance(npu_fm.tiles.addresses[0], tvm.tir.BufferLoad)
        buffer = npu_fm.tiles.addresses[0].buffer.data
        if buffer in scratch_region_map.keys():
            address = scratch_region_map[buffer].offset
            region = scratch_region_map[buffer].region
        else:
            assert buffer in buffer_addresses.keys()
            address, buffer_type = buffer_addresses[buffer]
            region = _get_region(buffer_type)
        assert (
            len(npu_fm.tiles.addresses[0].indices) == 1
        ), "Ethos-U translation expects flattened buffers"
        index = npu_fm.tiles.addresses[0].indices[0] * (
            np.iinfo(np.dtype(npu_fm.tiles.addresses[0])).bits // 8
        )
        npu_fm.tiles.addresses[0] = address + int(index)
        npu_fm.tiles.addresses[1] = (
            address if isinstance(npu_fm.tiles.addresses[1], tvm.tir.BufferLoad) else 0
        )
        npu_fm.tiles.addresses[2] = (
            address if isinstance(npu_fm.tiles.addresses[2], tvm.tir.BufferLoad) else 0
        )
        npu_fm.tiles.addresses[3] = 0
        npu_fm.region = region
        return npu_fm

    def replace_npu_address_range_with_address(npu_addr_range):
        assert isinstance(npu_addr_range.address, tvm.tir.BufferLoad)
        buffer = npu_addr_range.address.buffer.data
        index = int(
            npu_addr_range.address.indices[0]
            * (np.iinfo(np.dtype(npu_addr_range.address)).bits // 8)
        )
        if buffer in scratch_region_map.keys():
            return vapi.NpuAddressRange(
                scratch_region_map[buffer].region,
                scratch_region_map[buffer].offset + index,
                npu_addr_range.length,
            )
        assert buffer in buffer_addresses.keys(), f"searching for buffer : {buffer}, but not found"
        address, buffer_type = buffer_addresses[buffer]
        address = address + int(npu_addr_range.address.indices[0].value)
        return vapi.NpuAddressRange(_get_region(buffer_type), address, npu_addr_range.length)

    def replace_tir_loads(npu_object):
        if isinstance(npu_object, vapi.NpuFeatureMap):
            return replace_npu_fm_with_address(npu_object)
        if isinstance(npu_object, vapi.NpuAddressRange):
            return replace_npu_address_range_with_address(npu_object)
        return npu_object

    def classify_io(buffer):
        for _npu_op in npu_ops:
            if issubclass(type(_npu_op), vapi.NpuBlockOperation):
                if _npu_op.ifm and _npu_op.ifm.tiles.addresses[0].buffer.data == buffer:
                    return BufferType.input
                if _npu_op.ifm2 and _npu_op.ifm2.tiles.addresses[0].buffer.data == buffer:
                    return BufferType.input
                if _npu_op.ofm and _npu_op.ofm.tiles.addresses[0].buffer.data == buffer:
                    return BufferType.output

        raise ValueError(f"Unused IO : {buffer} in tir module.")

    constant_hex_data = []
    total_constant_len = 0
    buffer_addresses = dict()
    for _buffer, info in buffer_info.items():
        if info.values is not None:
            assert info.btype == BufferType.constant
            assert len(info.shape) == 1
            buffer_addresses[_buffer] = (
                (total_constant_len, info.btype) if constant_hex_data else (0, info.btype)
            )
            dtype_bytes = np.iinfo(np.dtype(info.dtype)).bits // 8
            size_in_bytes = dtype_bytes * np.prod(list(info.shape))
            # Every memory address the NPU access have to be 16 byte aligned
            size_in_bytes = util.round_up(size_in_bytes, 16)
            constant_tensor = np.resize(info.values, size_in_bytes // dtype_bytes)
            constant_tensor = constant_tensor.tobytes().hex()
            constant_hex_data.append(constant_tensor)
            total_constant_len += len(constant_tensor) // 2
        else:
            if info.btype == BufferType.input_or_output or info.btype == BufferType.input:
                buffer_type = info.btype
                if info.btype == BufferType.input_or_output:
                    buffer_type = classify_io(_buffer)
                assert buffer_type in (BufferType.input, BufferType.output)
                address = 0
                buffer_addresses[_buffer] = (address, buffer_type)
                buffer_info[_buffer] = BufferInfo(
                    values=None, shape=info.dtype, dtype=info.dtype, btype=buffer_type
                )
            elif info.btype == BufferType.shram:
                accl_config = util.get_accelerator_config()
                arch_config = get_accelerator_arch_config(accl_config)
                address = arch_config.lut_start_address
                buffer_addresses[_buffer] = (address, info.btype)
            else:
                # These buffer_vars are already updated in scratch_region_map
                assert info.btype == BufferType.scratch

    for npu_op in npu_ops:
        for attr_name, attr in npu_op.__dict__.items():
            if isinstance(attr, list):
                new_attr = list()
                for attr_ in attr:
                    new_attr.append(replace_tir_loads(attr_))
                setattr(npu_op, attr_name, new_attr)
            else:
                setattr(npu_op, attr_name, replace_tir_loads(attr))

    constant_data = "".join(constant_hex_data)
    return (npu_ops, constant_data)


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
        "ethosu_identity": translate_ethosu_pooling,
        "ethosu_unary_elementwise": translate_ethosu_unary_elementwise,
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
    if npu_op.ofm.quantization.scale_f32:
        clip_min_actual = (
            clip_min_quant - npu_op.ofm.quantization.zero_point
        ) * npu_op.ofm.quantization.scale_f32
        clip_max_actual = (
            clip_max_quant - npu_op.ofm.quantization.zero_point
        ) * npu_op.ofm.quantization.scale_f32
    else:
        clip_min_actual = clip_min_quant
        clip_max_actual = clip_max_quant
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
    has_two_weights = serial_2d_convolution.weight2.address != -1
    has_two_biases = serial_2d_convolution.scale_bias2.address != -1

    npu_conv2d_op = vapi.NpuConv2DOperation()
    npu_conv2d_op.ifm = _create_npu_feature_map(serial_2d_convolution.ifm)
    npu_conv2d_op.ofm = _create_npu_feature_map(serial_2d_convolution.ofm)
    npu_conv2d_op.kernel = _create_npu_kernel(serial_2d_convolution.kernel)
    npu_conv2d_op.weights = (
        [
            _create_npu_address_range(serial_2d_convolution.weight),
            _create_npu_address_range(serial_2d_convolution.weight2),
        ]
        if has_two_weights
        else [_create_npu_address_range(serial_2d_convolution.weight)]
    )
    weights_zero_point = np.int64(serial_2d_convolution.weight_zero_point.value)
    npu_conv2d_op.biases = (
        [
            _create_npu_address_range(serial_2d_convolution.scale_bias),
            _create_npu_address_range(serial_2d_convolution.scale_bias2),
        ]
        if has_two_biases
        else [_create_npu_address_range(serial_2d_convolution.scale_bias)]
    )
    npu_conv2d_op.padding = _create_npu_padding(serial_2d_convolution.padding)

    npu_conv2d_op.activation = _create_npu_activation(serial_2d_convolution.activation)
    if (
        npu_conv2d_op.activation
        and npu_conv2d_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_conv2d_op)

    npu_conv2d_op.rounding_mode = _create_npu_rounding_mode(serial_2d_convolution.rounding_mode)
    npu_conv2d_op.ifm_upscale = _create_npu_resampling_mode(serial_2d_convolution.upscale)
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
    npu_conv2d_op.block_config = _create_npu_block_config(serial_2d_convolution.block_config)

    if not npu_conv2d_op.block_config:
        target_accel_config = vela_api.get_accelerator_config()
        block_config = vela_api.get_optimal_block_config(npu_conv2d_op, target_accel_config)
        npu_conv2d_op.block_config = block_config

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

    npu_depthwise_conv2d_op.rounding_mode = _create_npu_rounding_mode(
        serial_2d_depthwise.rounding_mode
    )
    npu_depthwise_conv2d_op.ifm_upscale = _create_npu_resampling_mode(serial_2d_depthwise.upscale)
    npu_depthwise_conv2d_op.block_config = _create_npu_block_config(
        serial_2d_depthwise.block_config
    )

    if not npu_depthwise_conv2d_op.block_config:
        target_accel_config = vela_api.get_accelerator_config()
        block_config = vela_api.get_optimal_block_config(
            npu_depthwise_conv2d_op, target_accel_config
        )
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
    scale = float(scale)
    if scale == 0.0:
        scale = None
    return vapi.NpuQuantization(scale_f32=scale, zero_point=int(zero_point))


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


def _create_npu_block_config(serial_block_config: spec.SerialBlockConfig) -> vapi.NpuShape3D:
    """A helper function to convert a SerialBlockConfig into an NpuShape3D"""
    if serial_block_config.height * serial_block_config.width * serial_block_config.depth == 0:
        return None

    block_config = vapi.NpuShape3D(
        height=int(serial_block_config.height),
        width=int(serial_block_config.width),
        depth=int(serial_block_config.depth),
    )
    return block_config


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
        "TANH": vapi.NpuActivationOp.TABLE_LOOKUP,
        "SIGMOID": vapi.NpuActivationOp.TABLE_LOOKUP,
        "LUT": vapi.NpuActivationOp.TABLE_LOOKUP,
    }
    op = str(serial_activation.op.value)
    assert op in op_map.keys()
    act_op = vapi.NpuActivation(op_map[op])
    if serial_activation.op == "CLIP":
        act_op.min = int(serial_activation.clip_min.value)
        act_op.max = int(serial_activation.clip_max.value)
    if op_map[op] == vapi.NpuActivationOp.TABLE_LOOKUP:
        act_op.lookup_table_index = 0
    return act_op


def _create_npu_resampling_mode(
    mode: str,
) -> vapi.NpuResamplingMode:
    """This is a helper function to capture a list
    of arguments to create Vela NpuResamplingMode object."""
    mode_map = {
        "NONE": vapi.NpuResamplingMode.NONE,
        "NEAREST": vapi.NpuResamplingMode.NEAREST,
        "ZEROS": vapi.NpuResamplingMode.TRANSPOSE,
    }
    mode = str(mode.value)
    assert mode in mode_map.keys()
    return mode_map[mode]


def _create_npu_rounding_mode(
    mode: str,
) -> vapi.NpuRoundingMode:
    """This is a helper function to capture a list
    of arguments to create Vela NpuRoundingMode object."""
    mode_map = {
        "TFL": vapi.NpuRoundingMode.TFL,
        "TRUNCATE": vapi.NpuRoundingMode.TRUNCATE,
        "NATURAL": vapi.NpuRoundingMode.NATURAL,
    }
    mode = str(mode.value)
    assert mode in mode_map.keys()
    return mode_map[mode]


def _create_npu_dma_op(serial_copy):
    """This is a helper function to capture the list of arguments
    to create a NpuDmaOperation object"""
    data_type_bytes = np.iinfo(np.dtype(serial_copy.read_address.dtype)).bits // 8
    length = int(serial_copy.length.value) * data_type_bytes
    # The buffer size in bytes must be at least 16 bytes
    length = max(length, 16)
    src = vapi.NpuAddressRange(
        # region will be updated later
        region=0,
        address=serial_copy.read_address,
        length=length,
    )
    dest = vapi.NpuAddressRange(
        # region will be updated later
        region=0,
        address=serial_copy.write_address,
        length=length,
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
    elif pooling_type == "SUM":
        npu_pooling_op = vapi.NpuPoolingOp.REDUCE_SUM

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

    npu_pooling_op.rounding_mode = _create_npu_rounding_mode(serial_pooling.rounding_mode)
    npu_pooling_op.ifm_upscale = _create_npu_resampling_mode(serial_pooling.upscale)
    npu_pooling_op.block_config = _create_npu_block_config(serial_pooling.block_config)

    if not npu_pooling_op.block_config:
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
    if serial_binary_elementwise.rescale_config.use_rescale:
        npu_binary_elementwise_op.rescale = (
            serial_binary_elementwise.rescale_config.scale.value,
            serial_binary_elementwise.rescale_config.shift.value,
        )

    npu_binary_elementwise_op.activation = _create_npu_activation(
        serial_binary_elementwise.activation
    )
    if (
        npu_binary_elementwise_op.activation
        and npu_binary_elementwise_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_binary_elementwise_op)

    npu_binary_elementwise_op.rounding_mode = _create_npu_rounding_mode(
        serial_binary_elementwise.rounding_mode
    )
    npu_binary_elementwise_op.block_config = _create_npu_block_config(
        serial_binary_elementwise.block_config
    )

    if not npu_binary_elementwise_op.block_config:
        target_accel_config = vela_api.get_accelerator_config()
        block_config = vela_api.get_optimal_block_config(
            npu_binary_elementwise_op, target_accel_config
        )
        npu_binary_elementwise_op.block_config = block_config

    return npu_binary_elementwise_op


def translate_ethosu_unary_elementwise(
    tir_extern_call: tvm.tir.Call,
) -> vapi.NpuElementWiseOperation:
    """This function will translate a tir extern_call
    as produced by Relay to TIR compilation.
    Parameters
    ----------
    tir_extern_call : tvm.tir.Call
        This should be a tir external call that has a agreed upon ordering
        for the NPU TIR Compiler. See SerialUnaryElementwise in
        tvm/relay/backend/contrib/ethosu/tir/spec.py for the ordering.

    Returns
    -------
    ethosu.vela.api.NpuElementWiseOperation
        The vela object containing the params of ethosu_unary_elementwise
    """
    serial_object = spec.create_serial_object(spec.SerialUnaryElementwise, tir_extern_call.args[1:])
    return _create_npu_op_unary_elementwise(serial_object)


def _create_npu_op_unary_elementwise(serial_unary_elementwise):
    operator_type = serial_unary_elementwise.operator_type
    if operator_type == "ABS":
        op = vapi.NpuElementWiseOp.ABS
    if operator_type == "CLZ":
        op = vapi.NpuElementWiseOp.CLZ

    npu_unary_elementwise_op = vapi.NpuElementWiseOperation(op)
    npu_unary_elementwise_op.ifm = _create_npu_feature_map(serial_unary_elementwise.ifm)
    npu_unary_elementwise_op.ofm = _create_npu_feature_map(serial_unary_elementwise.ofm)

    npu_unary_elementwise_op.activation = _create_npu_activation(
        serial_unary_elementwise.activation
    )
    if (
        npu_unary_elementwise_op.activation
        and npu_unary_elementwise_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_unary_elementwise_op)

    npu_unary_elementwise_op.rounding_mode = _create_npu_rounding_mode(
        serial_unary_elementwise.rounding_mode
    )
    npu_unary_elementwise_op.block_config = _create_npu_block_config(
        serial_unary_elementwise.block_config
    )

    if not npu_unary_elementwise_op.block_config:
        target_accel_type = vela_api.get_accelerator_config()
        block_config = vela_api.get_optimal_block_config(
            npu_unary_elementwise_op, target_accel_type
        )
        npu_unary_elementwise_op.block_config = block_config

    return npu_unary_elementwise_op
