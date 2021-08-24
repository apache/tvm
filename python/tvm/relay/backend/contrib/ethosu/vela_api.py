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
This is an adapter module for conversions between TVM and Vela.
The following conversion APIs are added :
    *Obtaining the best block config
    *Compressing weights
    *Packing biases
"""
import logging
import math
import numpy as np
from ethosu.vela import api as vapi

from tvm.relay.backend.contrib.ethosu import util

# pylint: disable=invalid-name
logger = logging.getLogger("Ethos-U")

VELA_TO_NP_DTYPES = {
    vapi.NpuDataType.UINT8: np.uint8,
    vapi.NpuDataType.UINT16: np.uint16,
    vapi.NpuDataType.INT8: np.int8,
    vapi.NpuDataType.INT16: np.int16,
    vapi.NpuDataType.INT32: np.int32,
}

SCALE_BIAS_LENGTH = 10


def get_optimal_block_config(npu_op, accel_type):
    """
    "The NPU's unit of work is known as a block. It will fetch block(s) from Input
    Feature Map (IFM) and a compute block for Output Feature Map (OFM).
    Therefore, we need to pick an optimal block configuration considering bandwidth
    to bring IFM blocks and the number of OFM block computes need to happen
    to cover the OFM as indicated by the npu op.

    Parameters
    ----------
    npu_op : ethosu.vela.api.NpuOperation
        The NPU operation and its params
    accel_type : ethosu.vela.api.NpuAccelerator
        The NPU accelerator variant
    Returns
    -------
    ethosu.vela.api.NpuShape3d :
        The optimal block config for the operator
    """
    all_valid_block_configs = vapi.npu_find_block_configs(npu_op, accel_type)
    return _get_optimal_block_config(all_valid_block_configs)


def _get_optimal_block_config(all_valid_block_configs):
    """An internal function to get block config with largest depth
    and then highest volume/area"""
    assert isinstance(all_valid_block_configs, list)
    for block_cfg in all_valid_block_configs:
        assert isinstance(block_cfg, vapi.NpuShape3D)

    # Getting the largest volume block for benchmarksing
    all_valid_block_configs.sort(
        key=lambda _cfg: _cfg.depth * _cfg.height * _cfg.width, reverse=True
    )
    largest_volume_block_config = all_valid_block_configs[0]
    largest_volume = (
        largest_volume_block_config.depth
        * largest_volume_block_config.height
        * largest_volume_block_config.width
    )

    all_valid_block_configs.sort(key=lambda _cfg: _cfg.depth, reverse=True)
    max_d = all_valid_block_configs[0].depth
    max_depth_block_configs = [_cfg for _cfg in all_valid_block_configs if _cfg.depth == max_d]
    max_depth_block_configs.sort(key=lambda _cfg: _cfg.height * _cfg.width, reverse=True)
    max_area = max_depth_block_configs[0].height * max_depth_block_configs[0].width
    max_area_depth_block_configs = [
        _cfg for _cfg in max_depth_block_configs if _cfg.height * _cfg.width == max_area
    ]
    # This to get a deterministic anwser everytime
    max_area_depth_block_configs.sort(key=lambda _cfg: _cfg.height, reverse=True)
    assert len(max_area_depth_block_configs) > 0
    current_volume = (
        max_area_depth_block_configs[0].depth
        * max_area_depth_block_configs[0].height
        * max_area_depth_block_configs[0].width
    )
    logger.info("Using block config=%s", max_area_depth_block_configs[0])
    logger.info(
        "Quality of the block config w.r.t. max volume block config=%s",
        100.0 * (current_volume / largest_volume),
    )
    return max_area_depth_block_configs[0]


def compress_weights(
    weights,
    weights_zp,
    weights_layout,
    ifm_bitdepth,
    block_depth,
    dilation,
    accel_type,
    is_depthwise=False,
):
    """The NPU requires the weights to be compressed
    to be executed. Therefore, this function calls into
    the Vela APIs to compress the weights.

    Parameters
    ----------
    weights : numpy.ndarray
        The raw weights
    weights_zp : int
        The zero point of the weights
    weights_layout : str
        A string literal indicating the layout
        Supported values : HWIO, HWOI, OHWI
    ifm_bitdepth : int
        The bit depth of the ifm the weights are used with
    block_depth : int
        The depth of the optimal block config for the operator
    dilation : tuple
        A tuple of 2 elements indicating dilation in h and w
    accel_type : ethosu.vela.api.NpuAccelerator
        The NPU accelerator variant
    is_depthwise : bool, Optional
        This indicates whether the weights are compressed for depthwise convolution

    Returns
    -------
    compressed_weights : bytearray
        Compressed weights
    """
    layout_transform_indices = {"HWIO": (3, 0, 1, 2), "HWOI": (2, 0, 1, 3), "OHWI": (0, 1, 2, 3)}
    assert weights_layout in layout_transform_indices.keys()
    assert isinstance(weights_zp, np.int64)
    weights = weights.astype(np.int64) - weights_zp
    # Vela needs the weights in OHWI layout
    weights_ohwi = np.transpose(weights, layout_transform_indices[weights_layout])
    shape_ohwi = [
        weights.shape[layout_transform_indices[weights_layout][0]],
        weights.shape[layout_transform_indices[weights_layout][1]],
        weights.shape[layout_transform_indices[weights_layout][2]],
        weights.shape[layout_transform_indices[weights_layout][3]],
    ]
    block_traversal = calculate_block_traversal_mode(is_depthwise, shape_ohwi, ifm_bitdepth)
    compressed_weights = vapi.npu_encode_weights(
        accelerator=accel_type,
        weights_volume=weights_ohwi,
        dilation_xy=dilation,
        ifm_bitdepth=ifm_bitdepth,
        ofm_block_depth=block_depth,
        is_depthwise=is_depthwise,
        block_traversal=block_traversal,
    )
    return compressed_weights


def calculate_block_traversal_mode(is_depthwise, weights_shape_ohwi, ifm_bitdepth):
    """Calculate a block traversal mode given whether the op is depthwise convolution,
    shape of weights and bit-depth of the ifm.
    """

    if is_depthwise:
        return vapi.NpuBlockTraversal.DEPTH_FIRST
    # Determine which block traversal strategy has better DPU utilization
    kernel_size = weights_shape_ohwi[1] * weights_shape_ohwi[2]
    depth_utilization = weights_shape_ohwi[3] / util.round_up(
        weights_shape_ohwi[3], 32 if ifm_bitdepth == 8 else 16
    )
    part_kernel_utilization = (weights_shape_ohwi[3] / util.round_up(weights_shape_ohwi[3], 8)) * (
        kernel_size / util.round_up(kernel_size, 4 if ifm_bitdepth == 8 else 2)
    )
    if part_kernel_utilization >= depth_utilization or weights_shape_ohwi[3] <= 8:
        # Part-kernel first is always better for ifm depths <= 8
        return vapi.NpuBlockTraversal.PART_KERNEL_FIRST
    return vapi.NpuBlockTraversal.DEPTH_FIRST


def pack_biases(
    biases,
    ifm_scale,
    ifm_dtype,
    weight_scales,
    ofm_scale,
    is_activation_tanh_or_sigmoid=False,
):
    """
    The NPU requires the each bias value to be packed with
    output scale parameters in a 80-bit format (that is returned
    via npu_encode_bias API). This function will pack such values
    to a binary artifact that the NPU will use in the execution.


    Parameters
    ----------
    biases : numpy.ndarray
        The values of biases
    ifm_scale : float
        The quantization scale parameter of input feature map
    ifm_dtype : numpy.dtype
        The data type of input feature map data.
    weight_scales : numpy.ndarray
        The quantization scale parameter of weight feature map
        This could be a tuple if per-channel quantization is present.
    ofm_scale : float
        The quantization scale parameter of output feature map.
    is_activation_tanh_or_sigmoid : bool
        Indicates whether the fused activation function is tanh or sigmoid.

    Returns
    -------
    scale_bias : numpy.ndarray
        Packed scales/biases as the hardware requires them.
    """
    # The BYOC infra should not partition anything else.
    supported_ifm_dtypes = (np.uint8, np.int8, np.int16)
    assert ifm_dtype in supported_ifm_dtypes

    if weight_scales.size == 1:
        weight_scales = [weight_scales] * biases.size

    hw_bias_scales = _calculate_hw_bias_scales(
        ifm_scale, weight_scales, ofm_scale, ifm_dtype, is_activation_tanh_or_sigmoid
    )
    assert len(hw_bias_scales) == biases.size
    biases = biases.astype("int64")
    packed_biases = bytearray()
    for idx, scale in enumerate(hw_bias_scales):
        packed_biases.extend(vapi.npu_encode_bias(biases[idx], *scale))
    scale_bias = np.frombuffer(packed_biases, dtype=np.uint8)
    scale_bias = np.reshape(scale_bias, (-1, 10))
    return scale_bias


def _quantize_scale(scale):
    """Quantize floating point scale into 32-bit int scale with a 6-bit shift.
    This is to be used with 8-bit data.
    """
    mantissa, exponent = math.frexp(scale)
    mantissa_scaled = mantissa * (1 << 31)
    mantissa_scaled = int(util.round_away_zero(mantissa_scaled))
    required_shift = 31 - exponent
    assert 0 <= required_shift < (1 << 6)
    return mantissa_scaled, required_shift


def _reduced_quantize_scale(scale):
    """A reduction of precision is required for 16 bit data."""
    mantissa_scaled, required_shift = _quantize_scale(scale)
    # This is max a signed 16-bit number could represent
    max_reduced_mantissa_scaled = (1 << 15) - 1
    # if the current value is larger than pre-scaled max_reduced_mantissa_scaled
    # we need to saturate the anwser to max_reduced_mantissa_scaled
    if mantissa_scaled >= max_reduced_mantissa_scaled << 16:
        reduced_mantissa_scaled = max_reduced_mantissa_scaled
    else:
        reduced_mantissa_scaled = (mantissa_scaled + (1 << 15)) >> 16
    reduced_shift = required_shift - 16
    return reduced_mantissa_scaled, reduced_shift


def _calculate_hw_bias_scales(
    ifm_scale, weight_scales, ofm_scale, ifm_dtype, is_faf_tanh_sigmoid=False
):
    """This function will produce a scale that is calculated using scales of ifm,
    weights and ofm. It is also important to note that if per-channel / per-value
    quantization required they should go into hw bias scales"""
    if is_faf_tanh_sigmoid:
        ifm_scale = ifm_scale * 0x3000
    if ifm_dtype == np.uint8:
        bias_scales = [np.double(ifm_scale * ws) / np.double(ofm_scale) for ws in weight_scales]
    else:
        assert ifm_dtype in (np.int8, np.int16)
        ifm_scale_dbl = np.double(ifm_scale)
        ofm_scale_dbl = np.double(ofm_scale)
        bias_scales = [ifm_scale_dbl * np.double(ws) / ofm_scale_dbl for ws in weight_scales]

    if ifm_dtype == np.int16:
        hw_bias_scales = [_reduced_quantize_scale(bs) for bs in bias_scales]
    else:
        assert ifm_dtype in (np.uint8, np.int8)
        hw_bias_scales = [_quantize_scale(bs) for bs in bias_scales]

    return hw_bias_scales


def get_target_accel_type():
    """This is a helper function to convert TVMC command line argument to NpuAccelerator type"""
    npu_accel_str_map = {
        "ethos-u55-256": vapi.NpuAccelerator.Ethos_U55_256,
        "ethos-u55-128": vapi.NpuAccelerator.Ethos_U55_128,
        "ethos-u55-64": vapi.NpuAccelerator.Ethos_U55_64,
        "ethos-u55-32": vapi.NpuAccelerator.Ethos_U55_32,
    }
    accel_type_str = util.get_accelerator_config()
    assert accel_type_str in npu_accel_str_map.keys(), f"{accel_type_str} is not supported"
    return npu_accel_str_map[accel_type_str]
