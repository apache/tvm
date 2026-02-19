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
Example NPU Pattern Table with Architectural Concepts

This module demonstrates NPU-specific architectural patterns that are common
across different NPU vendors, including memory hierarchy, quantization,
tiling, and fusion strategies.
"""

from typing import Dict, Any, List
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import PatternCheckContext
from tvm.relax.struct_info import TensorStructInfo
from tvm import DataType
from tvm.ir import Op
from tvm import TVMError

from ...pattern_registry import register_patterns


# NPU-specific configuration constants (vendor-neutral)
class NPUConfig:
    """NPU architectural parameters common across vendors"""

    # Memory hierarchy sizes (in KB) - typical NPU values
    SRAM_SIZE_KB = 256  # On-chip SRAM/scratchpad
    CMX_SIZE_KB = 512  # Compute memory (near compute units)

    # Tiling constraints
    TILE_HEIGHT = 32
    TILE_WIDTH = 32
    VECTOR_SIZE = 16

    # Supported data types for NPU acceleration
    SUPPORTED_DTYPES = ["int8", "int16", "float16", "float32"]
    QUANTIZED_DTYPES = ["int8", "int16"]

    # NPU execution units
    MATRIX_ENGINE_SIZE = 16  # MxN matrix engine
    VECTOR_ENGINE_WIDTH = 64  # Vector processing width

    # Power modes
    POWER_MODES = ["high_performance", "balanced", "low_power"]


def _get_tensor_size_kb(shape: List[int], dtype: DataType) -> float:
    """Calculate tensor size in KB for memory planning"""
    if not shape:
        return 0

    bits_per_element = dtype.bits if hasattr(dtype, "bits") else 32
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    size_bytes = (total_elements * bits_per_element) // 8
    return size_bytes / 1024.0


def _check_npu_memory_constraints(context: PatternCheckContext) -> bool:
    """
    Check if operation fits NPU memory hierarchy constraints.

    This demonstrates how NPUs manage their multi-level memory:
    - L0: Register file (immediate access)
    - L1: SRAM/Scratchpad (single cycle)
    - L2: CMX/Shared memory (few cycles)
    - L3: DRAM (high latency)
    """
    # Extract tensor info from context
    if hasattr(context, "annotated_expr"):
        struct_info = context.annotated_expr.struct_info
        if isinstance(struct_info, TensorStructInfo):
            shape = struct_info.shape
            dtype = struct_info.dtype

            if shape and hasattr(shape, "values"):
                shape_values = [int(v) for v in shape.values]
                size_kb = _get_tensor_size_kb(shape_values, dtype)

                # Check if tensor fits in NPU SRAM
                if size_kb > NPUConfig.SRAM_SIZE_KB:
                    # Would need tiling or streaming
                    return True  # Still valid, but needs decomposition

    return True


def _check_npu_quantization(context: PatternCheckContext) -> bool:
    """
    Check NPU quantization requirements.

    NPUs often have specialized units for quantized operations:
    - INT8 for inference acceleration
    - INT16 for higher precision
    - Mixed precision support
    """
    if hasattr(context, "annotated_expr"):
        struct_info = context.annotated_expr.struct_info
        if isinstance(struct_info, TensorStructInfo):
            dtype = str(struct_info.dtype)

            # Check if dtype is supported by NPU
            if dtype not in NPUConfig.SUPPORTED_DTYPES:
                return False

            # Quantized ops get priority on NPU
            if dtype in NPUConfig.QUANTIZED_DTYPES:
                # Mark for NPU quantized path
                return True

    return True


def _check_npu_tiling(shape_values: List[int]) -> Dict[str, Any]:
    """
    Calculate NPU-friendly tiling parameters.

    NPUs process data in tiles to:
    - Fit in on-chip memory
    - Maximize compute unit utilization
    - Enable pipeline parallelism
    """
    tiling_info = {
        "tile_height": NPUConfig.TILE_HEIGHT,
        "tile_width": NPUConfig.TILE_WIDTH,
        "tiles_needed": 1,
    }

    if len(shape_values) >= 2:
        height, width = shape_values[-2:]
        tiles_h = (height + NPUConfig.TILE_HEIGHT - 1) // NPUConfig.TILE_HEIGHT
        tiles_w = (width + NPUConfig.TILE_WIDTH - 1) // NPUConfig.TILE_WIDTH
        tiling_info["tiles_needed"] = tiles_h * tiles_w

    return tiling_info


def _check_npu_fusion_opportunity(
    context: PatternCheckContext,  # pylint: disable=unused-argument
) -> bool:
    """
    Check for NPU-specific fusion opportunities.

    NPUs benefit from fusing:
    - Conv + Activation (single pass through data)
    - Conv + BatchNorm + Activation
    - Multiple elementwise ops
    """
    # In real implementation, check surrounding ops for fusion
    return True


def conv2d_relu_fused_pattern():
    """
    NPU-optimized Conv2D+ReLU fusion pattern.

    This is a key NPU optimization - fusing convolution with activation
    avoids memory traffic between operations.
    """

    def _make_conv2d_relu_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        conv = is_op("relax.nn.conv2d")(input_tensor, weight)
        relu = is_op("relax.nn.relu")(conv)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "conv": conv,
            "root": relu,
            "npu_fusion": "conv2d_relu",
            "memory_tier": "L1_SRAM",  # Keep intermediate in SRAM
        }
        return relu, annotations

    def _check_conv2d_relu(context: PatternCheckContext) -> bool:
        """Check if Conv2D+ReLU fusion is beneficial for NPU"""
        if not _check_npu_memory_constraints(context):
            return False
        if not _check_npu_quantization(context):
            return False
        return True

    return ("example_npu.conv2d_relu_fused", *_make_conv2d_relu_pattern(), _check_conv2d_relu)


def matmul_patterns():
    """
    NPU-optimized matrix multiplication patterns.

    NPUs typically have dedicated matrix engines (systolic arrays,
    tensor cores) that require specific layouts and sizes.
    """

    def _make_matmul_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.matmul")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
            "npu_engine": "matrix_unit",
            "preferred_layout": "NHWC",  # NPUs often prefer channel-last
        }
        return output, annotations

    def _check_matmul(context: PatternCheckContext) -> bool:
        """Check if matmul can use NPU matrix engine"""
        if not _check_npu_memory_constraints(context):
            return False

        # Check if dimensions align with matrix engine size
        if hasattr(context, "annotated_expr"):
            struct_info = context.annotated_expr.struct_info
            if isinstance(struct_info, TensorStructInfo) and struct_info.shape:
                shape_values = [int(v) for v in struct_info.shape.values]
                # Check if divisible by matrix engine size
                if len(shape_values) >= 2:
                    if shape_values[-1] % NPUConfig.MATRIX_ENGINE_SIZE != 0:
                        # Would need padding
                        pass

        return _check_npu_quantization(context)

    def _matmul_pattern(pattern_name):
        return (pattern_name, *_make_matmul_pattern(), _check_matmul)

    # Register both common names used for matrix multiplication in patterns/tests
    return [
        _matmul_pattern("example_npu.dense"),
        _matmul_pattern("example_npu.matmul"),
    ]


def conv1d_patterns():
    """
    1D Convolution patterns optimized for NPU execution.

    NPUs handle 1D convolution by mapping to 2D operations
    or using specialized 1D processing units.
    """

    def _make_conv1d_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv1d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
            "npu_engine": "vector_unit",
            "vectorization": NPUConfig.VECTOR_SIZE,
        }
        return output, annotations

    def _check_conv1d(context: PatternCheckContext) -> bool:
        """Check if conv1d can use NPU vector engine"""
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    def _conv1d_pattern(pattern_name):
        return (pattern_name, *_make_conv1d_pattern(), _check_conv1d)

    return [_conv1d_pattern("example_npu.conv1d")]


def conv2d_patterns():
    """
    2D Convolution patterns with NPU tiling and memory management.

    2D convolution is the most important NPU operation, with
    dedicated hardware for efficient processing.
    """

    def _make_conv2d_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv2d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
            "npu_engine": "conv_engine",
            "tiling_strategy": "spatial",  # Tile across H/W dimensions
            "memory_layout": "NHWC",  # NPU-friendly layout
        }
        return output, annotations

    def _check_conv2d(context: PatternCheckContext) -> bool:
        """Check conv2d NPU constraints including tiling needs"""
        if not _check_npu_memory_constraints(context):
            return False
        if not _check_npu_quantization(context):
            return False

        # Check if tiling is needed
        if hasattr(context, "annotated_expr"):
            struct_info = context.annotated_expr.struct_info
            if isinstance(struct_info, TensorStructInfo) and struct_info.shape:
                shape_values = [int(v) for v in struct_info.shape.values]
                _ = _check_npu_tiling(shape_values)
                # Store tiling info for runtime use

        return True

    def _conv2d_pattern(pattern_name):
        return (pattern_name, *_make_conv2d_pattern(), _check_conv2d)

    return [_conv2d_pattern("example_npu.conv2d")]


def depthwise_conv2d_patterns():
    """
    Depthwise convolution - critical for mobile NPUs.

    Many NPUs have specialized units for depthwise operations
    used in MobileNet-style architectures.
    """

    def _make_depthwise_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv2d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
            "npu_engine": "depthwise_unit",
            "channel_parallel": True,  # Process channels independently
        }
        return output, annotations

    def _check_depthwise(context: PatternCheckContext) -> bool:
        """Check if this is a depthwise conv that NPU can accelerate"""
        # Check for groups == channels (depthwise)
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    return [("example_npu.depthwise_conv2d", *_make_depthwise_pattern(), _check_depthwise)]


def pooling_patterns():
    """
    Pooling operations with NPU memory streaming.

    NPUs often process pooling with the convolution engine
    or dedicated pooling units.
    """

    def _make_maxpool2d_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.max_pool2d")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
            "npu_engine": "pooling_unit",
            "streaming_mode": True,  # Can stream without storing intermediate
        }
        return output, annotations

    def _make_avgpool2d_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.avg_pool2d")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
            "npu_engine": "pooling_unit",
            "accumulation_type": "int32",  # For quantized inputs
        }
        return output, annotations

    def _check_pooling(context: PatternCheckContext) -> bool:
        """Check pooling NPU constraints"""
        return _check_npu_memory_constraints(context)

    return [
        ("example_npu.max_pool2d", *_make_maxpool2d_pattern(), _check_pooling),
        ("example_npu.avg_pool2d", *_make_avgpool2d_pattern(), _check_pooling),
    ]


def batch_norm_patterns():
    """
    Batch normalization - often fused with conv on NPUs.

    NPUs typically fuse BN into convolution to avoid
    separate memory passes.
    """

    def _make_batch_norm_pattern():
        input_tensor = wildcard()
        gamma = wildcard()
        beta = wildcard()
        moving_mean = wildcard()
        moving_var = wildcard()

        output = is_op("relax.nn.batch_norm")(input_tensor, gamma, beta, moving_mean, moving_var)

        annotations = {
            "input": input_tensor,
            "root": output,
            "npu_fusion_candidate": True,  # Usually fused with previous conv
            "precision": "float16",  # Often computed in reduced precision
        }
        return output, annotations

    def _check_batch_norm(context: PatternCheckContext) -> bool:
        """Check if batch norm should be offloaded or fused"""
        return _check_npu_quantization(context)

    return [("example_npu.batch_norm", *_make_batch_norm_pattern(), _check_batch_norm)]


def activation_patterns():
    """
    NPU activation functions with specialized hardware.

    NPUs have dedicated activation units that can handle
    various functions efficiently.
    """

    def _make_activation_pattern(op_name: str, npu_properties: Dict[str, Any]):
        def _pattern():
            input_tensor = wildcard()
            output = is_op(op_name)(input_tensor)

            annotations = {
                "input": input_tensor,
                "root": output,
                "npu_engine": "activation_unit",
                **npu_properties,
            }
            return output, annotations

        return _pattern

    def _check_activation(context: PatternCheckContext) -> bool:
        """Check if activation can use NPU activation unit"""
        return _check_npu_quantization(context)

    # Different activations have different NPU support
    activations = [
        ("example_npu.relu", "relax.nn.relu", {"lookup_table": False}),
        ("example_npu.relu6", "relax.nn.relu6", {"clamp_value": 6.0}),
        ("example_npu.sigmoid", "relax.nn.sigmoid", {"lookup_table": True}),
        ("example_npu.tanh", "relax.nn.tanh", {"lookup_table": True}),
        ("example_npu.gelu", "relax.nn.gelu", {"approximation": "tanh"}),
    ]

    patterns = []
    for pattern_name, op_name, properties in activations:
        try:
            Op.get(op_name)
        except TVMError:  # pylint: disable=broad-exception-caught
            continue

        pattern_fn = _make_activation_pattern(op_name, properties)
        patterns.append((pattern_name, *pattern_fn(), _check_activation))

    return patterns


def elementwise_patterns():
    """
    Element-wise operations that NPUs can vectorize.

    NPUs process element-wise ops using vector units
    with SIMD capabilities.
    """

    def _make_elementwise_pattern(op_name: str):
        def _pattern():
            input1 = wildcard()
            input2 = wildcard()
            output = is_op(op_name)(input1, input2)

            annotations = {
                "input1": input1,
                "input2": input2,
                "root": output,
                "npu_engine": "vector_unit",
                "vectorization": NPUConfig.VECTOR_ENGINE_WIDTH,
            }
            return output, annotations

        return _pattern

    def _check_elementwise(context: PatternCheckContext) -> bool:
        """Check if elementwise op can use NPU vector unit"""
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    ops = ["relax.add", "relax.multiply", "relax.subtract", "relax.divide"]
    patterns = []
    for op in ops:
        try:
            Op.get(op)
        except TVMError:  # pylint: disable=broad-exception-caught
            continue

        op_short = op.split(".")[-1]
        pattern_fn = _make_elementwise_pattern(op)
        patterns.append((f"example_npu.{op_short}", *pattern_fn(), _check_elementwise))

    return patterns


def quantization_patterns():
    """
    Quantization/dequantization patterns for NPU.

    NPUs need explicit quantization boundaries to switch
    between precision levels.
    """

    def _make_quantize_pattern():
        input_tensor = wildcard()
        output = is_op("relax.quantize")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
            "npu_operation": "quantize",
            "target_dtype": "int8",
        }
        return output, annotations

    def _make_dequantize_pattern():
        input_tensor = wildcard()
        output = is_op("relax.dequantize")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
            "npu_operation": "dequantize",
            "target_dtype": "float32",
        }
        return output, annotations

    def _check_quantization(
        context: PatternCheckContext,  # pylint: disable=unused-argument
    ) -> bool:
        """Check quantization operations"""
        return True

    patterns = []

    try:
        Op.get("relax.quantize")
        patterns.append(("example_npu.quantize", *_make_quantize_pattern(), _check_quantization))
    except TVMError:  # pylint: disable=broad-exception-caught
        pass

    try:
        Op.get("relax.dequantize")
        patterns.append(
            ("example_npu.dequantize", *_make_dequantize_pattern(), _check_quantization)
        )
    except TVMError:  # pylint: disable=broad-exception-caught
        pass

    return patterns


# Register all NPU patterns with architectural awareness
register_patterns(
    [
        conv2d_relu_fused_pattern(),  # Fused patterns first (higher priority)
        *matmul_patterns(),
        *conv1d_patterns(),
        *conv2d_patterns(),
        *depthwise_conv2d_patterns(),
        *pooling_patterns(),
        *batch_norm_patterns(),
        *activation_patterns(),
        *elementwise_patterns(),
        *quantization_patterns(),
    ]
)
