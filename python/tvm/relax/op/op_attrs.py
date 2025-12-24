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
"""The attributes node used for Relax operators"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from ir import FloatImm, IntImm, VDevice
    from relax.expr import PrimValue
    from tir import IndexMap
    from tvm_ffi import Object, dtype
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)
from tvm.ir import Attrs
import tvm_ffi


@tvm_ffi.register_object("relax.attrs.CallTIRWithGradAttrs")
class CallTIRWithGradAttrs(Attrs):
    """Attributes used in call_tir_with_grad operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.CallTIRWithGradAttrs
    # fmt: off
    te_grad_name: str
    te_grad_kwargs: Mapping[str, Any]
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.InitAttrs")
class InitAttrs(Attrs):
    """Attributes used in full/full_like, ones/ones_like, and zeros/zeros_like operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.InitAttrs
    # fmt: off
    dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.TriluAttrs")
class TriluAttrs(Attrs):
    """Attributes used in tril and triu operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.TriluAttrs
    # fmt: off
    k: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AstypeAttrs")
class AstypeAttrs(Attrs):
    """Attributes used in astype operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AstypeAttrs
    # fmt: off
    dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.TakeAttrs")
class TakeAttrs(Attrs):
    """Attributes used in take operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.TakeAttrs
    # fmt: off
    axis: int | None
    mode: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.StridedSliceAttrs")
class StridedSliceAttrs(Attrs):
    """Attributes used in strided_slice operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.StridedSliceAttrs
    # fmt: off
    assume_inbound: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.MatmulAttrs")
class MatmulAttrs(Attrs):
    """Attributes for matmul operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.MatmulAttrs
    # fmt: off
    out_dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Conv2DAttrs")
class Conv2DAttrs(Attrs):
    """Attributes for nn.conv2d"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Conv2DAttrs
    # fmt: off
    strides: Sequence[IntImm]
    padding: Sequence[IntImm]
    dilation: Sequence[IntImm]
    groups: int
    data_layout: str
    kernel_layout: str
    out_layout: str
    out_dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Conv3DAttrs")
class Conv3DAttrs(Attrs):
    """Attributes for nn.conv3d"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Conv3DAttrs
    # fmt: off
    strides: Sequence[IntImm]
    padding: Sequence[IntImm]
    dilation: Sequence[IntImm]
    groups: int
    data_layout: str
    kernel_layout: str
    out_layout: str
    out_dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Conv2DTransposeAttrs")
class Conv2DTransposeAttrs(Attrs):
    """Attributes for nn.conv2d_transpose"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Conv2DTransposeAttrs
    # fmt: off
    strides: Sequence[IntImm]
    padding: Sequence[IntImm]
    output_padding: Sequence[IntImm]
    dilation: Sequence[IntImm]
    groups: int
    data_layout: str
    kernel_layout: str
    out_layout: str
    out_dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Pool2DAttrs")
class Pool2DAttrs(Attrs):
    """Attributes for nn.max_pool2d"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Pool2DAttrs
    # fmt: off
    pool_size: Sequence[IntImm]
    strides: Sequence[IntImm]
    dilation: Sequence[IntImm]
    padding: Sequence[IntImm]
    ceil_mode: bool
    count_include_pad: bool
    layout: str
    out_layout: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AdaptivePool2DAttrs")
class AdaptivePool2DAttrs(Attrs):
    """Attributes for 2d adaptive pool operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AdaptivePool2DAttrs
    # fmt: off
    output_size: Sequence[IntImm] | None
    layout: str
    out_layout: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.SoftmaxAttrs")
class SoftmaxAttrs(Attrs):
    """Attributes for nn.softmax"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.SoftmaxAttrs
    # fmt: off
    axis: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.BatchNormAttrs")
class BatchNormAttrs(Attrs):
    """Attributes used in batch_norm operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.BatchNormAttrs
    # fmt: off
    axis: int
    epsilon: float
    center: bool
    scale: bool
    momentum: float
    training: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.LayerNormAttrs")
class LayerNormAttrs(Attrs):
    """Attributes used in layer_norm operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.LayerNormAttrs
    # fmt: off
    axes: Sequence[IntImm]
    epsilon: float
    center: bool
    scale: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.InstanceNormAttrs")
class InstanceNormAttrs(Attrs):
    """Attributes used in instance_norm operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.InstanceNormAttrs
    # fmt: off
    channel_axis: int
    axes: Sequence[IntImm]
    epsilon: float
    center: bool
    scale: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.DropoutAttrs")
class DropoutAttrs(Attrs):
    """Attributes for dropout operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.DropoutAttrs
    # fmt: off
    rate: float
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.StatisticalAttrs")
class StatisticalAttrs(Attrs):
    """Attributes used in statistical operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.StatisticalAttrs
    # fmt: off
    axis: Sequence[IntImm] | None
    keepdims: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ConcatAttrs")
class ConcatAttrs(Attrs):
    """Attributes for concat operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ConcatAttrs
    # fmt: off
    axis: int | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ExpandDimsAttrs")
class ExpandDimsAttrs(Attrs):
    """Attributes for expand_dims operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ExpandDimsAttrs
    # fmt: off
    axis: Sequence[IntImm]
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.PermuteDimsAttrs")
class PermuteDimsAttrs(Attrs):
    """Attributes for permute_dims operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.PermuteDimsAttrs
    # fmt: off
    axes: Sequence[IntImm] | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.SortAttrs")
class SortAttrs(Attrs):
    """Attributes for sort operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.SortAttrs
    # fmt: off
    axis: int
    descending: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ArgsortAttrs")
class ArgsortAttrs(Attrs):
    """Attributes for argsort operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ArgsortAttrs
    # fmt: off
    axis: int
    descending: bool
    dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.SplitAttrs")
class SplitAttrs(Attrs):
    """Attributes used in split operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.SplitAttrs
    # fmt: off
    indices_or_sections: Object
    axis: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.SqueezeAttrs")
class SqueezeAttrs(Attrs):
    """Attributes for squeeze operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.SqueezeAttrs
    # fmt: off
    axis: Sequence[IntImm] | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.StackAttrs")
class StackAttrs(Attrs):
    """Attributes for concat operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.StackAttrs
    # fmt: off
    axis: IntImm | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.IndexPutAttrs")
class IndexPutAttrs(Attrs):
    """Attributes for index_put operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.IndexPutAttrs
    # fmt: off
    accumulate: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.LayoutTransformAttrs")
class LayoutTransformAttrs(Attrs):
    """Attributes used in layout_transform operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.LayoutTransformAttrs
    # fmt: off
    index_map: IndexMap
    pad_value: PrimValue | None
    axis_separators: Sequence[IntImm] | None
    input_axis_separators: Sequence[IntImm] | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Resize2DAttrs")
class Resize2DAttrs(Attrs):
    """Attributes used in image resize2d operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Resize2DAttrs
    # fmt: off
    roi: Sequence[FloatImm]
    layout: str
    method: str
    coordinate_transformation_mode: str
    rounding_method: str
    cubic_alpha: float
    cubic_exclude: int
    extrapolation_value: float
    out_dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ArgmaxArgminAttrs")
class ArgmaxArgminAttrs(Attrs):
    """Attributes for argmax/argmin operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ArgmaxArgminAttrs
    # fmt: off
    axis: int | None
    keepdims: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.RepeatAttrs")
class RepeatAttrs(Attrs):
    """Attributes for repeat operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.RepeatAttrs
    # fmt: off
    repeats: int
    axis: int | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.TileAttrs")
class TileAttrs(Attrs):
    """Attributes for tile operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.TileAttrs
    # fmt: off
    repeats: Sequence[IntImm]
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ScanopAttrs")
class ScanopAttrs(Attrs):
    """Attributes for scan operators"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ScanopAttrs
    # fmt: off
    axis: int | None
    dtype: dtype
    exclusive: IntImm
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.TopKAttrs")
class TopKAttrs(Attrs):
    """Attributes for topk operators"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.TopKAttrs
    # fmt: off
    k: int
    axis: int
    ret_type: str
    largest: bool
    dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.EinsumAttrs")
class EinsumAttrs(Attrs):
    """Attributes for einsum operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.EinsumAttrs
    # fmt: off
    subscripts: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.FlipAttrs")
class FlipAttrs(Attrs):
    """Attributes for flip operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.FlipAttrs
    # fmt: off
    axis: IntImm
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.PadAttrs")
class PadAttrs(Attrs):
    """Attributes used in pad operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.PadAttrs
    # fmt: off
    pad_width: Sequence[IntImm]
    pad_value: float
    pad_mode: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.MultinomialFromUniformAttrs")
class MultinomialFromUniformAttrs(Attrs):
    """Attributes for multinomial_from_uniform operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.MultinomialFromUniformAttrs
    # fmt: off
    dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.CallInplacePackedAttrs")
class CallInplacePackedAttrs(Attrs):
    """Attributes used in call_inplace_packed operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.CallInplacePackedAttrs
    # fmt: off
    inplace_indices: Sequence[IntImm]
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.CallTIRInplaceAttrs")
class CallTIRInplaceAttrs(Attrs):
    """Attributes used in call_tir_inplace operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.CallTIRInplaceAttrs
    # fmt: off
    inplace_indices: Sequence[IntImm]
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ToVDeviceAttrs")
class ToVDeviceAttrs(Attrs):
    """Attributes used in to_vdevice operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ToVDeviceAttrs
    # fmt: off
    dst_vdevice: VDevice
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.HintOnDeviceAttrs")
class HintOnDeviceAttrs(Attrs):
    """Attributes used in hint_on_device operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.HintOnDeviceAttrs
    # fmt: off
    device_type: int
    index: int
    memory_scope: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ScatterCollectiveAttrs")
class ScatterCollectiveAttrs(Attrs):
    """Attributes used in scatter collective operators"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ScatterCollectiveAttrs
    # fmt: off
    num_workers: int
    axis: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AttentionAttrs")
class AttentionAttrs(Attrs):
    """Attributes used in attention operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AttentionAttrs
    # fmt: off
    scale: FloatImm | None
    causal_mask: str | None
    window_size: IntImm | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AllClassNonMaximumSuppressionAttrs")
class AllClassNonMaximumSuppressionAttrs(Attrs):
    """Attributes for vision.all_class_non_max_suppression"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AllClassNonMaximumSuppressionAttrs
    # fmt: off
    output_format: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Conv1DAttrs")
class Conv1DAttrs(Attrs):
    """Attributes for nn.conv1d"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Conv1DAttrs
    # fmt: off
    strides: Sequence[IntImm]
    padding: Sequence[IntImm]
    dilation: Sequence[IntImm]
    groups: int
    data_layout: str
    kernel_layout: str
    out_layout: str
    out_dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Conv1DTransposeAttrs")
class Conv1DTransposeAttrs(Attrs):
    """Attributes for nn.conv1d_transpose"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Conv1DTransposeAttrs
    # fmt: off
    strides: Sequence[IntImm]
    padding: Sequence[IntImm]
    output_padding: Sequence[IntImm]
    dilation: Sequence[IntImm]
    groups: int
    data_layout: str
    kernel_layout: str
    out_layout: str
    out_dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Pool1DAttrs")
class Pool1DAttrs(Attrs):
    """Attributes for nn.max_pool1d and nn.avg_pool1d"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Pool1DAttrs
    # fmt: off
    pool_size: Sequence[IntImm]
    strides: Sequence[IntImm]
    dilation: Sequence[IntImm]
    padding: Sequence[IntImm]
    ceil_mode: bool
    count_include_pad: bool
    layout: str
    out_layout: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.Pool3DAttrs")
class Pool3DAttrs(Attrs):
    """Attributes for nn.max_pool3d and nn.avg_pool3d"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.Pool3DAttrs
    # fmt: off
    pool_size: Sequence[IntImm]
    strides: Sequence[IntImm]
    dilation: Sequence[IntImm]
    padding: Sequence[IntImm]
    ceil_mode: bool
    count_include_pad: bool
    layout: str
    out_layout: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AdaptivePool1DAttrs")
class AdaptivePool1DAttrs(Attrs):
    """Attributes for 1d adaptive pool operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AdaptivePool1DAttrs
    # fmt: off
    output_size: Sequence[IntImm] | None
    layout: str
    out_layout: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AdaptivePool3DAttrs")
class AdaptivePool3DAttrs(Attrs):
    """Attributes for 3d adaptive pool operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AdaptivePool3DAttrs
    # fmt: off
    output_size: Sequence[IntImm] | None
    layout: str
    out_layout: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.LeakyReluAttrs")
class LeakyReluAttrs(Attrs):
    """Attributes used in leaky_relu operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.LeakyReluAttrs
    # fmt: off
    alpha: float
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.SoftplusAttrs")
class SoftplusAttrs(Attrs):
    """Attributes used in softplus operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.SoftplusAttrs
    # fmt: off
    beta: float
    threshold: float
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.PReluAttrs")
class PReluAttrs(Attrs):
    """Attributes used in prelu operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.PReluAttrs
    # fmt: off
    axis: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.PixelShuffleAttrs")
class PixelShuffleAttrs(Attrs):
    """Attributes used in pixel_shuffle operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.PixelShuffleAttrs
    # fmt: off
    upscale_factor: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.GroupNormAttrs")
class GroupNormAttrs(Attrs):
    """Attributes used in group_norm operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.GroupNormAttrs
    # fmt: off
    num_groups: int
    channel_axis: int
    axes: Sequence[IntImm]
    epsilon: float
    center: bool
    scale: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.RMSNormAttrs")
class RMSNormAttrs(Attrs):
    """Attributes used in rms_norm operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.RMSNormAttrs
    # fmt: off
    axes: Sequence[IntImm]
    epsilon: float
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.NLLLossAttrs")
class NLLLossAttrs(Attrs):
    """Attributes used in nll_loss operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.NLLLossAttrs
    # fmt: off
    reduction: str
    ignore_index: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AllReduceAttrs")
class AllReduceAttrs(Attrs):
    """Attributes used in allreduce operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AllReduceAttrs
    # fmt: off
    op_type: str
    in_group: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.AllGatherAttrs")
class AllGatherAttrs(Attrs):
    """Attributes used in allgather operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.AllGatherAttrs
    # fmt: off
    num_workers: int
    in_group: bool
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.WrapParamAttrs")
class WrapParamAttrs(Attrs):
    """Attributes used in wrap_param operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.WrapParamAttrs
    # fmt: off
    dtype: dtype
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.QuantizeAttrs")
class QuantizeAttrs(Attrs):
    """Attributes used in quantize/dequantize operators"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.QuantizeAttrs
    # fmt: off
    out_dtype: dtype
    axis: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.GatherElementsAttrs")
class GatherElementsAttrs(Attrs):
    """Attributes for gather_elements operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.GatherElementsAttrs
    # fmt: off
    axis: IntImm
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.GatherNDAttrs")
class GatherNDAttrs(Attrs):
    """Attributes for gather_nd operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.GatherNDAttrs
    # fmt: off
    batch_dims: IntImm
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.MeshgridAttrs")
class MeshgridAttrs(Attrs):
    """Attributes for meshgrid operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.MeshgridAttrs
    # fmt: off
    indexing: str | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ScatterElementsAttrs")
class ScatterElementsAttrs(Attrs):
    """Attributes for scatter_elements operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ScatterElementsAttrs
    # fmt: off
    axis: IntImm
    reduction: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.ScatterNDAttrs")
class ScatterNDAttrs(Attrs):
    """Attributes for scatter_nd operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.ScatterNDAttrs
    # fmt: off
    reduction: str
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.SliceScatterAttrs")
class SliceScatterAttrs(Attrs):
    """Attributes for slice_scatter operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.SliceScatterAttrs
    # fmt: off
    axis: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@tvm_ffi.register_object("relax.attrs.OneHotAttrs")
class OneHotAttrs(Attrs):
    """Attributes for one_hot operator"""

    # tvm-ffi-stubgen(begin): object/relax.attrs.OneHotAttrs
    # fmt: off
    depth: int
    axis: int
    # fmt: on
    # tvm-ffi-stubgen(end)
