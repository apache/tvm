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
from tvm.ir import Attrs
import tvm_ffi


@tvm_ffi.register_object("tvm.relax.attrs.CallTIRWithGradAttrs")
class CallTIRWithGradAttrs(Attrs):
    """Attributes used in call_tir_with_grad operator"""


@tvm_ffi.register_object("tvm.relax.attrs.InitAttrs")
class InitAttrs(Attrs):
    """Attributes used in full/full_like, ones/ones_like, and zeros/zeros_like operator"""


@tvm_ffi.register_object("tvm.relax.attrs.TriluAttrs")
class TriluAttrs(Attrs):
    """Attributes used in tril and triu operator"""


@tvm_ffi.register_object("tvm.relax.attrs.AstypeAttrs")
class AstypeAttrs(Attrs):
    """Attributes used in astype operator"""


@tvm_ffi.register_object("tvm.relax.attrs.TakeAttrs")
class TakeAttrs(Attrs):
    """Attributes used in take operator"""


@tvm_ffi.register_object("tvm.relax.attrs.StridedSliceAttrs")
class StridedSliceAttrs(Attrs):
    """Attributes used in strided_slice operator"""


@tvm_ffi.register_object("tvm.relax.attrs.MatmulAttrs")
class MatmulAttrs(Attrs):
    """Attributes for matmul operator"""


@tvm_ffi.register_object("tvm.relax.attrs.Conv2DAttrs")
class Conv2DAttrs(Attrs):
    """Attributes for nn.conv2d"""


@tvm_ffi.register_object("tvm.relax.attrs.Conv3DAttrs")
class Conv3DAttrs(Attrs):
    """Attributes for nn.conv3d"""


@tvm_ffi.register_object("tvm.relax.attrs.Conv2DTransposeAttrs")
class Conv2DTransposeAttrs(Attrs):
    """Attributes for nn.conv2d_transpose"""


@tvm_ffi.register_object("tvm.relax.attrs.Pool2DAttrs")
class Pool2DAttrs(Attrs):
    """Attributes for nn.max_pool2d"""


@tvm_ffi.register_object("tvm.relax.attrs.AdaptivePool2DAttrs")
class AdaptivePool2DAttrs(Attrs):
    """Attributes for 2d adaptive pool operator"""


@tvm_ffi.register_object("tvm.relax.attrs.SoftmaxAttrs")
class SoftmaxAttrs(Attrs):
    """Attributes for nn.softmax"""


@tvm_ffi.register_object("tvm.relax.attrs.BatchNormAttrs")
class BatchNormAttrs(Attrs):
    """Attributes used in batch_norm operator"""


@tvm_ffi.register_object("tvm.relax.attrs.LayerNormAttrs")
class LayerNormAttrs(Attrs):
    """Attributes used in layer_norm operator"""


@tvm_ffi.register_object("tvm.relax.attrs.InstanceNormAttrs")
class InstanceNormAttrs(Attrs):
    """Attributes used in instance_norm operator"""


@tvm_ffi.register_object("tvm.relax.attrs.DropoutAttrs")
class DropoutAttrs(Attrs):
    """Attributes for dropout operator"""


@tvm_ffi.register_object("tvm.relax.attrs.StatisticalAttrs")
class StatisticalAttrs(Attrs):
    """Attributes used in statistical operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ConcatAttrs")
class ConcatAttrs(Attrs):
    """Attributes for concat operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ExpandDimsAttrs")
class ExpandDimsAttrs(Attrs):
    """Attributes for expand_dims operator"""


@tvm_ffi.register_object("tvm.relax.attrs.PermuteDimsAttrs")
class PermuteDimsAttrs(Attrs):
    """Attributes for permute_dims operator"""


@tvm_ffi.register_object("tvm.relax.attrs.SortAttrs")
class SortAttrs(Attrs):
    """Attributes for sort operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ArgsortAttrs")
class ArgsortAttrs(Attrs):
    """Attributes for argsort operator"""


@tvm_ffi.register_object("tvm.relax.attrs.SplitAttrs")
class SplitAttrs(Attrs):
    """Attributes used in split operator"""


@tvm_ffi.register_object("tvm.relax.attrs.SqueezeAttrs")
class SqueezeAttrs(Attrs):
    """Attributes for squeeze operator"""


@tvm_ffi.register_object("tvm.relax.attrs.StackAttrs")
class StackAttrs(Attrs):
    """Attributes for concat operator"""


@tvm_ffi.register_object("tvm.relax.attrs.IndexPutAttrs")
class IndexPutAttrs(Attrs):
    """Attributes for index_put operator"""


@tvm_ffi.register_object("tvm.relax.attrs.LayoutTransformAttrs")
class LayoutTransformAttrs(Attrs):
    """Attributes used in layout_transform operator"""


@tvm_ffi.register_object("tvm.relax.attrs.Resize2DAttrs")
class Resize2DAttrs(Attrs):
    """Attributes used in image resize2d operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ArgmaxArgminAttrs")
class ArgmaxArgminAttrs(Attrs):
    """Attributes for argmax/argmin operator"""


@tvm_ffi.register_object("tvm.relax.attrs.RepeatAttrs")
class RepeatAttrs(Attrs):
    """Attributes for repeat operator"""


@tvm_ffi.register_object("tvm.relax.attrs.TileAttrs")
class TileAttrs(Attrs):
    """Attributes for tile operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ScanopAttrs")
class ScanopAttrs(Attrs):
    """Attributes for scan operators"""


@tvm_ffi.register_object("tvm.relax.attrs.TopKAttrs")
class TopKAttrs(Attrs):
    """Attributes for topk operators"""


@tvm_ffi.register_object("tvm.relax.attrs.EinsumAttrs")
class EinsumAttrs(Attrs):
    """Attributes for einsum operator"""


@tvm_ffi.register_object("tvm.relax.attrs.FlipAttrs")
class FlipAttrs(Attrs):
    """Attributes for flip operator"""


@tvm_ffi.register_object("tvm.relax.attrs.PadAttrs")
class PadAttrs(Attrs):
    """Attributes used in pad operator"""


@tvm_ffi.register_object("tvm.relax.attrs.MultinomialFromUniformAttrs")
class MultinomialFromUniformAttrs(Attrs):
    """Attributes for multinomial_from_uniform operator"""


@tvm_ffi.register_object("tvm.relax.attrs.CallInplacePackedAttrs")
class CallInplacePackedAttrs(Attrs):
    """Attributes used in call_inplace_packed operator"""


@tvm_ffi.register_object("tvm.relax.attrs.CallTIRInplaceAttrs")
class CallTIRInplaceAttrs(Attrs):
    """Attributes used in call_tir_inplace operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ToVDeviceAttrs")
class ToVDeviceAttrs(Attrs):
    """Attributes used in to_vdevice operator"""


@tvm_ffi.register_object("tvm.relax.attrs.HintOnDeviceAttrs")
class HintOnDeviceAttrs(Attrs):
    """Attributes used in hint_on_device operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ScatterCollectiveAttrs")
class ScatterCollectiveAttrs(Attrs):
    """Attributes used in scatter collective operators"""


@tvm_ffi.register_object("tvm.relax.attrs.AttentionAttrs")
class AttentionAttrs(Attrs):
    """Attributes used in attention operator"""


@tvm_ffi.register_object("tvm.relax.attrs.AllClassNonMaximumSuppressionAttrs")
class AllClassNonMaximumSuppressionAttrs(Attrs):
    """Attributes for vision.all_class_non_max_suppression"""


@tvm_ffi.register_object("tvm.relax.attrs.Conv1DAttrs")
class Conv1DAttrs(Attrs):
    """Attributes for nn.conv1d"""


@tvm_ffi.register_object("tvm.relax.attrs.Conv1DTransposeAttrs")
class Conv1DTransposeAttrs(Attrs):
    """Attributes for nn.conv1d_transpose"""


@tvm_ffi.register_object("tvm.relax.attrs.Pool1DAttrs")
class Pool1DAttrs(Attrs):
    """Attributes for nn.max_pool1d and nn.avg_pool1d"""


@tvm_ffi.register_object("tvm.relax.attrs.Pool3DAttrs")
class Pool3DAttrs(Attrs):
    """Attributes for nn.max_pool3d and nn.avg_pool3d"""


@tvm_ffi.register_object("tvm.relax.attrs.AdaptivePool1DAttrs")
class AdaptivePool1DAttrs(Attrs):
    """Attributes for 1d adaptive pool operator"""


@tvm_ffi.register_object("tvm.relax.attrs.AdaptivePool3DAttrs")
class AdaptivePool3DAttrs(Attrs):
    """Attributes for 3d adaptive pool operator"""


@tvm_ffi.register_object("tvm.relax.attrs.LeakyReluAttrs")
class LeakyReluAttrs(Attrs):
    """Attributes used in leaky_relu operator"""


@tvm_ffi.register_object("tvm.relax.attrs.SoftplusAttrs")
class SoftplusAttrs(Attrs):
    """Attributes used in softplus operator"""


@tvm_ffi.register_object("tvm.relax.attrs.PReluAttrs")
class PReluAttrs(Attrs):
    """Attributes used in prelu operator"""


@tvm_ffi.register_object("tvm.relax.attrs.PixelShuffleAttrs")
class PixelShuffleAttrs(Attrs):
    """Attributes used in pixel_shuffle operator"""


@tvm_ffi.register_object("tvm.relax.attrs.GroupNormAttrs")
class GroupNormAttrs(Attrs):
    """Attributes used in group_norm operator"""


@tvm_ffi.register_object("tvm.relax.attrs.RMSNormAttrs")
class RMSNormAttrs(Attrs):
    """Attributes used in rms_norm operator"""


@tvm_ffi.register_object("tvm.relax.attrs.NLLLossAttrs")
class NLLLossAttrs(Attrs):
    """Attributes used in nll_loss operator"""


@tvm_ffi.register_object("tvm.relax.attrs.AllReduceAttrs")
class AllReduceAttrs(Attrs):
    """Attributes used in allreduce operator"""


@tvm_ffi.register_object("tvm.relax.attrs.AllGatherAttrs")
class AllGatherAttrs(Attrs):
    """Attributes used in allgather operator"""


@tvm_ffi.register_object("tvm.relax.attrs.WrapParamAttrs")
class WrapParamAttrs(Attrs):
    """Attributes used in wrap_param operator"""


@tvm_ffi.register_object("tvm.relax.attrs.QuantizeAttrs")
class QuantizeAttrs(Attrs):
    """Attributes used in quantize/dequantize operators"""


@tvm_ffi.register_object("tvm.relax.attrs.GatherElementsAttrs")
class GatherElementsAttrs(Attrs):
    """Attributes for gather_elements operator"""


@tvm_ffi.register_object("tvm.relax.attrs.GatherNDAttrs")
class GatherNDAttrs(Attrs):
    """Attributes for gather_nd operator"""


@tvm_ffi.register_object("tvm.relax.attrs.MeshgridAttrs")
class MeshgridAttrs(Attrs):
    """Attributes for meshgrid operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ScatterElementsAttrs")
class ScatterElementsAttrs(Attrs):
    """Attributes for scatter_elements operator"""


@tvm_ffi.register_object("tvm.relax.attrs.ScatterNDAttrs")
class ScatterNDAttrs(Attrs):
    """Attributes for scatter_nd operator"""


@tvm_ffi.register_object("tvm.relax.attrs.SliceScatterAttrs")
class SliceScatterAttrs(Attrs):
    """Attributes for slice_scatter operator"""


@tvm_ffi.register_object("tvm.relax.attrs.OneHotAttrs")
class OneHotAttrs(Attrs):
    """Attributes for one_hot operator"""
