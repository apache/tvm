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
"""The attributes node used for Relay operators"""
from tvm.ir import Attrs
import tvm._ffi


@tvm._ffi.register_object("relay.attrs.Conv1DAttrs")
class Conv1DAttrs(Attrs):
    """Attributes for nn.conv1d"""


@tvm._ffi.register_object("relay.attrs.Conv2DAttrs")
class Conv2DAttrs(Attrs):
    """Attributes for nn.conv2d"""


@tvm._ffi.register_object("relay.attrs.Conv2DWinogradAttrs")
class Conv2DWinogradAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_without_weight_transform"""


@tvm._ffi.register_object("relay.attrs.Conv3DAttrs")
class Conv3DAttrs(Attrs):
    """Attributes for nn.conv3d"""


@tvm._ffi.register_object("relay.attrs.Conv3DWinogradAttrs")
class Conv3DWinogradAttrs(Attrs):
    """Attributes for nn.contrib_conv3d_winograd_without_weight_transform"""


@tvm._ffi.register_object("relay.attrs.ConvWinogradWeightTransformAttrs")
class ConvWinogradWeightTransformAttrs(Attrs):
    """Attributes for nn.contrib_convNd_winograd_weight_transform"""


@tvm._ffi.register_object("relay.attrs.Conv2DWinogradNNPACKWeightTransformAttrs")
class Conv2DWinogradNNPACKWeightTransformAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_nnpack_weight_transform"""


@tvm._ffi.register_object("relay.attrs.GlobalPool2DAttrs")
class GlobalPool2DAttrs(Attrs):
    """Attributes for nn.global_pool"""


@tvm._ffi.register_object("relay.attrs.BiasAddAttrs")
class BiasAddAttrs(Attrs):
    """Atttribute of nn.bias_add"""


@tvm._ffi.register_object("relay.attrs.DenseAttrs")
class DenseAttrs(Attrs):
    """Attributes for nn.dense"""


@tvm._ffi.register_object("relay.attrs.SoftmaxAttrs")
class SoftmaxAttrs(Attrs):
    """Attributes for nn.softmax"""


@tvm._ffi.register_object("relay.attrs.FIFOBufferAttrs")
class FIFOBufferAttrs(Attrs):
    """Attributes for nn.fifo_buffer"""


@tvm._ffi.register_object("relay.attrs.UpSamplingAttrs")
class UpSamplingAttrs(Attrs):
    """Attributes for nn.upsampling"""

@tvm._ffi.register_object("relay.attrs.UpSampling3DAttrs")
class UpSampling3DAttrs(Attrs):
    """Attributes for nn.upsampling3d"""

@tvm._ffi.register_object("relay.attrs.PadAttrs")
class PadAttrs(Attrs):
    """Attributes for nn.pad"""

@tvm._ffi.register_object("relay.attrs.MirrorPadAttrs")
class MirrorPadAttrs(Attrs):
    """Attributes for nn.mirror_pad"""

@tvm._ffi.register_object("relay.attrs.LeakyReluAttrs")
class LeakyReluAttrs(Attrs):
    """Attributes for nn.leaky_relu"""


@tvm._ffi.register_object("relay.attrs.PReluAttrs")
class PReluAttrs(Attrs):
    """Attributes for nn.prelu"""


@tvm._ffi.register_object("relay.attrs.DropoutAttrs")
class DropoutAttrs(Attrs):
    """Attributes for nn.dropout"""


@tvm._ffi.register_object("relay.attrs.BatchNormAttrs")
class BatchNormAttrs(Attrs):
    """Attributes for nn.batch_norm"""


@tvm._ffi.register_object("relay.attrs.LRNAttrs")
class LRNAttrs(Attrs):
    """Attributes for nn.lrn"""


@tvm._ffi.register_object("relay.attrs.L2NormalizeAttrs")
class L2NormalizeAttrs(Attrs):
    """Attributes for nn.l2_normalize"""


@tvm._ffi.register_object("relay.attrs.DeformableConv2DAttrs")
class DeformableConv2DAttrs(Attrs):
    """Attributes for nn.deformable_conv2d"""


@tvm._ffi.register_object("relay.attrs.ResizeAttrs")
class ResizeAttrs(Attrs):
    """Attributes for image.resize"""


@tvm._ffi.register_object("relay.attrs.CropAndResizeAttrs")
class CropAndResizeAttrs(Attrs):
    """Attributes for image.crop_and_resize"""


@tvm._ffi.register_object("relay.attrs.Dilation2DAttrs")
class Dilation2DAttrs(Attrs):
    """Attributes for image.dilation2d"""


@tvm._ffi.register_object("relay.attrs.ArgsortAttrs")
class ArgsortAttrs(Attrs):
    """Attributes for algorithm.argsort"""


@tvm._ffi.register_object("relay.attrs.OnDeviceAttrs")
class OnDeviceAttrs(Attrs):
    """Attributes for annotation.on_device"""


@tvm._ffi.register_object("relay.attrs.DebugAttrs")
class DebugAttrs(Attrs):
    """Attributes for debug"""


@tvm._ffi.register_object("relay.attrs.CompilerAttrs")
class CompilerAttrs(Attrs):
    """Attributes for compiler"""


@tvm._ffi.register_object("relay.attrs.DeviceCopyAttrs")
class DeviceCopyAttrs(Attrs):
    """Attributes for annotation.device_copy"""


@tvm._ffi.register_object("relay.attrs.CastAttrs")
class CastAttrs(Attrs):
    """Attributes for transform.cast"""


@tvm._ffi.register_object("relay.attrs.ConcatenateAttrs")
class ConcatenateAttrs(Attrs):
    """Attributes for tensor.concatenate"""


@tvm._ffi.register_object("relay.attrs.TransposeAttrs")
class TransposeAttrs(Attrs):
    """Attributes for transform.transpose"""


@tvm._ffi.register_object("relay.attrs.ReshapeAttrs")
class ReshapeAttrs(Attrs):
    """Attributes for transform.reshape"""


@tvm._ffi.register_object("relay.attrs.TakeAttrs")
class TakeAttrs(Attrs):
    """Attributes for transform.take"""


@tvm._ffi.register_object("relay.attrs.InitOpAttrs")
class InitOpAttrs(Attrs):
    """Attributes for ops specifying a tensor"""


@tvm._ffi.register_object("relay.attrs.ArangeAttrs")
class ArangeAttrs(Attrs):
    """Attributes used in arange operators"""


@tvm._ffi.register_object("relay.attrs.StackAttrs")
class StackAttrs(Attrs):
    """Attributes used in stack operators"""


@tvm._ffi.register_object("relay.attrs.RepeatAttrs")
class RepeatAttrs(Attrs):
    """Attributes used in repeat operators"""


@tvm._ffi.register_object("relay.attrs.TileAttrs")
class TileAttrs(Attrs):
    """Attributes used in tile operators"""


@tvm._ffi.register_object("relay.attrs.ReverseAttrs")
class ReverseAttrs(Attrs):
    """Attributes used in reverse operators"""


@tvm._ffi.register_object("relay.attrs.SqueezeAttrs")
class SqueezeAttrs(Attrs):
    """Attributes used in squeeze operators"""


@tvm._ffi.register_object("relay.attrs.SplitAttrs")
class SplitAttrs(Attrs):
    """Attributes for transform.split"""


@tvm._ffi.register_object("relay.attrs.StridedSliceAttrs")
class StridedSliceAttrs(Attrs):
    """Attributes for transform.stranded_slice"""


@tvm._ffi.register_object("relay.attrs.SliceLikeAttrs")
class SliceLikeAttrs(Attrs):
    """Attributes for transform.slice_like"""


@tvm._ffi.register_object("relay.attrs.ClipAttrs")
class ClipAttrs(Attrs):
    """Attributes for transform.clip"""


@tvm._ffi.register_object("relay.attrs.LayoutTransformAttrs")
class LayoutTransformAttrs(Attrs):
    """Attributes for transform.layout_transform"""


@tvm._ffi.register_object("relay.attrs.ShapeOfAttrs")
class ShapeOfAttrs(Attrs):
    """Attributes for tensor.shape_of"""


@tvm._ffi.register_object("relay.attrs.MultiBoxPriorAttrs")
class MultiBoxPriorAttrs(Attrs):
    """Attributes for vision.multibox_prior"""


@tvm._ffi.register_object("relay.attrs.MultiBoxTransformLocAttrs")
class MultiBoxTransformLocAttrs(Attrs):
    """Attributes for vision.multibox_transform_loc"""


@tvm._ffi.register_object("relay.attrs.GetValidCountsAttrs")
class GetValidCountsAttrs(Attrs):
    """Attributes for vision.get_valid_counts"""


@tvm._ffi.register_object("relay.attrs.NonMaximumSuppressionAttrs")
class NonMaximumSuppressionAttrs(Attrs):
    """Attributes for vision.non_maximum_suppression"""


@tvm._ffi.register_object("relay.attrs.ROIAlignAttrs")
class ROIAlignAttrs(Attrs):
    """Attributes for vision.roi_align"""


@tvm._ffi.register_object("relay.attrs.ROIPoolAttrs")
class ROIPoolAttrs(Attrs):
    """Attributes for vision.roi_pool"""


@tvm._ffi.register_object("relay.attrs.YoloReorgAttrs")
class YoloReorgAttrs(Attrs):
    """Attributes for vision.yolo_reorg"""


@tvm._ffi.register_object("relay.attrs.ProposalAttrs")
class ProposalAttrs(Attrs):
    """Attributes used in proposal operators"""


@tvm._ffi.register_object("relay.attrs.MaxPool2DAttrs")
class MaxPool2DAttrs(Attrs):
    """Attributes used in max_pool2d operators"""


@tvm._ffi.register_object("relay.attrs.AvgPool2DAttrs")
class AvgPool2DAttrs(Attrs):
    """Attributes used in avg_pool2d operators"""


@tvm._ffi.register_object("relay.attrs.MaxPool1DAttrs")
class MaxPool1DAttrs(Attrs):
    """Attributes used in max_pool1d operators"""


@tvm._ffi.register_object("relay.attrs.AvgPool1DAttrs")
class AvgPool1DAttrs(Attrs):
    """Attributes used in avg_pool1d operators"""


@tvm._ffi.register_object("relay.attrs.MaxPool3DAttrs")
class MaxPool3DAttrs(Attrs):
    """Attributes used in max_pool3d operators"""


@tvm._ffi.register_object("relay.attrs.AvgPool3DAttrs")
class AvgPool3DAttrs(Attrs):
    """Attributes used in avg_pool3d operators"""


@tvm._ffi.register_object("relay.attrs.BitPackAttrs")
class BitPackAttrs(Attrs):
    """Attributes used in bitpack operator"""


@tvm._ffi.register_object("relay.attrs.BinaryConv2DAttrs")
class BinaryConv2DAttrs(Attrs):
    """Attributes used in bitserial conv2d operators"""


@tvm._ffi.register_object("relay.attrs.BinaryDenseAttrs")
class BinaryDenseAttrs(Attrs):
    """Attributes used in bitserial dense operators"""


@tvm._ffi.register_object("relay.attrs.Conv2DTransposeAttrs")
class Conv2DTransposeAttrs(Attrs):
    """Attributes used in Transposed Conv2D operators"""


@tvm._ffi.register_object("relay.attrs.DilateAttrs")
class DilateAttrs(Attrs):
    """Attributes used in dilate operators"""


@tvm._ffi.register_object("relay.attrs.SubPixelAttrs")
class SubPixelAttrs(Attrs):
    """Attributes used in depth to space and space to depth operators"""
