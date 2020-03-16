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

import tvm._ffi
from tvm.ir import Attrs


def _register_relay_attr_node(type_key=None):
    """Register a Relay attribute node.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    return tvm._ffi.register_object(
        "relay.attrs." + type_key.__name__)(type_key)


@_register_relay_attr_node
class Conv1DAttrs(Attrs):
    """Attributes for nn.conv1d"""


@_register_relay_attr_node
class Conv2DAttrs(Attrs):
    """Attributes for nn.conv2d"""


@_register_relay_attr_node
class Conv2DWinogradAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_without_weight_transform"""


@_register_relay_attr_node
class Conv2DWinogradWeightTransformAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_weight_transform"""


@_register_relay_attr_node
class Conv2DWinogradNNPACKWeightTransformAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_nnpack_weight_transform"""


@_register_relay_attr_node
class GlobalPool2DAttrs(Attrs):
    """Attributes for nn.global_pool"""


@_register_relay_attr_node
class BiasAddAttrs(Attrs):
    """Atttribute of nn.bias_add"""


@_register_relay_attr_node
class DenseAttrs(Attrs):
    """Attributes for nn.dense"""


@_register_relay_attr_node
class FIFOBufferAttrs(Attrs):
    """Attributes for nn.fifo_buffer"""


@_register_relay_attr_node
class UpSamplingAttrs(Attrs):
    """Attributes for nn.upsampling"""

@_register_relay_attr_node
class UpSampling3DAttrs(Attrs):
    """Attributes for nn.upsampling3d"""

@_register_relay_attr_node
class PadAttrs(Attrs):
    """Attributes for nn.pad"""

@_register_relay_attr_node
class MirrorPadAttrs(Attrs):
    """Attributes for nn.mirror_pad"""

@_register_relay_attr_node
class LeakyReluAttrs(Attrs):
    """Attributes for nn.leaky_relu"""


@_register_relay_attr_node
class PReluAttrs(Attrs):
    """Attributes for nn.prelu"""


@_register_relay_attr_node
class DropoutAttrs(Attrs):
    """Attributes for nn.dropout"""


@_register_relay_attr_node
class BatchNormAttrs(Attrs):
    """Attributes for nn.batch_norm"""


@_register_relay_attr_node
class LRNAttrs(Attrs):
    """Attributes for nn.lrn"""


@_register_relay_attr_node
class L2NormalizeAttrs(Attrs):
    """Attributes for nn.l2_normalize"""


@_register_relay_attr_node
class DeformableConv2DAttrs(Attrs):
    """Attributes for nn.deformable_conv2d"""


@_register_relay_attr_node
class ResizeAttrs(Attrs):
    """Attributes for image.resize"""

@_register_relay_attr_node
class CropAndResizeAttrs(Attrs):
    """Attributes for image.crop_and_resize"""

@_register_relay_attr_node
class ArgsortAttrs(Attrs):
    """Attributes for algorithm.argsort"""


@_register_relay_attr_node
class OnDeviceAttrs(Attrs):
    """Attributes for annotation.on_device"""


@_register_relay_attr_node
class DebugAttrs(Attrs):
    """Attributes for debug"""


@_register_relay_attr_node
class DeviceCopyAttrs(Attrs):
    """Attributes for tensor.device_copy"""


@_register_relay_attr_node
class CastAttrs(Attrs):
    """Attributes for transform.cast"""


@_register_relay_attr_node
class ConcatenateAttrs(Attrs):
    """Attributes for tensor.concatenate"""


@_register_relay_attr_node
class TransposeAttrs(Attrs):
    """Attributes for transform.transpose"""


@_register_relay_attr_node
class ReshapeAttrs(Attrs):
    """Attributes for transform.reshape"""


@_register_relay_attr_node
class TakeAttrs(Attrs):
    """Attributes for transform.take"""


@_register_relay_attr_node
class InitOpAttrs(Attrs):
    """Attributes for ops specifying a tensor"""


@_register_relay_attr_node
class ArangeAttrs(Attrs):
    """Attributes used in arange operators"""


@_register_relay_attr_node
class StackAttrs(Attrs):
    """Attributes used in stack operators"""


@_register_relay_attr_node
class RepeatAttrs(Attrs):
    """Attributes used in repeat operators"""


@_register_relay_attr_node
class TileAttrs(Attrs):
    """Attributes used in tile operators"""


@_register_relay_attr_node
class ReverseAttrs(Attrs):
    """Attributes used in reverse operators"""


@_register_relay_attr_node
class SqueezeAttrs(Attrs):
    """Attributes used in squeeze operators"""


@_register_relay_attr_node
class SplitAttrs(Attrs):
    """Attributes for transform.split"""


@_register_relay_attr_node
class StridedSliceAttrs(Attrs):
    """Attributes for transform.stranded_slice"""


@_register_relay_attr_node
class SliceLikeAttrs(Attrs):
    """Attributes for transform.slice_like"""


@_register_relay_attr_node
class ClipAttrs(Attrs):
    """Attributes for transform.clip"""


@_register_relay_attr_node
class LayoutTransformAttrs(Attrs):
    """Attributes for transform.layout_transform"""


@_register_relay_attr_node
class ShapeOfAttrs(Attrs):
    """Attributes for tensor.shape_of"""


@_register_relay_attr_node
class MultiBoxPriorAttrs(Attrs):
    """Attributes for vision.multibox_prior"""


@_register_relay_attr_node
class MultiBoxTransformLocAttrs(Attrs):
    """Attributes for vision.multibox_transform_loc"""


@_register_relay_attr_node
class GetValidCountsAttrs(Attrs):
    """Attributes for vision.get_valid_counts"""


@_register_relay_attr_node
class NonMaximumSuppressionAttrs(Attrs):
    """Attributes for vision.non_maximum_suppression"""


@_register_relay_attr_node
class ROIAlignAttrs(Attrs):
    """Attributes for vision.roi_align"""


@_register_relay_attr_node
class ROIPoolAttrs(Attrs):
    """Attributes for vision.roi_pool"""


@_register_relay_attr_node
class YoloReorgAttrs(Attrs):
    """Attributes for vision.yolo_reorg"""


@_register_relay_attr_node
class ProposalAttrs(Attrs):
    """Attributes used in proposal operators"""


@_register_relay_attr_node
class MaxPool2DAttrs(Attrs):
    """Attributes used in max_pool2d operators"""


@_register_relay_attr_node
class AvgPool2DAttrs(Attrs):
    """Attributes used in avg_pool2d operators"""


@_register_relay_attr_node
class MaxPool1DAttrs(Attrs):
    """Attributes used in max_pool1d operators"""


@_register_relay_attr_node
class AvgPool1DAttrs(Attrs):
    """Attributes used in avg_pool1d operators"""


@_register_relay_attr_node
class MaxPool3DAttrs(Attrs):
    """Attributes used in max_pool3d operators"""


@_register_relay_attr_node
class AvgPool3DAttrs(Attrs):
    """Attributes used in avg_pool3d operators"""


@_register_relay_attr_node
class BitPackAttrs(Attrs):
    """Attributes used in bitpack operator"""


@_register_relay_attr_node
class BinaryConv2DAttrs(Attrs):
    """Attributes used in bitserial conv2d operators"""


@_register_relay_attr_node
class BinaryDenseAttrs(Attrs):
    """Attributes used in bitserial dense operators"""


@_register_relay_attr_node
class Conv2DTransposeAttrs(Attrs):
    """Attributes used in Transposed Conv2D operators"""


@_register_relay_attr_node
class SubPixelAttrs(Attrs):
    """Attributes used in depth to space and space to depth operators"""
