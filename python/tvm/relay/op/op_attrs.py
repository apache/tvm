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

from ...attrs import Attrs
from ..base import register_relay_attr_node

@register_relay_attr_node
class Conv2DAttrs(Attrs):
    """Attributes for nn.conv2d"""


@register_relay_attr_node
class Conv2DWinogradAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_without_weight_transform"""


@register_relay_attr_node
class Conv2DWinogradWeightTransformAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_weight_transform"""


@register_relay_attr_node
class Conv2DWinogradNNPACKWeightTransformAttrs(Attrs):
    """Attributes for nn.contrib_conv2d_winograd_nnpack_weight_transform"""


@register_relay_attr_node
class GlobalPool2DAttrs(Attrs):
    """Attributes for nn.global_pool"""


@register_relay_attr_node
class BiasAddAttrs(Attrs):
    """Atttribute of nn.bias_add"""


@register_relay_attr_node
class DenseAttrs(Attrs):
    """Attributes for nn.dense"""


@register_relay_attr_node
class UpSamplingAttrs(Attrs):
    """Attributes for nn.upsampling"""

@register_relay_attr_node
class PadAttrs(Attrs):
    """Attributes for nn.pad"""


@register_relay_attr_node
class LeakyReluAttrs(Attrs):
    """Attributes for nn.leaky_relu"""


@register_relay_attr_node
class PReluAttrs(Attrs):
    """Attributes for nn.prelu"""


@register_relay_attr_node
class DropoutAttrs(Attrs):
    """Attributes for nn.dropout"""


@register_relay_attr_node
class BatchNormAttrs(Attrs):
    """Attributes for nn.batch_norm"""


@register_relay_attr_node
class LRNAttrs(Attrs):
    """Attributes for nn.lrn"""


@register_relay_attr_node
class L2NormalizeAttrs(Attrs):
    """Attributes for nn.l2_normalize"""


@register_relay_attr_node
class DeformableConv2DAttrs(Attrs):
    """Attributes for nn.deformable_conv2d"""


@register_relay_attr_node
class ResizeAttrs(Attrs):
    """Attributes for image.resize"""


@register_relay_attr_node
class ArgsortAttrs(Attrs):
    """Attributes for algorithm.argsort"""


@register_relay_attr_node
class OnDeviceAttrs(Attrs):
    """Attributes for annotation.on_device"""


@register_relay_attr_node
class DebugAttrs(Attrs):
    """Attributes for debug"""


@register_relay_attr_node
class DeviceCopyAttrs(Attrs):
    """Attributes for tensor.device_copy"""


@register_relay_attr_node
class CastAttrs(Attrs):
    """Attributes for transform.cast"""


@register_relay_attr_node
class ConcatenateAttrs(Attrs):
    """Attributes for tensor.concatenate"""


@register_relay_attr_node
class TransposeAttrs(Attrs):
    """Attributes for transform.transpose"""


@register_relay_attr_node
class ReshapeAttrs(Attrs):
    """Attributes for transform.reshape"""


@register_relay_attr_node
class TakeAttrs(Attrs):
    """Attributes for transform.take"""


@register_relay_attr_node
class InitOpAttrs(Attrs):
    """Attributes for ops specifying a tensor"""


@register_relay_attr_node
class ArangeAttrs(Attrs):
    """Attributes used in arange operators"""


@register_relay_attr_node
class StackAttrs(Attrs):
    """Attributes used in stack operators"""


@register_relay_attr_node
class RepeatAttrs(Attrs):
    """Attributes used in repeat operators"""


@register_relay_attr_node
class TileAttrs(Attrs):
    """Attributes used in tile operators"""


@register_relay_attr_node
class ReverseAttrs(Attrs):
    """Attributes used in reverse operators"""


@register_relay_attr_node
class SqueezeAttrs(Attrs):
    """Attributes used in squeeze operators"""


@register_relay_attr_node
class SplitAttrs(Attrs):
    """Attributes for transform.split"""


@register_relay_attr_node
class StridedSliceAttrs(Attrs):
    """Attributes for transform.stranded_slice"""


@register_relay_attr_node
class SliceLikeAttrs(Attrs):
    """Attributes for transform.slice_like"""


@register_relay_attr_node
class ClipAttrs(Attrs):
    """Attributes for transform.clip"""


@register_relay_attr_node
class LayoutTransformAttrs(Attrs):
    """Attributes for transform.layout_transform"""


@register_relay_attr_node
class ShapeOfAttrs(Attrs):
    """Attributes for tensor.shape_of"""


@register_relay_attr_node
class MultiBoxPriorAttrs(Attrs):
    """Attributes for vision.multibox_prior"""


@register_relay_attr_node
class MultiBoxTransformLocAttrs(Attrs):
    """Attributes for vision.multibox_transform_loc"""


@register_relay_attr_node
class GetValidCountsAttrs(Attrs):
    """Attributes for vision.get_valid_counts"""


@register_relay_attr_node
class NonMaximumSuppressionAttrs(Attrs):
    """Attributes for vision.non_maximum_suppression"""


@register_relay_attr_node
class ROIAlignAttrs(Attrs):
    """Attributes for vision.roi_align"""


@register_relay_attr_node
class ROIPoolAttrs(Attrs):
    """Attributes for vision.roi_pool"""


@register_relay_attr_node
class YoloReorgAttrs(Attrs):
    """Attributes for vision.yolo_reorg"""


@register_relay_attr_node
class ProposalAttrs(Attrs):
    """Attributes used in proposal operators"""
