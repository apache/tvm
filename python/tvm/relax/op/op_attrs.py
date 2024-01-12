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
import tvm._ffi


@tvm._ffi.register_object("relax.attrs.CallTIRWithGradAttrs")
class CallTIRWithGradAttrs(Attrs):
    """Attributes used in call_tir_with_grad operator"""


@tvm._ffi.register_object("relax.attrs.InitAttrs")
class InitAttrs(Attrs):
    """Attributes used in full/full_like, ones/ones_like, and zeros/zeros_like operator"""


@tvm._ffi.register_object("relax.attrs.TriluAttrs")
class TriluAttrs(Attrs):
    """Attributes used in tril and triu operator"""


@tvm._ffi.register_object("relax.attrs.AstypeAttrs")
class AstypeAttrs(Attrs):
    """Attributes used in astype operator"""


@tvm._ffi.register_object("relax.attrs.TakeAttrs")
class TakeAttrs(Attrs):
    """Attributes used in take operator"""


@tvm._ffi.register_object("relax.attrs.StridedSliceAttrs")
class StridedSliceAttrs(Attrs):
    """Attributes used in strided_slice operator"""


@tvm._ffi.register_object("relax.attrs.MatmulAttrs")
class MatmulAttrs(Attrs):
    """Attributes for matmul operator"""


@tvm._ffi.register_object("relax.attrs.Conv2DAttrs")
class Conv2DAttrs(Attrs):
    """Attributes for nn.conv2d"""


@tvm._ffi.register_object("relax.attrs.Conv2DTransposeAttrs")
class Conv2DTransposeAttrs(Attrs):
    """Attributes for nn.conv2d_transpose"""


@tvm._ffi.register_object("relax.attrs.Pool2DAttrs")
class Pool2DAttrs(Attrs):
    """Attributes for nn.max_pool2d"""


@tvm._ffi.register_object("relax.attrs.AdaptivePool2DAttrs")
class AdaptivePool2DAttrs(Attrs):
    """Attributes for 2d adaptive pool operator"""


@tvm._ffi.register_object("relax.attrs.SoftmaxAttrs")
class SoftmaxAttrs(Attrs):
    """Attributes for nn.softmax"""


@tvm._ffi.register_object("relax.attrs.BatchNormAttrs")
class BatchNormAttrs(Attrs):
    """Attributes used in batch_norm operator"""


@tvm._ffi.register_object("relax.attrs.LayerNormAttrs")
class LayerNormAttrs(Attrs):
    """Attributes used in layer_norm operator"""


@tvm._ffi.register_object("relax.attrs.DropoutAttrs")
class DropoutAttrs(Attrs):
    """Attributes for dropout operator"""


@tvm._ffi.register_object("relax.attrs.StatisticalAttrs")
class StatisticalAttrs(Attrs):
    """Attributes used in statistical operator"""


@tvm._ffi.register_object("relax.attrs.ConcatAttrs")
class ConcatAttrs(Attrs):
    """Attributes for concat operator"""


@tvm._ffi.register_object("relax.attrs.ExpandDimsAttrs")
class ExpandDimsAttrs(Attrs):
    """Attributes for expand_dims operator"""


@tvm._ffi.register_object("relax.attrs.PermuteDimsAttrs")
class PermuteDimsAttrs(Attrs):
    """Attributes for permute_dims operator"""


@tvm._ffi.register_object("relax.attrs.SortAttrs")
class SortAttrs(Attrs):
    """Attributes for sort operator"""


@tvm._ffi.register_object("relax.attrs.ArgsortAttrs")
class ArgsortAttrs(Attrs):
    """Attributes for argsort operator"""


@tvm._ffi.register_object("relax.attrs.SplitAttrs")
class SplitAttrs(Attrs):
    """Attributes used in split operator"""


@tvm._ffi.register_object("relax.attrs.SqueezeAttrs")
class SqueezeAttrs(Attrs):
    """Attributes for squeeze operator"""


@tvm._ffi.register_object("relax.attrs.LayoutTransformAttrs")
class LayoutTransformAttrs(Attrs):
    """Attributes used in layout_transform operator"""


@tvm._ffi.register_object("relax.attrs.Resize2DAttrs")
class Resize2DAttrs(Attrs):
    """Attributes used in image resize2d operator"""


@tvm._ffi.register_object("relax.attrs.ArgmaxArgminAttrs")
class ArgmaxArgminAttrs(Attrs):
    """Attributes for argmax/argmin operator"""


@tvm._ffi.register_object("relax.attrs.RepeatAttrs")
class RepeatAttrs(Attrs):
    """Attributes for repeat operator"""


@tvm._ffi.register_object("relax.attrs.TileAttrs")
class TileAttrs(Attrs):
    """Attributes for tile operator"""


@tvm._ffi.register_object("relax.attrs.ScanopAttrs")
class ScanopAttrs(Attrs):
    """Attributes for scan operators"""


@tvm._ffi.register_object("relax.attrs.TopKAttrs")
class TopKAttrs(Attrs):
    """Attributes for topk operators"""


@tvm._ffi.register_object("relax.attrs.EinsumAttrs")
class EinsumAttrs(Attrs):
    """Attributes for einsum operator"""


@tvm._ffi.register_object("relax.attrs.FlipAttrs")
class FlipAttrs(Attrs):
    """Attributes for flip operator"""
