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
import tvm._ffi
from tvm.ir import Attrs


@tvm._ffi.register_object("relax.attrs.InitAttrs")
class InitAttrs(Attrs):
    """Attributes used in full/full_like, ones/ones_like, and zeros/zeros_like operator"""


@tvm._ffi.register_object("relax.attrs.TriluAttrs")
class TriluAttrs(Attrs):
    """Attributes used in tril and triu operator"""


@tvm._ffi.register_object("relax.attrs.TakeAttrs")
class TakeAttrs(Attrs):
    """Attributes used in take operator"""


@tvm._ffi.register_object("relax.attrs.StridedSliceAttrs")
class StridedSliceAttrs(Attrs):
    """Attributes used in strided_slice operator"""


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


@tvm._ffi.register_object("relax.attrs.ConcatAttrs")
class ConcatAttrs(Attrs):
    """Attributes for concat operator"""


@tvm._ffi.register_object("relax.attrs.SplitAttrs")
class SplitAttrs(Attrs):
    """Attributes used in split operator"""
