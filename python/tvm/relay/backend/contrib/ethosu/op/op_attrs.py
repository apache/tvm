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
"""The attributes node used for Arm(R) Ethos(TM)-U NPU Relay operators."""
from tvm.ir import Attrs
import tvm._ffi


@tvm._ffi.register_object("relay.attrs.EthosuConv2DAttrs")
class EthosuConv2DAttrs(Attrs):
    """Attributes for contrib.ethosu.conv2d."""


@tvm._ffi.register_object("relay.attrs.EthosuIdentityAttrs")
class EthosuIdentityAttrs(Attrs):
    """Attributes for contrib.ethosu.identity."""


@tvm._ffi.register_object("relay.attrs.EthosuDepthwiseConv2DAttrs")
class EthosuDepthwiseConv2DAttrs(Attrs):
    """Attributes for contrib.ethosu.depthwise_conv2d."""


@tvm._ffi.register_object("relay.attrs.EthosuPoolingAttrs")
class EthosuPooling2DAttrs(Attrs):
    """Attributes for contrib.ethosu.pooling."""


@tvm._ffi.register_object("relay.attrs.EthosuBinaryElementwiseAttrs")
class EthosuBinaryElementwiseAttrs(Attrs):
    """Attributes for contrib.ethosu.binary_elementwise"""


@tvm._ffi.register_object("relay.attrs.EthosuUnaryElementwiseAttrs")
class EthosuUnaryElementwiseAttrs(Attrs):
    """Attributes for contrib.ethosu.unary_elementwise"""
