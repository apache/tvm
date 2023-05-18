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
"""QNN Conv2d alter op functions for Hexagon"""

from tvm import relay
from ...nn import qnn_conv2d_alter_layout
from ...utils import get_const_tuple


@qnn_conv2d_alter_layout.register("hexagon")
def _alter_qnn_conv2d_layout(attrs, inputs, tinfos, _out_type):
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data_tensor, kernel_tensor, _, _, _, _ = tinfos

    if (
        "int8" in data_tensor.dtype
        and "int8" in kernel_tensor.dtype
        and data_layout == "NCHW"
        and kernel_layout == "OIHW"
    ):
        out_channel, in_channel, _, _ = get_const_tuple(kernel_tensor.shape)

        if out_channel % 32 != 0 or in_channel % 4 != 0:
            return None

        n_elems = 4
        oc_bn = 32
        ic_bn = min(in_channel, 32)

        new_attrs = dict(attrs)
        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = f"NCHW{ic_bn}c"
        new_attrs["kernel_layout"] = f"OIHW{ic_bn // n_elems:n}i{oc_bn:n}o{n_elems:n}i"
        new_attrs["out_layout"] = f"NCHW{oc_bn}c"

        return relay.qnn.op.conv2d(*inputs, **new_attrs)

    return None
