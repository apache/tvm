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

"""Definition of Hexagon operator strategy."""

# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import

from tvm import topi
from .generic import *
from .. import op as _op


@conv2d_strategy.register("hexagon")
def conv2d_strategy_hexagon(attrs, inputs, out_type, target):
    """Conv2d strategy for Hexagon"""
    strategy = _op.OpStrategy()
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout

    if data_layout == "NHWC" and kernel_layout == "HWIO":
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_nhwc),
            wrap_topi_schedule(topi.hexagon.schedule_conv2d_nhwc),
            name="conv2d.hexagon",
        )
        return strategy

    raise RuntimeError(
        "Unsupported layouts: data_layout:{}, kernel_layout:{}".format(data_layout, kernel_layout)
    )
