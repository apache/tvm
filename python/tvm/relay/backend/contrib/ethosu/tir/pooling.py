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
# pylint: disable=invalid-name, unused-argument
"""Extract information from the pooling operators in TIR."""
from typing import Tuple
import tvm
from .utils import get_outer_loops, get_op_attrs, get_loads, get_stores
from .dma import get_ifm_params, get_ofm_params
from .spec import SerialKernel, SerialActivation, SerialPooling
from .producers_consumers import ProducersConsumers


def get_pooling_params(
    stmt: tvm.tir.AttrStmt, producers_consumers: ProducersConsumers
) -> Tuple[SerialPooling, tvm.tir.Var, tvm.tir.Var]:
    """Get the parameters necessary to construct a call_extern for a pooling.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a convolution loop nest.
    producers_consumers: ProducersConsumers
        It associates pointers with the loop nest that produces
        their values and with the loop nest that consumes their values.

    Returns
    -------
    SerialPooling
        The parameters needed to construct a 2D convolution.
    output_pointer : tvm.tir.Var
        The output pointer of the convolution operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the convolution output pointer.
    is_allocator : bool
        Whether this operator allocates its output.
    """
    attrs, body = get_op_attrs(stmt)
    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    rh = inner
    rw = rh.body
    # loads = [output, input, LUT, LUT]
    loads = get_loads(rw.body)
    # stores = [output]
    stores = get_stores(rw.body)
    input_pointer = loads[1].buffer.data
    output_pointer = stores[0].buffer.data
    # Get feature map info
    serial_ifm, serial_padding = get_ifm_params(input_pointer, producers_consumers, stmt)
    serial_ofm, serial_block_config, replace_pointer, is_allocator = get_ofm_params(
        output_pointer, producers_consumers, stmt
    )
    # Get kernel info
    serial_kernel = SerialKernel(
        width=int(rw.extent),
        height=int(rh.extent),
        stride_w=int(attrs["stride_w"]),
        stride_h=int(attrs["stride_h"]),
        dilation_w=1,
        dilation_h=1,
    )

    # Get activation info
    serial_activation = SerialActivation(
        op=attrs["activation"], clip_min=attrs["clip_min"], clip_max=attrs["clip_max"]
    )
    return (
        SerialPooling(
            ifm=serial_ifm,
            ofm=serial_ofm,
            pooling_type=attrs["pooling_type"],
            pool_shape=serial_kernel,
            padding=serial_padding,
            activation=serial_activation,
            rounding_mode=attrs["rounding_mode"],
            upscale=attrs["upscale"],
            block_config=serial_block_config,
        ),
        output_pointer,
        replace_pointer,
        is_allocator,
    )
