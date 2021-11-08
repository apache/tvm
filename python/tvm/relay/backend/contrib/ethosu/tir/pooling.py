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
from typing import Dict, Tuple
import tvm
from .utils import get_outer_loops, get_op_attrs
from .dma import get_ifm_params, get_ofm_params
from .spec import SerialKernel, SerialActivation, SerialPooling


def get_pooling_params(
    stmt: tvm.tir.AttrStmt,
    producers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
    consumers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
) -> Tuple[SerialPooling, tvm.tir.Var, tvm.tir.Var]:
    """Get the parameters necessary to construct a call_extern for a pooling.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a convolution loop nest.
    producers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that produces their values.
    consumers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that consumes their values.

    Returns
    -------
    SerialPooling
        The parameters needed to construct a 2D convolution.
    output_pointer : tvm.tir.Var
        The output pointer of the convolution operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the convolution output pointer.
    """
    attrs, body = get_op_attrs(stmt)
    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    rh = inner
    rw = rh.body
    compute = rw.body.value.b
    input_pointer = compute.buffer_var
    output_pointer = rw.body.buffer_var
    # Get feature map info
    serial_ifm, serial_padding = get_ifm_params(input_pointer, producers)
    serial_ofm, replace_pointer = get_ofm_params(output_pointer, consumers)
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
            upscale="NONE",
        ),
        output_pointer,
        replace_pointer,
    )
