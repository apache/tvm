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
"""Extract information from the identity operator in TIR."""
from typing import Dict, Tuple
import tvm
from .dma import get_read_params, get_write_params
from .spec import SerialKernel, SerialActivation, SerialPooling, SerialPadding
from .utils import get_op_attrs


def get_identity_params(
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
        The parameters needed to construct a 2D pooling.
    output_pointer : tvm.tir.Var
        The output pointer of the pooling operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the pooling output pointer.

    """
    attrs, _ = get_op_attrs(stmt)
    # Find the inner loop
    while hasattr(stmt, "body"):
        stmt = stmt.body

    input_pointer = stmt.value.buffer_var
    output_pointer = stmt.buffer_var

    read = producers[input_pointer]
    write = consumers[output_pointer]

    serial_ifm, _, _ = get_read_params(read)
    serial_ofm, _, write_output_pointer = get_write_params(write)

    replace_pointer = write_output_pointer

    # TODO (maybe): Support stand alone RELU through clamping in identity
    serial_activation = SerialActivation(op=attrs["activation"], clip_min=0, clip_max=0)

    # Create a serialized identity pooling to be run on the NPU
    return (
        SerialPooling(
            ifm=serial_ifm,
            ofm=serial_ofm,
            pooling_type="AVG",
            pool_shape=SerialKernel(1, 1, 1, 1, 1, 1),
            padding=SerialPadding(0, 0, 0, 0),
            activation=serial_activation,
            upscale="NONE",
        ),
        output_pointer,
        replace_pointer,
    )
