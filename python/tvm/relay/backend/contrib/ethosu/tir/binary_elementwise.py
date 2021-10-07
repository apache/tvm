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
"""Extract information from the binary_elementwise operators in TIR."""
from typing import Dict, Tuple
import tvm
from .utils import get_outer_loops, get_op_attrs
from .dma import get_ifm_params, get_ofm_params
from .spec import SerialActivation, SerialBinaryElementwise


def get_binary_elementwise_params(
    stmt: tvm.tir.AttrStmt,
    producers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
    consumers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
) -> Tuple[SerialBinaryElementwise, tvm.tir.Var, tvm.tir.Var]:
    """Get the parameters necessary to construct a call_extern for a binary_elementwise.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a binary elementwise loop nest.
    producers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that produces their values.
    consumers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that consumes their values.

    Returns
    -------
    SerialBinaryElementwise
        The parameters needed to construct a binary elementwise operator.
    output_pointer : tvm.tir.Var
        The output pointer of the binary elementwise operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the binary elementwise output pointer.
    """
    attrs, body = get_op_attrs(stmt)
    reversed_operands = attrs["reversed_operands"]

    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    input_pointer = inner.value.a.buffer_var
    input_pointer1 = inner.value.b.buffer_var
    if reversed_operands:
        input_pointer, input_pointer1 = input_pointer1, input_pointer
    output_pointer = inner.buffer_var
    # Get feature map info
    serial_ifm, _ = get_ifm_params(input_pointer, producers)
    serial_ifm2, _ = get_ifm_params(input_pointer1, producers)
    serial_ofm, replace_pointer = get_ofm_params(output_pointer, consumers)
    # Get activation info
    serial_activation = SerialActivation(
        op=attrs["activation"], clip_min=attrs["clip_min"], clip_max=attrs["clip_max"]
    )
    return (
        SerialBinaryElementwise(
            ifm=serial_ifm,
            ifm2=serial_ifm2,
            ofm=serial_ofm,
            operator_type=attrs["operator_type"],
            reversed_operands=reversed_operands,
            activation=serial_activation,
        ),
        output_pointer,
        replace_pointer,
    )
