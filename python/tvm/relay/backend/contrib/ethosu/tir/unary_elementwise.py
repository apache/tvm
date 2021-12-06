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
"""Extract information from the unary_elementwise operators in TIR."""
from tvm import tir
from .utils import get_outer_loops, get_op_attrs
from .dma import get_ifm_params, get_ofm_params
from .spec import SerialActivation, SerialUnaryElementwise


def get_unary_elementwise_params(stmt, producers, consumers):
    """Get the parameters necessary to construct a call_extern for a unary_elementwise.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a unary elementwise loop nest.
    producers : dict of tvm.tir.Var to tvm.tir.AttrStmt
        A dictionary to associate pointers with the loop nest
        that produces their values.
    consumers : dict of tvm.tir.Var to tvm.tir.AttrStmt
        A dictionary to associate pointers with the loop nest
        that consumes their values.

    Returns
    -------
    SerialUnaryElementwise
        The parameters needed to construct a unary elementwise operator.
    output_pointer : tvm.tir.Var
        The output pointer of the unary elementwise operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the unary elementwise output pointer.

    """
    attrs, body = get_op_attrs(stmt)

    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    input_pointer = None
    if isinstance(inner.value, tir.expr.Select):
        # ABS
        input_pointer = inner.value.condition.b.buffer_var
    if isinstance(inner.value, tir.expr.Sub):
        # CLZ
        input_pointer = inner.value.b.args[0].buffer_var
    output_pointer = inner.buffer_var
    # Get feature map info
    serial_ifm, _ = get_ifm_params(input_pointer, producers)
    serial_ofm, replace_pointer = get_ofm_params(output_pointer, consumers)
    # Get activation info
    serial_activation = SerialActivation(
        op=attrs["activation"], clip_min=attrs["clip_min"], clip_max=attrs["clip_max"]
    )
    return (
        SerialUnaryElementwise(
            ifm=serial_ifm,
            ofm=serial_ofm,
            operator_type=attrs["operator_type"],
            activation=serial_activation,
            rounding_mode=attrs["rounding_mode"],
        ),
        output_pointer,
        replace_pointer,
    )
