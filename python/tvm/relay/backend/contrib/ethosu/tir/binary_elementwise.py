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
from typing import Tuple
import tvm
from .utils import get_outer_loops, get_op_attrs
from .dma import get_ifm_params, get_ofm_params
from .spec import SerialActivation, SerialBinaryElementwise
from .producers_consumers import ProducersConsumers


def ignore_cast(tir_load: tvm.tir.expr.Load) -> tvm.tir.Var:
    """When the datatype of the ifm, ifm2 and ofm do not match,
    casts are inserted in TE to handle the difference in these types.
    Since TIR is not directly run on the NPU we can simply ignore
    these, and allow the NPU to handle the difference in datatypes
    itself.

    Parameters
    ----------
    tir_load : tvm.tir.expr.Load

    Returns
    -------
    tvm.tir.Var
    """
    return tir_load.value if isinstance(tir_load, tvm.tir.Cast) else tir_load


def get_binary_elementwise_params(
    stmt: tvm.tir.AttrStmt, producers_consumers: ProducersConsumers
) -> Tuple[SerialBinaryElementwise, tvm.tir.Var, tvm.tir.Var]:
    """Get the parameters necessary to construct a call_extern for a binary_elementwise.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a binary elementwise loop nest.
    producers_consumers: ProducersConsumers
        It associates pointers with the loop nest that produces
        their values and with the loop nest that consumes their values.

    Returns
    -------
    SerialBinaryElementwise
        The parameters needed to construct a binary elementwise operator.
    output_pointer : tvm.tir.Var
        The output pointer of the binary elementwise operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the binary elementwise output pointer.
    is_allocator : bool
        Whether this operator allocates its output.

    """
    attrs, body = get_op_attrs(stmt)
    reversed_operands = attrs["reversed_operands"]

    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    op = ignore_cast(inner.value)
    input_pointer = ignore_cast(op.a).buffer.data
    input_pointer1 = ignore_cast(op.b).buffer.data

    if reversed_operands:
        input_pointer, input_pointer1 = input_pointer1, input_pointer
    output_pointer = inner.buffer.data
    # Get feature map info
    serial_ifm, _ = get_ifm_params(input_pointer, producers_consumers, stmt)
    serial_ifm2, _ = get_ifm_params(input_pointer1, producers_consumers, stmt)
    serial_ofm, serial_block_config, replace_pointer, is_allocator = get_ofm_params(
        output_pointer, producers_consumers, stmt
    )
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
            rounding_mode=attrs["rounding_mode"],
            block_config=serial_block_config,
        ),
        output_pointer,
        replace_pointer,
        is_allocator,
    )
