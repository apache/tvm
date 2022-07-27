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
"""Extract parameters from the transform operators in TIR."""
import tvm
from .spec import SerialCopy
from .utils import get_base_address, get_op_attrs


def get_copy_params(stmt, producers_consumers):
    """Get the parameters necessary to construct a call_extern for a copy.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a copy loop nest.
    producers_consumers: ProducersConsumers
        It associates pointers with the loop nest that produces
        their values and with the loop nest that consumes their values.

    Returns
    -------
    SerialCopy
        The parameters needed to construct a copy.
    tvm.tir.Var
        The output pointer of the copy operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the convolution output pointer.
    is_allocator : bool
        Whether this operator allocates its output.
    """
    _, body = get_op_attrs(stmt)
    length = body.extent
    write_store = body.body
    write_base = [get_base_address(index) for index in write_store.indices]
    read_load = body.body.value
    read_base = [get_base_address(index) for index in read_load.indices]
    return (
        SerialCopy(
            read_address=tvm.tir.expr.BufferLoad(read_load.buffer, read_base),
            length=length,
            write_address=tvm.tir.expr.BufferLoad(write_store.buffer, write_base),
        ),
        write_store.buffer.data,
        None,
        True,
    )
