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


def get_copy_params(stmt, producers, consumers):
    """Get the parameters necessary to construct a call_extern for a copy.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a copy loop nest.
    producers : dict of tvm.tir.Var to tvm.tir.AttrStmt
        A dictionary to associate pointers with the loop nest
        that produces their values.
    consumers : dict of tvm.tir.Var to tvm.tir.AttrStmt
        A dictionary to associate pointers with the loop nest
        that consumes their values.

    Returns
    -------
    SerialCopy
        The parameters needed to construct a copy.
    tvm.tir.Var
        The output pointer of the copy operation.

    """
    _, body = get_op_attrs(stmt)
    length = body.extent
    write_store = body.body
    write_base = get_base_address(write_store.index)
    read_load = body.body.value
    read_base = get_base_address(read_load.index)
    dtype = body.body.value.dtype
    return (
        SerialCopy(
            read_address=tvm.tir.expr.Load(dtype, read_load.buffer_var, read_base),
            length=length,
            write_address=tvm.tir.expr.Load(dtype, write_store.buffer_var, write_base),
        ),
        write_store.buffer_var,
        None,
    )
