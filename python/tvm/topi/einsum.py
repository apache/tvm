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
# pylint: disable=invalid-name,consider-using-enumerate,redefined-outer-name
"""Einsum operator"""
from . import cpp


def einsum(subscripts, *operand):
    """Evaluates the Einstein summation convention on the operands.

    Parameters
    ----------
    subscripts : string
        Specifies the subscripts for summation as comma separated list of subscript labels.
        An implicit (classical Einstein summation) calculation is performed unless the
        explicit indicator ‘->’ is included as well as subscript labels of the precise
        output form.

    a_tuple : tuple of tvm.te.Tensor
        These are the Tensors for the operation.
        The only difference of einsum between in tvm and numpy is it needs an extra brackets
        for the tensors. For example, topi.einsum("ij, jk -> ik", (A, B)).

    Returns
    -------
    out : tvm.te.Tensor
        The calculation based on the Einstein summation convention.
    """

    return cpp.einsum(subscripts, operand)
