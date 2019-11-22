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
"""External function interface to cuBLAS libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin

def matmul(lhs, rhs_data, rhs_indices, rhs_indptr, transb=False):
    """Create an extern op that compute matrix multiplication of
       lhs and rhs with cuSPARSE

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs_data : Tensor
        The CSR format data array of rhs
    rhs_indices : Tensor
        The CSR format index array of rhs
    rhs_indptr : Tensor
        The CSR format index pointer array of rhs
    transb : bool
        Whether to transpose rhs

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    # transb needs to be False, i.e., rhs matrix cannot be transposed
    # due to lack of shape info in current interface"
    assert transb is False

    n = lhs.shape[0]
    m = rhs_indptr.shape[0]-1

    return _api.extern(
        (n, m), [lhs, rhs_data, rhs_indices, rhs_indptr],\
            lambda ins, outs: _intrin.call_packed("tvm.contrib.cusparse.matmul",\
            ins[0], ins[1], ins[2], ins[3], outs[0], transb),\
            name="C", dtype=lhs.dtype)
