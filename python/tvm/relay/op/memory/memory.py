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
"""Operators for manipulating low-level memory."""
from __future__ import absolute_import as _abs
from . import _make

def invoke_tvm_op(func, inputs, outputs):
    """Call a primitive function with the TVM operator calling convention.

    Parameters
    ----------
    inputs : tvm.relay.Expr
        A tuple of the inputs to pass to the TVM function.

    outputs : tvm.relay.Expr
        A tuple of the outputs to pass to the TVM function.

    Returns
    -------
    result : tvm.relay.Expr
        The invoke_tvm_op call node.
    """
    return _make.invoke_tvm_op(func, inputs, outputs)

def alloc_tensor(storage, shape, dtype='float32', assert_shape=None):
    """Allocate a tensor with the provided shape, and dtype.

    Parameters
    ----------
    storage : tvm.relay.Expr
        The storage to allocate from.

    shape : tvm.relay.Expr
        The shape of the tensor to allocate.

    dtype: str
        The dtype of the tensor.

    assert_shape: Control the static shape when computed by dynamic shape expression.

    Returns
    -------
    result : tvm.relay.Expr
        The alloc_tensor expression.
    """
    return _make.alloc_tensor(storage, shape, dtype, assert_shape)

def alloc_storage(size, alignment, dtype_hint='float32'):
    """Allocate a piece of tensor storage.

    Parameters
    ----------
    size : tvm.relay.Expr
        The size of the allocation.
    alignment : tvm.relay.Expr
        The alignment of the allocation.
    dtype : str
        The dtype_hint of the allocation.

    Returns
    -------
    result : tvm.relay.Expr
        The alloc_storage expression.
    """
    return _make.alloc_storage(size, alignment, dtype_hint)

def shape_func(func, inputs, outputs, dependent=False):
    """Invoke the shape function of the passed function.

    Parameters
    ----------
    func : tvm.relay.Expr
        The primitive function from which to compute the shape function.
    inputs : tvm.relay.Tuple
        The tupled inputs.
    outputs : tvm.relay.Tuple
        The tupled outputs.

    Returns
    -------
    result : tvm.relay.Expr
        The shape function expression.
    """
    return _make.shape_func(func, inputs, outputs, dependent)
