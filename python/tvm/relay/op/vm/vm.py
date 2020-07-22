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
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks
"""Dialect operators for Relay VM."""
from . import _ffi_api


def shape_of(expr):
    """Invoke a function to get the shape of a tensor.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The expr used to evaluate its tensor shape.

    Returns
    -------
    result : tvm.relay.Expr
        The expression with the evaluated tensor shape.
    """
    return _ffi_api.shape_of(expr)


def invoke_tvm_op(func, inputs, outputs):
    """Call a primitive function with the TVM operator calling convention.

    Parameters
    ----------
    func : tvm.relay.Expr
        The input expr.

    inputs : tvm.relay.Expr
        A tuple of the inputs to pass to the TVM function.

    outputs : tvm.relay.Expr
        A tuple of the outputs to pass to the TVM function.

    Returns
    -------
    result : tvm.relay.Expr
        The invoke_tvm_op call node.
    """
    return _ffi_api.invoke_tvm_op(func, inputs, outputs)


def shape_func(func, inputs, outputs, is_inputs):
    """Invoke the shape function of the passed function.

    Parameters
    ----------
    func : tvm.relay.Expr
        The primitive function from which to compute the shape function.

    inputs : tvm.relay.Tuple
        The tupled inputs.

    outputs : tvm.relay.Tuple
        The tupled outputs.

    is_inputs : List[bool]
        A boolean list indicating whether the shape function should expect
        shape or input at each position.

    Returns
    -------
    result : tvm.relay.Expr
        The shape function expression.
    """
    return _ffi_api.shape_func(func, inputs, outputs, is_inputs)


def reshape_tensor(data, shape, newshape):
    """Invoke the VM ReshapeTensor instruction.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data.

    shape : tvm.relay.Expr
        The newshape tensor.

    newshape : List[tvm.ir.PrimExpr]
        The new shape.
    """
    return _ffi_api.reshape_tensor(data, shape, newshape)
