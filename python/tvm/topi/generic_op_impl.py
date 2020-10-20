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
"""Implementation of generic operators in the presence of Tensor"""
# pylint: disable=invalid-name, too-many-arguments
import tvm
from tvm import te
from . import broadcast as _broadcast
from . import math as _math


def _make_bop(broadcast_bop, orig_bop):
    """Make a specific overloaded binary operator of Tensor when applicable;
    apply the original operator if it is not supposed to be overloaded.

    Consider the following scenario:
    OP :   + | - | * | /
    R0 :   int | float | Expr | TensorSlice | Tensor (rank zero)
    R1 :   Tensor (positive rank)

    In terms of (LHS OP RHS), we apply the following overloading rules:
    (1) We use broadcast_OP(LHS, RHS), when both LHS and RHS are R1.
    (2) We perform element-wise operation of Tensor and scalar,
        when one of LHS and RHS is R1 and another is R0.
    (3) We do not overload OP (i.e. stick to orig_bop) otherwise.

    Parameters
    ----------
    broadcast_bop : operator function
        Operator for broadcast tensor-tensor operation, for rule (1).

    orig_bop: operator function
        Operator before overloading, for rule (3).

    Returns
    -------
    ret : operator function
        The overloaded operator function if applicable or orig_bop otherwise.
    """

    name = orig_bop.__name__

    def _tensor_bop_impl(lhs, rhs):
        """Overloaded {op} operator.

        If both operands are non-zero-rank Tensors, it performs
        tensor-tensor {op} operation, and broadcasts inputs when necessary.

        If one operand is non-zero-rank Tensor, while the other operand is
        scalar like type (e.g., numeric types, Expr, or TensorSlice),
        it performs tensor-scalar {op} operation on an element-wise basis.

        Otherwise, it performs default generic.{op} operation, as defined
        in tvm.generic module.

        Parameters
        ----------
        lhs : object
            Left operand.
        rhs : object
            Right operand.

        Returns
        -------
        ret : tvm.te.Tensor (if at least one operand is non-zero-rank Tensor)
              tvm.Expr (otherwise)
            The result of {op} operation.
        """
        if not isinstance(lhs, te.tensor.Tensor) and not isinstance(rhs, te.tensor.Tensor):
            return orig_bop(lhs, rhs)
        return broadcast_bop(lhs, rhs)

    _tensor_bop_impl.__doc__ = _tensor_bop_impl.__doc__.format(op=name)
    return _tensor_bop_impl


def _bind_generic_ops():
    """Bind generic operators for Tensor."""
    # Check __op_priority__ to make sure the binding happens only once.
    __op_priority__ = 1
    if __op_priority__ > tvm.tir.generic.__op_priority__:
        tvm.tir.generic.__op_priority__ = __op_priority__
        tvm.tir.generic.add = _make_bop(_broadcast.add, tvm.tir.generic.add)
        tvm.tir.generic.subtract = _make_bop(_broadcast.subtract, tvm.tir.generic.subtract)
        tvm.tir.generic.multiply = _make_bop(_broadcast.multiply, tvm.tir.generic.multiply)
        tvm.tir.generic.divide = _make_bop(_broadcast.divide, tvm.tir.generic.divide)
        tvm.tir.generic.cast = _math.cast


_bind_generic_ops()
