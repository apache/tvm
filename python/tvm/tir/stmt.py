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
"""Statement AST Node in TVM.

Each statement node have subfields that can be visited from python side.

.. code-block:: python

    x = tvm.tir.Var("n", "int32")
    a = tvm.tir.Var("array", "handle")
    st = tvm.tir.stmt.Store(a, x + 1, 1)
    assert isinstance(st, tvm.tir.stmt.Store)
    assert(st.buffer_var == a)
"""
import tvm._ffi

from tvm.runtime import Object
from . import _ffi_api


class Stmt(Object):
    """Base class of all the statements."""


@tvm._ffi.register_object
class LetStmt(Stmt):
    """LetStmt node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : PrimExpr
        The value in to be binded.

    body : Stmt
        The body statement.
    """
    def __init__(self, var, value, body):
        self.__init_handle_by_constructor__(
            _ffi_api.LetStmt, var, value, body)


@tvm._ffi.register_object
class AssertStmt(Stmt):
    """AssertStmt node.

    Parameters
    ----------
    condition : PrimExpr
        The assert condition.

    message : PrimExpr
        The error message.

    body : Stmt
        The body statement.
    """
    def __init__(self, condition, message, body):
        self.__init_handle_by_constructor__(
            _ffi_api.AssertStmt, condition, message, body)


@tvm._ffi.register_object
class For(Stmt):
    """For node.

    Parameters
    ----------
    loop_var : Var
        The loop variable.

    min_val : PrimExpr
        The begining value.

    extent : PrimExpr
        The length of the loop.

    for_type : int
        The for type.

    device_api : int
        The device api type.

    body : Stmt
        The body statement.
    """
    Serial = 0
    Parallel = 1
    Vectorized = 2
    Unrolled = 3
    def __init__(self,
                 loop_var,
                 min_val,
                 extent,
                 for_type,
                 device_api,
                 body):
        self.__init_handle_by_constructor__(
            _ffi_api.For, loop_var, min_val, extent,
            for_type, device_api, body)


@tvm._ffi.register_object
class Store(Stmt):
    """Store node.

    Parameters
    ----------
    buffer_var : Var
        The buffer Variable.

    value : PrimExpr
        The value we want to store.

    index : PrimExpr
        The index in the store expression.

    predicate : PrimExpr
        The store predicate.
    """
    def __init__(self, buffer_var, value, index, predicate=None):
        args = [] if predicate is None else [predicate]
        self.__init_handle_by_constructor__(
            _ffi_api.Store, buffer_var, value, index, *args)


@tvm._ffi.register_object
class BufferStore(Stmt):
    """Buffer store node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    value : PrimExpr
        The value we to be stored.

    indices : List[PrimExpr]
        The indices location to be stored.
    """
    def __init__(self, buffer, value, indices):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferStore, buffer, value, indices)


@tvm._ffi.register_object
class BufferRealize(Stmt):
    """Buffer realize node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    bounds : List[Range]
        The value we to be stored.

    condition : PrimExpr
        The realize condition.

    body : Stmt
        The body of the statement.
    """
    def __init__(self, buffer, bounds, condition, body):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferRealize, buffer, bounds, condition, body)


@tvm._ffi.register_object
class Provide(Stmt):
    """Provide node.

    Parameters
    ----------
    func : Operation
        The operation to create the function.

    value_index : int
        The output value index

    value : PrimExpr
        The value to be stored.

    args : list of Expr
        The index arguments of the Provide.
    """
    def __init__(self, func, value_index, value, args):
        self.__init_handle_by_constructor__(
            _ffi_api.Provide, func, value_index, value, args)


@tvm._ffi.register_object
class Allocate(Stmt):
    """Allocate node.

    Parameters
    ----------
    buffer_var : Var
        The buffer variable.

    dtype : str
        The data type of the buffer.

    extents : list of Expr
        The extents of the allocate

    condition : PrimExpr
        The condition.

    body : Stmt
        The body statement.
    """
    def __init__(self,
                 buffer_var,
                 dtype,
                 extents,
                 condition,
                 body):
        self.__init_handle_by_constructor__(
            _ffi_api.Allocate, buffer_var, dtype,
            extents, condition, body)


@tvm._ffi.register_object
class AttrStmt(Stmt):
    """AttrStmt node.

    Parameters
    ----------
    node : Node
        The node to annotate the attribute

    attr_key : str
        Attribute type key.

    value : PrimExpr
        The value of the attribute

    body : Stmt
        The body statement.
    """
    def __init__(self, node, attr_key, value, body):
        self.__init_handle_by_constructor__(
            _ffi_api.AttrStmt, node, attr_key, value, body)


@tvm._ffi.register_object
class Free(Stmt):
    """Free node.

    Parameters
    ----------
    buffer_var : Var
        The buffer variable.
    """
    def __init__(self, buffer_var):
        self.__init_handle_by_constructor__(
            _ffi_api.Free, buffer_var)


@tvm._ffi.register_object
class Realize(Stmt):
    """Realize node.

    Parameters
    ----------
    func : Operation
        The operation to create the function.

    value_index : int
        The output value index

    dtype : str
        The data type of the operation.

    bounds : list of range
        The bound of realize

    condition : PrimExpr
        The realize condition.

    body : Stmt
        The realize body
    """
    def __init__(self,
                 func,
                 value_index,
                 dtype,
                 bounds,
                 condition,
                 body):
        self.__init_handle_by_constructor__(
            _ffi_api.Realize, func, value_index, dtype,
            bounds, condition, body)


@tvm._ffi.register_object
class SeqStmt(Stmt):
    """Sequence of statements.

    Parameters
    ----------
    seq : List[Stmt]
        The statements
    """
    def __init__(self, seq):
        self.__init_handle_by_constructor__(
            _ffi_api.SeqStmt, seq)

    def __getitem__(self, i):
        return self.seq[i]

    def __len__(self):
        return len(self.seq)


@tvm._ffi.register_object
class IfThenElse(Stmt):
    """IfThenElse node.

    Parameters
    ----------
    condition : PrimExpr
        The expression

    then_case : Stmt
        The statement to execute if condition is true.

    else_case : Stmt
        The statement to execute if condition is false.
    """
    def __init__(self, condition, then_case, else_case):
        self.__init_handle_by_constructor__(
            _ffi_api.IfThenElse, condition, then_case, else_case)


@tvm._ffi.register_object
class Evaluate(Stmt):
    """Evaluate node.

    Parameters
    ----------
    value : PrimExpr
        The expression to be evalued.
    """
    def __init__(self, value):
        self.__init_handle_by_constructor__(
            _ffi_api.Evaluate, value)


@tvm._ffi.register_object
class Prefetch(Stmt):
    """Prefetch node.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be prefetched.

    bounds : list of Range
        The bounds to be prefetched.
    """
    def __init__(self, buffer, bounds):
        self.__init_handle_by_constructor__(
            _ffi_api.Prefetch, buffer, bounds)


def stmt_seq(*args):
    """Make sequence of statements

    Parameters
    ----------
    args : list of Expr or Var
        List of statements to be combined as sequence.

    Returns
    -------
    stmt : Stmt
        The combined statement.
    """
    ret = []
    for value in args:
        if not isinstance(value, Stmt):
            value = Evaluate(value)
        ret.append(value)
    if len(ret) == 1:
        return ret[0]
    return SeqStmt(ret)


def stmt_list(stmt):
    """Make list of stmt from blocks.

    Parameters
    ----------
    stmt : A block statement

    Returns
    -------
    stmt_list : list of Stmt
         The unpacked list of statements
    """
    if isinstance(stmt, SeqStmt):
        res = []
        for x in stmt:
            res += stmt_list(x)
        return res
    return [stmt]
