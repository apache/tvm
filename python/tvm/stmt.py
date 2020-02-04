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

User do not need to deal with AST node directly.
But they can be helpful for developer to do quick proptyping.
While not displayed in the document and python file.
Each statement node have subfields that can be visited from python side.

.. code-block:: python

    x = tvm.var("n")
    a = tvm.var("array", tvm.handle)
    st = tvm.make.Store(a, x + 1, 1)
    assert isinstance(st, tvm.stmt.Store)
    assert(st.buffer_var == a)
"""
import tvm._ffi
from ._ffi.object import Object
from . import make as _make


class Stmt(Object):
    pass

@tvm._ffi.register_object
class LetStmt(Stmt):
    """LetStmt node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : Expr
        The value in to be binded.

    body : Stmt
        The body statement.
    """
    def __init__(self, var, value, body):
        self.__init_handle_by_constructor__(
            _make.LetStmt, var, value, body)


@tvm._ffi.register_object
class AssertStmt(Stmt):
    """AssertStmt node.

    Parameters
    ----------
    condition : Expr
        The assert condition.

    message : Expr
        The error message.

    body : Stmt
        The body statement.
    """
    def __init__(self, condition, message, body):
        self.__init_handle_by_constructor__(
            _make.AssertStmt, condition, message, body)


@tvm._ffi.register_object
class ProducerConsumer(Stmt):
    """ProducerConsumer node.

    Parameters
    ----------
    func : Operation
        The Operation.

    is_producer : bool
        Whether if the node is producer.

    body : Stmt
        The body statement.
    """
    def __init__(self, func, is_producer, body):
        self.__init_handle_by_constructor__(
            _make.ProducerConsumer, func, is_producer, body)


@tvm._ffi.register_object
class For(Stmt):
    """For node.

    Parameters
    ----------
    loop_var : Var
        The loop variable.

    min_val : Expr
        The begining value.

    extent : Expr
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
            _make.For, loop_var, min_val, extent,
            for_type, device_api, body)


@tvm._ffi.register_object
class Store(Stmt):
    """Store node.

    Parameters
    ----------
    buffer_var : Var
        The buffer Variable.

    value : Expr
        The value we want to store.

    index : Expr
        The index in the store expression.

    predicate : Expr
        The store predicate.
    """
    def __init__(self, buffer_var, value, index, predicate):
        self.__init_handle_by_constructor__(
            _make.Store, buffer_var, value, index, predicate)


@tvm._ffi.register_object
class Provide(Stmt):
    """Provide node.

    Parameters
    ----------
    func : Operation
        The operation to create the function.

    value_index : int
        The output value index

    value : Expr
        The value to be stored.

    args : list of Expr
        The index arguments of the Provide.
    """
    def __init__(self, func, value_index, value, args):
        self.__init_handle_by_constructor__(
            _make.Provide, func, value_index, value, args)


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

    condition : Expr
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
            _make.Allocate, buffer_var, dtype,
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

    value : Expr
        The value of the attribute

    body : Stmt
        The body statement.
    """
    def __init__(self, node, attr_key, value, body):
        self.__init_handle_by_constructor__(
            _make.AttrStmt, node, attr_key, value, body)


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
            _make.Free, buffer_var)


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

    condition : Expr
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
            _make.Realize, func, value_index, dtype,
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
            _make.SeqStmt, seq)

    def __getitem__(self, i):
        return self.seq[i]

    def __len__(self):
        return len(self.seq)


@tvm._ffi.register_object
class IfThenElse(Stmt):
    """IfThenElse node.

    Parameters
    ----------
    condition : Expr
        The expression

    then_case : Stmt
        The statement to execute if condition is true.

    else_case : Stmt
        The statement to execute if condition is false.
    """
    def __init__(self, condition, then_case, else_case):
        self.__init_handle_by_constructor__(
            _make.IfThenElse, condition, then_case, else_case)


@tvm._ffi.register_object
class Evaluate(Stmt):
    """Evaluate node.

    Parameters
    ----------
    value : Expr
        The expression to be evalued.
    """
    def __init__(self, value):
        self.__init_handle_by_constructor__(
            _make.Evaluate, value)


@tvm._ffi.register_object
class Prefetch(Stmt):
    """Prefetch node.

    Parameters
    ----------
    func : Operation
        The operation to create the function.

    value_index : int
        The output value index

    dtype : str
        The data type to be prefetched.

    bounds : list of Range
        The bounds to be prefetched.
    """
    def __init__(self, func, value_index, dtype, bounds):
        self.__init_handle_by_constructor__(
            _make.Prefetch, func, value_index, dtype, bounds)


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
    if isinstance(stmt, ProducerConsumer):
        return stmt_list(stmt.body)
    return [stmt]


_make.stmt_list = stmt_list
