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
# pylint: disable=redefined-builtin
"""The basic Relax operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg


## (TVM-TOOL) py_op begin basic/*
def call_builtin_with_ctx(
    func: ty.ExternFunc,
    args: ty.TupleExpr,
    sinfo_args: ty.StructInfo,
) -> Call:
    """Call a VM builtin PackedFunc in destination-passing style (DPS). The difference between
        `call_builtin_with_ctx` and `call_dps_packed` is that `call_builtin_with_ctx` takes
        an extra argument `ctx` at the beginning of the arguments, which is the context of the
        current VM.

    Parameters
    ----------
    func : ty.ExternFunc
        The function being called.
    args : ty.TupleExpr
        The arguments to the packed func. Always a Relax Tuple expression.
    sinfo_args : ty.StructInfo
        The StructInfo of the arguments.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The created call node.
    """
    func = tg.check(0, "func", tg.ExternFunc(), func)
    args = tg.check(1, "args", tg.TupleExpr(), args)
    sinfo_args = tg.check(2, "sinfo_args", tg.StructInfo(), sinfo_args)
    _ffi_func = _ffi.get_global_func("relax.op.call_builtin_with_ctx")
    return _ffi_func(func, args, sinfo_args)


def call_dps_packed(
    func: ty.ExternFunc,
    args: ty.TupleExpr,
    out_sinfo: ty.StructInfo,
) -> Call:
    """Call a PackedFunc in destination-passing style (DPS).

    Parameters
    ----------
    func : ty.ExternFunc
        The function being called.
    args : ty.TupleExpr
        The arguments to the packed func. Always a Relax Tuple expression whose length indicates
        the number `n + m` in the example.
    out_sinfo : ty.StructInfo
        The StructInfo of the output.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The created call node.
    """
    func = tg.check(0, "func", tg.ExternFunc(), func)
    args = tg.check(1, "args", tg.TupleExpr(), args)
    out_sinfo = tg.check(2, "out_sinfo", tg.StructInfo(), out_sinfo)
    _ffi_func = _ffi.get_global_func("relax.op.call_dps_packed")
    return _ffi_func(func, args, out_sinfo)


def call_tir(
    gvar: ty.GlobalVar,
    args: ty.TupleExpr,
    out_sinfo: ty.StructInfo,
    tir_vars: ty.Optional[ty.Shape] = None,
) -> Call:
    """Call a PrimFunc in TensorIR, and return its output using a special calling convention
        called destination-passing style (DPS) in TVM.

    Parameters
    ----------
    gvar : ty.GlobalVar
        The global variable that points to the function being called.
    args : ty.TupleExpr
        The arguments to the function. Always a Relax Tuple expression whose length indicates
        the number `n` in the example the number of arguments.
    out_sinfo : ty.StructInfo
        The StructInfo of the output. It is used to infer the number of outputs, and indicates
        the number `m` in the example.
    tir_vars : ty.Optional[ty.Shape]
        The TIR variables to be used with the call. They are usually used for symbolic shapes.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The created call node.
    """
    gvar = tg.check(0, "gvar", tg.GlobalVar(), gvar)
    args = tg.check(1, "args", tg.TupleExpr(), args)
    out_sinfo = tg.check(2, "out_sinfo", tg.StructInfo(), out_sinfo)
    tir_vars = tg.check(3, "tir_vars", tg.Optional(tg.Shape()), tir_vars)
    _ffi_func = _ffi.get_global_func("relax.op.call_tir")
    return _ffi_func(gvar, args, out_sinfo, tir_vars)


def invoke_closure(
    closure: ty.AnyRelaxExpr,
    args: ty.TupleExpr,
    sinfo_args: ty.StructInfo,
) -> Call:
    """Invoke a closure.

    Parameters
    ----------
    closure : ty.AnyRelaxExpr
        The closure being invoked.
    args : ty.TupleExpr
        The arguments to the closure. Always a Relax Tuple expression.
    sinfo_args : ty.StructInfo
        The StructInfo of the output

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The created call node.
    """
    closure = tg.check(0, "closure", tg.AnyRelaxExpr(), closure)
    args = tg.check(1, "args", tg.TupleExpr(), args)
    sinfo_args = tg.check(2, "sinfo_args", tg.StructInfo(), sinfo_args)
    _ffi_func = _ffi.get_global_func("relax.op.invoke_closure")
    return _ffi_func(closure, args, sinfo_args)


def make_closure(
    func: ty.GlobalVar,
    args: ty.TupleExpr,
) -> Call:
    """Create a closure with free variables and return the closure.

    Parameters
    ----------
    func : ty.GlobalVar
        The function being called.
    args : ty.TupleExpr
        The arguments to the packed func. Always a Relax Tuple expression.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The created call node.
    """
    func = tg.check(0, "func", tg.GlobalVar(), func)
    args = tg.check(1, "args", tg.TupleExpr(), args)
    _ffi_func = _ffi.get_global_func("relax.op.make_closure")
    return _ffi_func(func, args)


def null_value() -> Call:
    """Create a call node that represents a null value object.

    Returns
    -------
    ret : ty.AnyRelaxExpr
        The created call node.
    """

    _ffi_func = _ffi.get_global_func("relax.op.null_value")
    return _ffi_func()


def shape_of(
    expr: ty.Tensor,
) -> Call:
    """Get shape of a tensor. It gets TensorStructInfo and returns ShapeStructInfo

    Parameters
    ----------
    expr : ty.Tensor
        The input expression of TensorStructInfo.

    Returns
    -------
    ret : ty.Shape
        The created call node.
    """
    expr = tg.check(0, "expr", tg.Tensor([]), expr)
    _ffi_func = _ffi.get_global_func("relax.op.shape_of")
    return _ffi_func(expr)


## (TVM-TOOL) py_op end basic/*
