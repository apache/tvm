from . import _make
from . import ir

This module includes MyPy type signatures for all of the
exposed modules.
"""
from __future__ import absolute_import as _abs
from .._ffi.function import _init_api

# Base Constructors
Span = _make.Span

# Environment
Environment = _make.Environment

# Type Constructors
TensorType  = _make.TensorType
TypeParam = _make.TypeParam
FuncType    = _make.FuncType

# Types
def IntType(bits: int, lanes: int=1) -> ir.Type:
    """Constructs a integer base type.

       :param bits: The bit width of the integer type.
       :param lanes: The number of vector elements for this datatype.

    """
    return _make.IntType(bits, lanes)


def UIntType(bits: int, lanes: int=1) -> ir.Type:
    """Constructs a unsigned integer base type.

       :param bits: The bit width of the unsigned type.
       :param lanes: The number of vector elements for this datatype.
    """
    return _make.UIntType(bits, lanes)


def FloatType(bits: int, lanes: int=1) -> ir.Type:
    """Constructs a floating point base type.

       :param bits: The bit width of the unsigned type.
       :param lanes: The number of vector elements for this datatype.
    """
    return _make.FloatType(bits, lanes)


def BoolType(lanes: int =1) -> ir.Type:
    """Constructs a boolean base type.

       :param bits: The bit width of the unsigned type.
       :param lanes: The number of vector elements for this datatype.
    """
    return _make.BoolType(lanes)

# Expr Constructors
Constant = _make.Constant
Tuple = _make.Tuple
LocalVar = _make.LocalVar
GlobalVar = _make.GlobalVar
Param = _make.Param
Function = _make.Function
Call = _make.Call
Let = _make.Let
If = _make.If
IncompleteType = _make.IncompleteType

# Unifier
UnionFind = _make.UnionFind
TypeUnifier = _make.TypeUnifier

# Utility Functionality @TODO(jroesch): move to another location
_type_alpha_eq = _make._type_alpha_eq
