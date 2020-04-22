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
"""Internal utilities for parsing Python subset to HalideIR"""

import ast
import inspect
import logging
import sys
import numpy

import tvm.runtime
from tvm._ffi.base import numeric_types
from tvm.ir.container import Array

from tvm.tir import expr as _expr
from tvm.tir import stmt as _stmt
from tvm.te.tensor import Tensor


#pylint: disable=invalid-name
np_arg_types = tuple(list(numeric_types) + [numpy.ndarray])
tvm_arg_types = (Tensor, Array, _expr.Var, _expr.ConstExpr)
halide_imm_types = (_expr.IntImm, _expr.FloatImm)


def _internal_assert(cond, err):
    """Simplify the code segment like if not XXX then raise an error"""
    if not cond:
        raise ValueError(err)


# Useful constants. In avoid of runtime dependences, we use function calls to return them.
def make_nop():
    """Returns a 'no operation' node in HalideIR."""
    return _stmt.Evaluate(tvm.runtime.const(0, dtype='int32'))


def is_docstring(node):
    """Checks if a Python AST node is a docstring"""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)


def _pruned_source(func):
    """Prune source code's extra leading spaces"""
    try:
        lines = inspect.getsource(func).split('\n')
        leading_space = len(lines[0]) - len(lines[0].lstrip(' '))
        lines = [line[leading_space:] for line in lines]
        return '\n'.join(lines)
    except IOError as err:
        if sys.version_info[0] == 2 and str(err) == 'could not get source code':
            logging.log(logging.CRITICAL, \
                        'This module is not fully operated under Python2... ' \
                        'Please move to Python3!')
            raise err


def replace_io(body, rmap):
    """Replacing tensors usage according to the dict given"""
    # pylint: disable=import-outside-toplevel
    from tvm.tir import stmt_functor

    def replace(op):
        if isinstance(op, _stmt.Provide) and op.func in rmap.keys():
            buf = rmap[op.func]
            return _stmt.Provide(buf.op, op.value_index, op.value, op.args)
        if isinstance(op, _expr.Call) and  op.func in rmap.keys():
            buf = rmap[op.func]
            return _expr.Call(buf.dtype, buf.name, op.args, \
                              _expr.Call.Halide, buf.op, buf.value_index)
        return None

    return stmt_functor.ir_transform(body, None, replace, ['Provide', 'Call'])


def _is_tvm_arg_types(args):
    """Determine a list of element is either a list of tvm arguments of a list of numpy arguments.
    If neither is true, raise a value error."""
    if isinstance(args[0], tvm_arg_types):
        for elem in args[1:]:
            _internal_assert(isinstance(elem, tvm_arg_types),
                             "Expecting a Var, Tensor or ConstExpr instance but %s get!" \
                             % str(type(elem)))
        return True

    _internal_assert(isinstance(args[0], np_arg_types), \
                     "Expect a numpy type but %s get!" % str(type(args[0])))
    for elem in args[1:]:
        _internal_assert(isinstance(elem, np_arg_types), \
                         "Expect a numpy type but %s get!" % str(type(elem)))
    return False
