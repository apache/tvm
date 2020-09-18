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
"""Hybrid Script Parser Function Registry """
# pylint: disable=inconsistent-return-statements
import inspect
from enum import Enum
from typed_ast import ast3 as ast

import tvm


class Category(Enum):
    INTRIN = 0
    WITH_SCOPE = 1
    FOR_SCOPE = 2
    SPECIAL_STMT = 3


class Registry(object):
    """Registration map
    All these maps are static
    """

    functions = dict()

    @staticmethod
    def look_up_function(func_name):
        """look up a registered function by name"""
        if func_name in Registry.functions:
            return Registry.functions[func_name][0]
        return None

    @staticmethod
    def is_intrin(func):
        """check whether a function belongs to intrin"""
        return (func, Category.INTRIN) in Registry.functions.values()

    @staticmethod
    def is_with_scope(func):
        """check whether a function belongs to with scope handlers"""
        return (func, Category.WITH_SCOPE) in Registry.functions.values()

    @staticmethod
    def is_for_scope(func):
        """check whether a function belongs to for scope handlers"""
        return (func, Category.FOR_SCOPE) in Registry.functions.values()

    @staticmethod
    def is_special_stmt(func):
        """check whether a function belongs to special stmts"""
        return (func, Category.SPECIAL_STMT) in Registry.functions.values()

    @staticmethod
    def is_registered(func):
        """check whether a function is registered"""
        return (
            Registry.is_intrin(func)
            or Registry.is_with_scope(func)
            or Registry.is_for_scope(func)
            or Registry.is_special_stmt(func)
        )


class CallArgumentReader(object):
    """A helper class which read required argument from passed arguments"""

    def __init__(self, func_name, args, kwargs, parser):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.parser = parser

    def get_pos_only_arg(self, pos, name):
        """Get corresponding position only function argument from argument list"""
        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name not in self.kwargs:
            self.parser.report_error(self.func_name + " misses argument " + name)
        else:
            arg = self.kwargs[name]

        return arg

    def get_kwarg(self, pos, name, default):
        """Get corresponding keyword function argument from argument list
        If user doesn't provide the argument, set it to default value
        """
        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name in self.kwargs:
            arg = self.kwargs[name]
        else:
            return default

        return arg

    def get_varargs(self, pos):
        """Get corresponding variable argument from argument list"""
        if len(self.args) >= pos and len(self.kwargs) == 0:
            return self.args[pos - 1 :]
        return []

    def auto_insert_body(self, pos, body):
        """Automatically provide body as function call argument"""
        if len(self.args) >= pos:
            self.args.insert(pos - 1, body)
        else:
            self.kwargs["body"] = body


def func_wrapper(func_name, func_to_register, arg_list, category, concise=False, with_var=False):
    """Helper function to wrap a function to be registered """

    def wrap_func(parser, node, args, kwargs):
        if category == Category.FOR_SCOPE:
            # automatically parse loop vars and body for for_scope handlers
            loop_var_names = list()
            if isinstance(node.target, ast.Name):
                loop_var_names.append(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if not isinstance(elt, ast.Name):
                        parser.report_error("Invalid loop var")
                    loop_var_names.append(elt.id)
            else:
                parser.report_error("Invalid loop var")
            loop_vars = [tvm.te.var(name, dtype="int32") for name in loop_var_names]

            parser.scope_emitter.new_scope()
            parser.scope_emitter.node_stack[-1].extend(reversed(node.body))
            for loop_var in loop_vars:
                parser.scope_emitter.update_symbol(loop_var.name, loop_var)
            body = parser.get_body()
            parser.scope_emitter.pop_scope()
        elif category == Category.WITH_SCOPE:
            if not with_var:
                if isinstance(node, ast.With) and node.items[0].optional_vars is not None:
                    parser.report_error("Function " + func_name + " expects no optional vars")
                # automatically parse body for with_scope handlers without optional vars
                if isinstance(node, ast.With):
                    parser.scope_emitter.new_scope()
                    parser.scope_emitter.node_stack[-1].extend(reversed(node.body))
                    body = parser.get_body()
                    parser.scope_emitter.pop_scope()
                else:
                    body = parser.get_body()
            else:
                if isinstance(node, ast.With) and node.items[0].optional_vars is None:
                    parser.report_error("Function " + func_name + " expects optional vars")
                body = None

            if not isinstance(node, ast.With) and not concise:
                parser.report_error("Concise scoping is not allowed here")

        reader = CallArgumentReader(func_name, args, kwargs, parser)
        pos_only, kwargs, varargs = arg_list

        internal_args = list()
        if category == Category.WITH_SCOPE:
            if not with_var:
                internal_args.extend([parser, node, body])
            else:
                internal_args.extend([parser, node])
        elif category == Category.FOR_SCOPE:
            internal_args.extend([parser, node, body, loop_vars])
        elif category == Category.SPECIAL_STMT:
            internal_args.extend([parser, node])

        for i, arg_name in enumerate(pos_only):
            internal_args.append(reader.get_pos_only_arg(i + 1, arg_name))

        for i, arg_info in enumerate(kwargs):
            arg_name, default = arg_info
            internal_args.append(reader.get_kwarg(i + 1 + len(pos_only), arg_name, default=default))

        if varargs is not None:
            internal_args.extend(reader.get_varargs(len(pos_only) + len(kwargs) + 1))

        return func_to_register(*internal_args)

    return wrap_func


def get_arg_list(origin_func, category, with_var=False):
    """Helper function to get the argument list of Function
    Parameters
    ----------
    origin_func: function
        The function to get the argument list
    category: Category
        The category of registered function
    with_var: bool, optional
        Whether the with scope handler neeeds optional vars
    """
    full_arg_spec = inspect.getfullargspec(origin_func)

    args, defaults = full_arg_spec.args, full_arg_spec.defaults

    if defaults is None:
        defaults = tuple()

    if category == Category.WITH_SCOPE:
        if not with_var:
            if len(args) < 3 or args[0] != "parser" or args[1] != "node" or args[2] != "body":
                raise RuntimeError(
                    "TVM Hybrid Script register error : the first three arguments of "
                    "this with scope handler must be parser, node, body"
                )
            args = args[3:]
        else:
            if len(args) < 2 or args[0] != "parser" or args[1] != "node":
                raise RuntimeError(
                    "TVM Hybrid Script register error : the first two arguments of "
                    "this with scope handler must be parser, node"
                )
            args = args[2:]
    elif category == Category.FOR_SCOPE:
        if (
            len(args) < 4
            or args[0] != "parser"
            or args[1] != "node"
            or args[2] != "body"
            or args[3] != "loop_vars"
        ):
            raise RuntimeError(
                "TVM Hybrid Script register error : the first three arguments of for scope handler"
                "must be parser, node, body, loop_vars"
            )
        args = args[4:]
    elif category == Category.SPECIAL_STMT:
        if len(args) < 2 or args[0] != "parser" or args[1] != "node":
            raise RuntimeError(
                "TVM Hybrid Script register error : the first three arguments of special stmt"
                "must be parser, node"
            )
        args = args[2:]

    if full_arg_spec.varkw is not None:
        raise RuntimeError(
            "TVM Hybrid Script register error : variable keyword argument is not supported now"
        )
    if not len(full_arg_spec.kwonlyargs) == 0:
        raise RuntimeError(
            "TVM Hybrid Script register error : keyword only argument is not supported now"
        )

    pos_only = list()
    for arg in args[: len(args) - len(defaults)]:
        pos_only.append(arg)
    kwargs = list()
    for default, arg in zip(defaults, args[len(args) - len(defaults) :]):
        kwargs.append((arg, default))

    return pos_only, kwargs, full_arg_spec.varargs


def register_intrin(name=None):
    """Decorator to register function under category intrin
    Parameters
    ----------
    name: str, optional
        registered name for the function
    Example
    ------
    .. code-block:: python
    @register_intrin
    def broadcast(value, lanes):
        lanes = lanes.value if not isinstance(lanes, int) else lanes
        return tvm.tir.Broadcast(value, lanes)
    """

    def decorate(origin_func):
        func_name = "tir." + origin_func.__qualname__ if name is None else name
        Registry.functions[func_name] = (
            func_wrapper(
                func_name, origin_func, get_arg_list(origin_func, Category.INTRIN), Category.INTRIN
            ),
            Category.INTRIN,
        )
        return origin_func

    return decorate


def register_with_scope(concise=False, with_var=False, name=None):
    """Decorator to register function under with scope handler
    Parameters
    ----------
    concise: bool, optional
        whether this with scope handler is allowed in concise scoping
    with_var: bool, optional
        whether this with scope handler neeeds optional vars
    name: str, optional
        registered name for the function
    Example
    ------
    .. code-block:: python
    @register_scope_handler(concise=True)
    def attr(parser, node, attr_node, attr_key, value, body):
        return tvm.tir.AttrStmt(attr_node, attr_key, tvm.runtime.convert(value), body)
    """

    def decorate(origin_func):
        """Register function under category with_scope"""
        func_name = "tir." + origin_func.__qualname__ if name is None else name
        Registry.functions[func_name] = (
            func_wrapper(
                func_name,
                origin_func,
                get_arg_list(origin_func, Category.WITH_SCOPE, with_var),
                Category.WITH_SCOPE,
                concise=concise,
                with_var=with_var,
            ),
            Category.WITH_SCOPE,
        )
        return origin_func

    return decorate


def register_for_scope(name=None):
    """Decorator to register function under for scope handler
    Parameters
    ----------
    name: str, optional
        registered name for the function
    """

    def decorate(origin_func):
        func_name = "tir." + origin_func.__qualname__ if name is None else name
        Registry.functions[func_name] = (
            func_wrapper(
                func_name,
                origin_func,
                get_arg_list(origin_func, Category.FOR_SCOPE),
                Category.FOR_SCOPE,
            ),
            Category.FOR_SCOPE,
        )
        return origin_func

    return decorate


def register_special_stmt(name=None):
    """Decorator to register function under category special_stmt
    Parameters
    ----------
    name: str, optional
        registered name for the function
    Example
    -------
    @register_special_stmt
    def buffer_decl(parser, node, shape, dtype="float32", data=None, strides=[], elem_offset=None,
                    scope="global", align=-1, offset_factor=0, buffer_type="default"):
        align = align.value if not isinstance(align, int) else align
        offset_factor = offset_factor.value if not isinstance(offset_factor, int) else offset_factor
        buffer = tvm.tir.decl_buffer(shape, dtype, parser.assign_target, data, strides,
                                    elem_offset, scope, align, offset_factor, buffer_type)
        return buffer
    """

    def decorate(origin_func):
        func_name = "tir." + origin_func.__qualname__ if name is None else name
        Registry.functions[func_name] = (
            func_wrapper(
                func_name,
                origin_func,
                get_arg_list(origin_func, Category.SPECIAL_STMT),
                Category.SPECIAL_STMT,
            ),
            Category.SPECIAL_STMT,
        )
        return origin_func

    return decorate
