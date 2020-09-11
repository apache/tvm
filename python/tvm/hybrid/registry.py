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
from typed_ast import ast3 as ast


class Registry(object):
    """Registration map
    All these maps are static
    """

    intrin = dict()
    with_scope = dict()
    for_scope = dict()
    special_stmt = dict()


class CallArgumentReader(object):
    """A helper class which read required argument from passed arguments"""

    def __init__(self, func_name, args, kwargs, parser):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.parser = parser

    def get_func_compulsory_arg(self, pos, name):
        """Get corresponding function argument from argument list which is compulsory"""

        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name not in self.kwargs.keys():
            self.parser.report_error(self.func_name + " misses argument " + name)
        else:
            arg = self.kwargs[name]

        return arg

    def get_func_optional_arg(self, pos, name, default):
        """Get corresponding function argument from argument list which is optional.
        If user doesn't provide the argument, set it to default value
        """

        if len(self.args) >= pos:
            arg = self.args[pos - 1]
        elif name in self.kwargs.keys():
            arg = self.kwargs[name]
        else:
            return default

        return arg


def func_wrapper(func_name, func_to_register, arg_list, need_parser_and_node, need_body, concise):
    """Helper function to wrap a function to be registered """

    def wrap_func(parser, node, args, kwargs):
        reader = CallArgumentReader(func_name, args, kwargs, parser)
        internal_args = list()

        if need_body and not isinstance(node, ast.For):
            # automatically parse body for with scope handlers
            if isinstance(node, ast.With):
                # the with scope handler is used inside with context
                parser.scope_emitter.new_scope()
                parser.scope_emitter.node_stack[-1].extend(reversed(node.body))
                body = parser.get_body()
                parser.scope_emitter.pop_scope()
            else:
                # the with scope handler is used in concise scoping
                if not concise:
                    parser.report_error("Concise scoping is not allowed here")
                body = parser.get_body()

        if need_parser_and_node:
            internal_args.append(parser)
            internal_args.append(node)

        for i, arg_info in enumerate(arg_list):
            if len(arg_info) == 1:
                (arg_name,) = arg_info
                if need_body and arg_name == "body":
                    internal_args.append(body)
                else:
                    internal_args.append(reader.get_func_compulsory_arg(i + 1, arg_name))
            else:
                arg_name, default = arg_info
                internal_args.append(reader.get_func_optional_arg(i + 1, arg_name, default=default))

        return func_to_register(*internal_args)

    return wrap_func


def get_arg_list(origin_func, need_parser_and_node):
    """Helper function to get the argument list of Function

    Parameters
    ----------
    origin_func: function
        The function to get the argument list

    need_parser_and_node: bool
        Whether the function need parser and node in its arguments
    """

    full_arg_spec = inspect.getfullargspec(origin_func)

    args, defaults = full_arg_spec.args, full_arg_spec.defaults

    if defaults is None:
        defaults = tuple()
    if need_parser_and_node:
        args = args[2:]

    if full_arg_spec.varargs is not None:
        raise RuntimeError(
            "TVM Hybrid Script register error : variable argument is not supported now"
        )
    if full_arg_spec.varkw is not None:
        raise RuntimeError(
            "TVM Hybrid Script register error : variable keyword argument is not supported now"
        )
    if not len(full_arg_spec.kwonlyargs) == 0:
        raise RuntimeError(
            "TVM Hybrid Script register error : keyword only argument is not supported now"
        )

    arg_list = list()
    for arg in args[: len(args) - len(defaults)]:
        arg_list.append((arg,))
    for default, arg in zip(defaults, args[len(args) - len(defaults) :]):
        arg_list.append((arg, default))

    return arg_list


def register_intrin(origin_func):
    """Decorator to register function under category intrin

    Example
    ------
    .. code-block:: python

    @register_intrin
    def broadcast(value, lanes):
        lanes = lanes.value if not isinstance(lanes, int) else lanes
        return tvm.tir.Broadcast(value, lanes)
    """
    func_name = origin_func.__qualname__
    Registry.intrin[func_name] = func_wrapper(
        func_name,
        origin_func,
        get_arg_list(origin_func, False),
        need_parser_and_node=False,
        need_body=False,
        concise=False,
    )
    return origin_func


def register_with_scope(concise=False):
    """Decorator to register function under with scope handler

    Parameters
    ----------
    concise: bool
        whether this scope handler is allowed in concise scoping

    Example
    ------
    .. code-block:: python

    @register_scope_handler(concise=True)
    def attr(parser, node, attr_node, attr_key, value, body):

        return tvm.tir.AttrStmt(attr_node, attr_key, tvm.runtime.convert(value), body)
    """

    def decorate(origin_func):
        """Register function under category with_scope"""
        func_name = origin_func.__qualname__
        Registry.with_scope[func_name] = func_wrapper(
            func_name,
            origin_func,
            get_arg_list(origin_func, True),
            need_parser_and_node=True,
            need_body=True,
            concise=concise,
        )
        return origin_func

    return decorate


def register_for_scope():
    """Decorator to register function under for scope handler"""

    def decorate(origin_func):
        """Register function under category for_scope"""
        func_name = origin_func.__qualname__
        Registry.for_scope[func_name] = func_wrapper(
            func_name,
            origin_func,
            get_arg_list(origin_func, True),
            need_parser_and_node=True,
            need_body=True,
            concise=False,
        )
        return origin_func

    return decorate


def register_special_stmt(origin_func):
    """Decorator to register function under category special_stmt

    Example
    -------
    @register_special_stmt
    def buffer_decl(parser, node, shape, dtype="float32", data=None, strides=[], elem_offset=None,
                    scope="global", align=-1, offset_factor=0, buffer_type="default"):
        align = align.value if not isinstance(align, int) else align
        offset_factor = offset_factor.value if not isinstance(offset_factor, int) else offset_factor
        buffer = tvm.tir.decl_buffer(shape, dtype, parser._assign_target, data, strides,
                                    elem_offset, scope, align, offset_factor, buffer_type)
        return buffer

    """

    func_name = origin_func.__qualname__
    Registry.special_stmt[func_name] = func_wrapper(
        func_name,
        origin_func,
        get_arg_list(origin_func, True),
        need_parser_and_node=True,
        need_body=False,
        concise=False,
    )
    return origin_func
