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
"""TVM Script Parser Scope Handler Classes"""
# pylint: disable=redefined-builtin, unused-argument, invalid-name, relative-beyond-top-level

from synr import ast
import tvm.tir
from .utils import get_param_list, from_synr_span
from .registry import register


class ScopeHandler:
    """Base class for all scope handlers"""

    def __init__(self, func):
        self.func = func
        self.body = None
        self.node = None
        self.context = None

    def signature(self):
        return "tir." + self.func.__name__, get_param_list(self.func)

    def enter_scope(self, node, context, arg_list, span):
        pass

    def exit_scope(self, node, context, arg_list, span):
        self.node = node
        self.context = context
        return self.func(*arg_list, span=from_synr_span(span))


class WithScopeHandler(ScopeHandler):
    """Base class for all with scope handlers"""

    def __init__(self, func, concise_scope, def_symbol):
        super().__init__(func)
        self.concise_scope = concise_scope
        self.def_symbol = def_symbol

    @staticmethod
    def get_optional_var_names(node, context):
        """Get list of names from ast.With's optional_vars"""
        assert isinstance(node, ast.With)

        var_names = None
        if isinstance(node.items[0].optional_vars, ast.Name):
            var_names = [node.items[0].optional_vars.id]
        elif isinstance(node.items[0].optional_vars, (ast.List, ast.Tuple)):
            for var in node.items[0].optional_vars.elts:
                if not isinstance(var, ast.Name):
                    context.report_error("Invalid optional var definition")
            var_names = [var.id for var in node.items[0].optional_vars.elts]
        else:
            context.report_error("Invalid optional var definition")
        return var_names


@register
class Allocate(WithScopeHandler):
    """ With scope handler tir.alloc_with_scope(var, extents, dtype, scope, condition) """

    def __init__(self):
        def allocate(extents, dtype, scope, condition=True, span=None):
            condition = tvm.runtime.convert(condition)
            scope = tvm.runtime.convert(scope)
            body = tvm.tir.Allocate(
                self.buffer_var, dtype, extents, condition, self.body, span=span
            )
            return tvm.tir.AttrStmt(self.buffer_var, "storage_scope", scope, body, span=span)

        super().__init__(allocate, concise_scope=True, def_symbol=True)
        self.buffer_var = None

    def enter_scope(self, node, context, arg_list, span):
        # define buffer vars in symbol table
        if isinstance(node, ast.With):
            names = WithScopeHandler.get_optional_var_names(node, context)
            if len(names) != 1:
                context.report_error("Unexpected number of vars", node.span)
            name = names[0]
        elif isinstance(node, ast.Assign):
            name = node.lhs.id.name
        else:
            raise Exception("Internal Bug")

        def setup_buffer_var(extents, dtype, scope, condition=True, span=None):
            """Setup buffer var for a given type."""
            buffer_ptr_type = tvm.ir.PointerType(tvm.ir.PrimType(dtype))
            self.buffer_var = tvm.tir.Var(name, buffer_ptr_type, span)

        setup_buffer_var(*arg_list, span=from_synr_span(node.lhs.id.span))
        context.update_symbol(name, self.buffer_var)


@register
class LaunchThread(WithScopeHandler):
    """ With scope handler tir.launch_thread(env_var, extent) """

    def __init__(self):
        def launch_thread(env_var, extent, span):
            extent = tvm.runtime.convert(extent, span=span)
            return tvm.tir.AttrStmt(
                tvm.tir.IterVar(
                    None,
                    env_var,
                    getattr(tvm.tir.IterVar, "ThreadIndex"),
                    self.context.func_var_env_dict[env_var],
                    span=span,
                ),
                "thread_extent",
                extent,
                self.body,
                span=span,
            )

        super().__init__(launch_thread, concise_scope=True, def_symbol=False)


@register
class Realize(WithScopeHandler):
    """ With scope handler tir.realize(buffer_bounds, scope, condition) """

    def __init__(self):
        def realize(buffer_bounds, scope, condition=True, span=None):
            buffer, bounds = buffer_bounds
            scope = tvm.runtime.convert(scope, span=span)
            return tvm.tir.AttrStmt(
                buffer,
                "realize_scope",
                scope,
                tvm.tir.BufferRealize(buffer, bounds, condition, self.body, span=span),
                span=span,
            )

        super().__init__(realize, concise_scope=True, def_symbol=False)


@register
class Attr(WithScopeHandler):
    """ With scope handler tir.attr(attr_node, attr_key, value) """

    def __init__(self):
        def attr(attr_node, attr_key, value, span):
            attr_node = tvm.runtime.convert(attr_node, span=span)
            value = tvm.runtime.convert(value, span=span)
            return tvm.tir.AttrStmt(attr_node, attr_key, value, self.body, span=span)

        super().__init__(attr, concise_scope=True, def_symbol=False)


@register
class AssertHandler(WithScopeHandler):
    """ With scope handler tir.Assert(condition, message) """

    def __init__(self):
        def Assert(condition, message, span):
            return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), self.body, span=span)

        super().__init__(Assert, concise_scope=True, def_symbol=False)


@register
class Let(WithScopeHandler):
    """ With scope handler tir.let(var, value) """

    def __init__(self):
        def let(var, value, span):
            return tvm.tir.LetStmt(var, value, self.body, span=span)

        super().__init__(let, concise_scope=False, def_symbol=False)


class ForScopeHandler(ScopeHandler):
    """Base class for all for scope handlers"""

    def __init__(self, func):
        super().__init__(func)
        self.loop_vars = None

    def enter_scope(self, node, context, arg_list, span):
        assert isinstance(node, ast.For)

        loop_var_names = list()
        spans = list()
        if isinstance(node.lhs, ast.Var):
            loop_var_names.append(node.lhs.id.name)
            spans.append(from_synr_span(node.lhs.id.span))
        elif isinstance(node.lhs, ast.Tuple):
            for elt in node.lhs.values:
                if not isinstance(elt, ast.Var):
                    context.report_error("Invalid loop var", elt.span)
                loop_var_names.append(elt.id.name)
                spans.append(from_synr_span(elt.id.span))
        else:
            context.report_error("Invalid loop var", node.lhs.span)

        self.loop_vars = [
            tvm.te.var(name, dtype="int32", span=span) for name, span in zip(loop_var_names, spans)
        ]
        for loop_var in self.loop_vars:
            context.update_symbol(loop_var.name, loop_var)


@register
class Serial(ForScopeHandler):
    """ For scope handler tir.serial(begin, end)"""

    def __init__(self):
        def serial(begin, end, span):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var", span)
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 0, 0, self.body, span=span)

        super().__init__(serial)


@register
class Parallel(ForScopeHandler):
    """ For scope handler tir.parallel(begin, end)"""

    def __init__(self):
        def parallel(begin, end, span):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 1, 0, self.body, span=span)

        super().__init__(parallel)


@register
class Vectorized(ForScopeHandler):
    """ For scope handler tir.vectorized(begin, end)"""

    def __init__(self):
        def vectorized(begin, end, span):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 2, 0, self.body, span=span)

        super().__init__(vectorized)


@register
class Unroll(ForScopeHandler):
    """ For scope handler tir.unroll(begin, end)"""

    def __init__(self):
        def unroll(begin, end, span):
            if len(self.loop_vars) != 1:
                self.context.report_error("Expect exact 1 loop var")
            ana = tvm.arith.Analyzer()
            extent = end if begin == 0 else ana.simplify(end - begin)
            return tvm.tir.For(self.loop_vars[0], begin, extent, 3, 0, self.body, span=span)

        super().__init__(unroll)
