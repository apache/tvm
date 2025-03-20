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
"""Pattern matching in SLM"""


import inspect
from typing import Dict, Tuple, List

import tvm
from tvm.relax.frontend import nn

from tvm.relax import dpl as relax_pattern


def _relax_function_to_pattern(
    func: "tvm.relax.Function",
) -> Tuple[List[relax_pattern.WildcardPattern], relax_pattern.DFPattern]:
    """Convert a relax function into a pattern to be matched

    TODO(Lunderberg): Replace `tvm.relax.dpl` with function objects.
    Pattern-matching and replacement can be done using a function
    object as the pattern.
    """

    params: List[relax_pattern.WildcardPattern] = []
    patterns: Dict[tvm.relax.Var, relax_pattern.DFPattern] = {}

    for param in func.params:
        wildcard = relax_pattern.WildcardPattern().has_struct_info(param.struct_info)
        params.append(wildcard)
        patterns[param] = wildcard

    def _make_pattern(expr: tvm.relax.Expr) -> relax_pattern.DFPattern:
        if isinstance(expr, tvm.relax.Var):
            return patterns[expr]
        elif isinstance(expr, tvm.relax.Call):
            op = _make_pattern(expr.op)
            args = [_make_pattern(arg) for arg in expr.args]
            return op(*args)
        elif isinstance(expr, tvm.relax.Tuple):
            fields = [_make_pattern(field) for field in expr.fields]
            return relax_pattern.TuplePattern(fields)
        elif isinstance(expr, tvm.ir.Op):
            return relax_pattern.ExprPattern(expr)
        else:
            raise TypeError(
                f"Cannot convert relax expression {expr} of type {type(expr)}, "
                f"which has struct info {expr.struct_info_}, "
                f"into DFPattern."
            )

    seq_expr = func.body
    for block in seq_expr.blocks:
        for binding in block.bindings:
            patterns[binding.var] = _make_pattern(binding.value)

    top_pattern = _make_pattern(seq_expr.body)

    return params, top_pattern


def _relax_function_to_rewriter(
    param_patterns: List[relax_pattern.WildcardPattern],
    replacement_func: "tvm.relax.Function",
) -> Tuple[List[relax_pattern.WildcardPattern], relax_pattern.DFPattern]:
    """Generate a rewriter from a relax.Function"""

    assert len(replacement_func.params) == len(param_patterns)

    def rewriter(expr, matches):
        match_results = [matches[param_pattern] for param_pattern in param_patterns]
        func = tvm.relax.utils.copy_with_new_vars(replacement_func)

        input_bindings = [
            tvm.relax.VarBinding(param, match_result)
            for param, match_result in zip(func.params, match_results)
        ]
        output_expr = tvm.relax.SeqExpr([tvm.relax.DataflowBlock(input_bindings)], func.body)

        output_var = tvm.relax.Var("match_result", expr.struct_info)
        output_binding = tvm.relax.VarBinding(output_var, output_expr)

        return tvm.relax.SeqExpr([tvm.relax.DataflowBlock([output_binding])], output_var)

    return rewriter


def _relax_transform_by_rewrite_call(pattern, rewriter):
    @tvm.ir.transform.module_pass(name="relax.PatternReplacement", opt_level=0)
    def transform(mod, _context):
        updates = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, tvm.relax.Function):
                new_func = relax_pattern.rewrite_call(pattern, rewriter, func)
                if not func.same_as(new_func):
                    updates[gvar] = new_func

        if updates:
            mod = mod.clone()
            mod.update(updates)

        return mod

    return transform


def _no_op_init__(self):  # pylint: ignore=unused-argument
    pass


class ReplaceWithSubclass(nn.Mutator):
    """A SLM mutator to inject an optimized subclass

    Parameters
    ----------
    optimized_subclass: type

        A optimized subclass of a `nn.Module` subclass.
    """

    def __init__(self, optimized_subclass: type):
        base_class = optimized_subclass.__base__

        assert issubclass(
            optimized_subclass, nn.Module
        ), "The optimized implementation must inherit from a subclass of nn.Module"
        assert (
            base_class is not nn.Module
        ), "The optimized implementation must not be a direct subclass of nn.Module"

        self.base_class = base_class
        self.optimized_subclass = optimized_subclass

    def visit_module(self, name: str, node: nn.Module) -> nn.Module:
        """Replace a nn.Module subclass with an optimized version"""

        node = super().visit_module(name, node)
        if isinstance(node, self.base_class):
            # We want to replace the nn.Module without needing to
            # construct a new instance.
            node.__class__ = self.optimized_subclass

            cached_init = self.base_class.__init__
            self.base_class.__init__ = _no_op_init__
            try:
                node.__init__()
            finally:
                self.base_class.__init__ = cached_init

        return node

    def as_relax_transform(self) -> tvm.ir.transform.Pass:
        """Produce a Relax-to-Relax transform"""
        init_sig = inspect.signature(self.base_class.__init__)

        init_kwargs = {}
        for name, param in init_sig.parameters.items():
            if name == "self":
                pass
            elif issubclass(int, param.annotation):
                # The annotation is either `int` on its own, or a
                # Union that includes `int`.
                init_kwargs[name] = tvm.tir.Var(name, "int64")
            else:
                raise TypeError(
                    f"Cannot determine argument type for __init__ argument {name}, "
                    f"with type annotation {param.annotation}"
                )

        forward_sig = inspect.signature(self.base_class.forward)
        forward_spec = {}
        for name, param in forward_sig.parameters.items():
            if name == "self":
                pass
            elif param.annotation is nn.Tensor:
                forward_spec[name] = nn.spec.Tensor(None, "void")
            else:
                raise TypeError(
                    f"Cannot determine argument type for __init__ argument {name}, "
                    f"with type annotation {param.annotation}"
                )

        spec = {"forward": forward_spec}

        base_impl = self.base_class(**init_kwargs)
        optimized_impl = self.optimized_subclass(**init_kwargs)

        base_tvm, _ = base_impl.export_tvm(spec)
        optimized_tvm, _ = optimized_impl.export_tvm(spec)

        base_tvm = base_tvm["forward"]
        optimized_tvm = optimized_tvm["forward"]

        param_patterns, match_pattern = _relax_function_to_pattern(base_tvm)
        match_rewriter = _relax_function_to_rewriter(param_patterns, optimized_tvm)

        return _relax_transform_by_rewrite_call(match_pattern, match_rewriter)


def replace_implementation(optimized_subclass: type):
    """Produce a mutator to replace an existing nn.Module

    This utility allows users to write an optimized implementation of
    an existing `nn.Module`, and to substitute it into an existing
    end-to-end model.

    Parameters
    ----------
    optimized_subclass: type

        A optimized subclass of a `nn.Module` subclass.

    Returns
    -------
    mutator: nn.Mutator

        A mutator that replaces `optimized_subclass.__base__` with
        `optimized_subclass`.
    """
    return ReplaceWithSubclass(optimized_subclass)
