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
# pylint: disable=invalid-name, unused-argument, missing-function-docstring, abstract-method
"""Relax LazyTransformParams pass."""
import tvm
from tvm import IRModule
from tvm import relax
from tvm.relax.expr_functor import visitor, mutator, PyExprMutator, PyExprVisitor


@visitor
class ForwardCollector(PyExprVisitor):
    """
    Perform a forward pass to collect the following information:
    out_tuple_map: map from var to its index in the output tuple
    var_tuple_get_item: list of var that is bound to v = params[i]

    Parameters
    ----------
    tuple_var: relax.Var
        The output tuple var

    input_params: relax.Var
        The input tuple var

    """

    def __init__(self, tuple_var: relax.Var, input_params: relax.Var) -> None:
        self.out_tuple_map = {}
        self.out_tuple_var = tuple_var
        self.input_params = input_params
        self.var_tuple_get_item = []
        self.is_tuple_get_item_input = False

    def visit_tuple_getitem_(self, op: relax.TupleGetItem) -> None:
        if op.tuple_value == self.input_params:
            self.is_tuple_get_item_input = True
        else:
            self.is_tuple_get_item_input = False
        super().visit_tuple_getitem_(op)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if binding.var == self.out_tuple_var:
            assert isinstance(binding.value, relax.Tuple)
            for i, expr in enumerate(binding.value.fields):
                self.out_tuple_map[expr] = relax.PrimValue(i)
        else:
            self.is_tuple_get_item_input = False
            super().visit_var_binding_(binding)
            if self.is_tuple_get_item_input:
                self.var_tuple_get_item.append(binding.var)


@visitor
class LivenessAnalysis(PyExprVisitor):
    """
    Perform a backward pass to collect the following information:
    var_liveness_end: map from var to the list of var whose liveness is killed by this var binding

    Parameters
    ----------
    out_tuple_var: relax.Var
        The output tuple var
    input_params: set
        The set of vars that are bound to v = params[i]
    """

    def __init__(self, out_tuple_var: relax.Var) -> None:
        self.last_appear_in_var_binding = None
        self.out_tuple_var = out_tuple_var
        self.var_liveness_end = {}
        self.ended_vars = set()

    def visit_binding_block_(self, block: relax.BindingBlock) -> None:
        for binding in reversed(block.bindings):
            self.visit_binding(binding)

    def visit_var_(self, op: relax.Var) -> None:
        if op not in self.ended_vars:
            self.last_appear_in_var_binding.append(op)
            self.ended_vars.add(op)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if self.out_tuple_var == binding.var:
            return
        self.last_appear_in_var_binding = []
        super().visit_var_binding_(binding)
        # param[i] is in output
        if binding.var not in self.ended_vars:
            self.last_appear_in_var_binding.append(binding.var)
            self.ended_vars.add(binding.var)
        self.var_liveness_end[binding.var] = self.last_appear_in_var_binding


@mutator
class LazyTransformParamsMutator(PyExprMutator):
    """
    Transform transform_params functions into a lazy version.

    Parameters
    ----------
    mod: IRModule
        The module to be transformed
    """

    def __init__(self, mod: IRModule = None) -> None:
        super().__init__(mod)
        self.mod = mod
        # the only input param, which should be a Tuple
        self.input_tuple_param = None
        self.input_params_set = None
        self.out_tuple_map = None
        self.out_tuple_var = None
        self.memory_free_insertion = None
        self.killed_vars = set()

    def transform(self, func: relax.Function) -> relax.Function:
        self.input_tuple_param = func.params[0]
        seq_expr = func.body
        self.out_tuple_var = seq_expr.body
        # Step 1. collect out_tuple_map and input_params_set
        forward_collector = ForwardCollector(self.out_tuple_var, self.input_tuple_param)
        forward_collector.visit_expr(func)
        self.out_tuple_map = forward_collector.out_tuple_map
        # input_params_set is the set of binding var for var = params[i]
        self.input_params_set = set(forward_collector.var_tuple_get_item)
        # Step 2. liveness analysis and get where to insert kill_object instruction
        liveness = LivenessAnalysis(self.out_tuple_var)
        liveness.visit_expr(func)
        self.memory_free_insertion = liveness.var_liveness_end
        # Step 3. rewrite get item and set item
        new_body = self.visit_expr(func.body)
        return relax.Function([], new_body, relax.ObjectStructInfo(), attrs=func.attrs)

    def visit_tuple_getitem_(self, op: relax.TupleGetItem) -> relax.Expr:
        # rewrite get item
        tuple_get_item = super().visit_tuple_getitem_(op)
        if tuple_get_item.tuple_value == self.input_tuple_param:
            return relax.Call(
                relax.ExternFunc("get_item"),
                [relax.PrimValue(tuple_get_item.index)],
                None,
                [relax.ObjectStructInfo()],
            )
        else:
            return tuple_get_item

    def visit_var_(self, var: relax.Var) -> None:
        assert var not in self.killed_vars
        return super().visit_var_(var)

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if binding.var == self.out_tuple_var:
            # The function after rewriting returns a empty tuple.
            func_output = self.builder_.emit(relax.Tuple([]))
            self.set_var_remap(binding.var.vid, func_output)
            return

        super().visit_var_binding_(binding)

        if binding.var in self.memory_free_insertion:
            for var in self.memory_free_insertion[binding.var]:
                if var in self.out_tuple_map:
                    self.killed_vars.add(var)
                    index = self.out_tuple_map[var]
                    # rewrite set item
                    self.builder_.emit(
                        relax.Call(
                            relax.ExternFunc("set_item"),
                            [index, super().visit_var_(var)],
                            None,
                            [relax.ObjectStructInfo()],
                        ),
                        name_hint="_",
                    )

                if var in self.input_params_set:
                    self.builder_.emit(
                        relax.op.vm.kill_object(super().visit_var_(var)), name_hint="_"
                    )


@tvm.transform.module_pass(opt_level=0, name="LazyTransformParams")
class LazyTransformParams:
    """
    Convert transform_params functions into a lazy version.
    (Load the input to memory on demand, and immediately free it after the last use.)

    Note: ToNonDataflow() and RemovePurityTracking() should be invoked before this pass.
    """

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        lazy_mutator = LazyTransformParamsMutator(mod)
        for gv in mod.functions:
            if gv.name_hint.endswith("transform_params"):
                func = mod[gv]
                if not isinstance(func, relax.Function):
                    continue
                func = lazy_mutator.transform(func)
                lazy_mutator.builder_.update_func(gv, func)

        return lazy_mutator.builder_.get()
