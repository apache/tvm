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
# pylint: disable=unused-argument, invalid-name, no-else-return, abstract-method, arguments-differ
"""Relax transformation passes for testing"""

import logging
import os
from typing import Dict, List, Set, Tuple
import tvm
from tvm import ir, relax
from tvm.ir import transform
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax import PyExprMutator
from tvm.relax.expr import Call, DataflowBlock, Var
from tvm.relay.backend.te_compiler import select_implementation
from tvm.runtime.object import Object
from tvm.target import Target


@ir.transform.module_pass(opt_level=0)
class LowerWithRelayOpStrategyPass(transform.Pass):
    """Lower Relax Op into TIR by using Relay OpStrategy.

    Since operators like conv2d, add, matmul are relay-, relax- independent,
    this pass assumes we can always find relay op equivalent for such relax ops,
    and use Relay Op Strategy (legacy) to perform lowering and find the TOPI implementation.

    Parameters
    ----------
    target : Target
        target info

    Returns
    -------
    pass : transform.Pass
        lowering pass
    """

    def __init__(self, target: Target):
        self.target = target

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        """Implement lowering mechanism.

        Parameters
        ----------
        mod : IRModule
            Input IRModule with Relax ops

        ctx: PassContext
            Pass context

        Returns
        -------
        out_mod : IRModule
            Output IRModule with lowered TIR functions
        """
        target = self.target

        @relax.expr_functor.mutator
        class Lowerer(PyExprMutator):
            """Mutator that performs lowering."""

            def visit_call_(self, call_node: Call):
                # Ignore function calls
                # We only target calls for operators
                if isinstance(call_node.op, (relax.GlobalVar, relax.expr.ExternFunc)):
                    return call_node

                # Current relax op name simply adds "relax." prefix to relay op name.
                # Thus, remove "relax." prefix to deduce relay op name.
                relay_op_name = call_node.op.name[6:]
                # Check if equivalent relay op exists. If not, return the original call.
                if relay_op_name in ir.Op.list_op_names():
                    relay_op = ir.Op.get(relay_op_name)

                    # Todo(relax-team): to be revisited - support dyn shape or deprecate.
                    tir_var_map = dict()
                    te_inputs = [relax.expr.te_tensor(arg, tir_var_map) for arg in call_node.args]
                    best_impl_tuple = select_implementation(
                        relay_op,
                        call_node.attrs,
                        te_inputs,
                        call_node.checked_type,
                        target,
                        use_autotvm=False,
                    )
                    compute_func = best_impl_tuple[0].compute
                    # Extract the name of the operator without the prefix
                    # e.g., for relay op "nn.conv2d", name_hint would be conv2d
                    name_hint = relay_op_name.split(".")[-1]

                    return self.builder_.call_te(
                        compute_func,
                        call_node.attrs,
                        call_node.args,
                        call_node.attrs,
                        primfunc_name_hint=name_hint,
                    )
                else:
                    return call_node

            # TOOD(@team): transform() wapper is necessary to include TIR functions.
            # IMO, this is bit unintuitive. Can we improve this?
            def transform(self):
                for gv, func in mod.functions_items():
                    if isinstance(func, relax.Function):
                        updated_func = self.visit_expr(func)
                        self.builder_.update_func(gv, updated_func)
                new_mod = self.builder_.get()
                new_mod = new_mod.with_attrs(mod.attrs) if mod.attrs else new_mod
                return new_mod

        return Lowerer().transform()


def ApplyEmptyCppMutator() -> tvm.ir.transform.Pass:
    packed_func = tvm.get_global_func("relax.testing.transform.ApplyEmptyCppMutator")
    return packed_func()


def dataflow_liveness_analysis(block: DataflowBlock) -> Dict[Var, Tuple[int, int]]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_liveness_analysis is exposed for testing only.")

    live_ranges = tvm.get_global_func("relax.testing.transform.DataflowLivenessAnalysis")(
        block
    )  # type: ignore
    ret = {}
    for var, live_range in live_ranges.items():
        ret[var] = tuple(live_range)
    return ret  # type: ignore


def dataflow_alias_analysis(
    block: DataflowBlock, inputs: List[Var]
) -> Tuple[Dict[Var, Set[int]], Dict[int, List[Set[int]]]]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_alias_analysis is exposed for testing only.")

    alias_sets, tuple_map = tvm.get_global_func("relax.testing.transform.DataflowAliasAnalysis")(
        block,
        inputs,
    )  # type: ignore
    res_alias_sets = {}
    res_tuple_map = {}
    for var, alias_set in alias_sets.items():
        res_alias_sets[var] = set(alias_set)
    for idx, elem_alias_sets in tuple_map.items():
        res_tuple_map[idx] = [set(alias_set) for alias_set in elem_alias_sets]
    return res_alias_sets, res_tuple_map  # type: ignore


@tvm._ffi.register_object("relax.transform.InplaceOpportunity")
class InplaceOpportunity(Object):
    """
    Represents an opportunity to make a binding in-place. Exposed only for testing;
    the constructor is not exposed.

    Parameters:
    -----------
    binding_idx: int
        Index of the binding within its block

    arg_idxs: List[int]
        Indices of arguments that are eligible to be used as in-place targets.
    """

    def __init__(self, _binding_idx, _arg_idxs):
        raise NotImplementedError("Constructor for InplaceOpportunity not exposed!")


def dataflow_inplace_analysis(
    block: DataflowBlock, inputs: List[Var], mod: IRModule
) -> Tuple[List[Tuple[int, Set[int]]], List[Tuple[int, Set[int]]]]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_inplace_analysis is exposed for testing only.")
    index_lists = tvm.get_global_func("relax.testing.transform.DataflowInplaceAnalysis")(
        block, inputs, mod
    )  # type: ignore

    def convert(opp_list):
        return list(map(lambda opp: (int(opp.binding_idx), set(map(int, opp.arg_idxs))), opp_list))

    return (convert(index_lists[0]), convert(index_lists[1]))  # type: ignore


def dataflow_single_inplace_call(
    mod: IRModule, call: Call, inplace_indices: List[int]
) -> Tuple[Call, IRModule]:
    """
    Inner function for the dataflow inplace transformation exposed for testing.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.warning("The function dataflow_single_inplace_call is exposed for testing only.")

    ret = tvm.get_global_func("relax.testing.transform.SingleInplaceCall")(
        mod,
        call,
        inplace_indices,
    )  # type: ignore
    return (ret[0], ret[1])  # type: ignore
