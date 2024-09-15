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
"""tvm.contrib.msc.core.frontend.translate"""

from typing import Dict, Optional, Tuple, List

import tvm
from tvm.relax.transform import BindParams
from tvm.relax import PyExprVisitor
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import dataflow_pattern as relay_pattern
from tvm.contrib.msc.core import transform as msc_transform
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.ir import MSCGraph, MSCTensor


def normalize_inputs(inputs: List[tuple]) -> List[tuple]:
    """Normalize the inputs info

    Parameters
    ----------
    inputs: list of <name, shape, dtype>
        The inputs info.

    Returns
    -------
    inputs: list of <name, shape, dtype>
        The normalized inputs info.
    """

    recorded_vars = {}

    def _normalize_input(inp):
        def _normalize(info):
            if not isinstance(info, (tuple, list)):
                return info
            dims = []
            for dim in info:
                if isinstance(dim, int):
                    dims.append(dim)
                elif dim in recorded_vars:
                    dims.append(recorded_vars[dim])
                elif isinstance(dim, str):
                    recorded_vars[dim] = tvm.tir.Var(dim, "int64")
                    dims.append(recorded_vars[dim])
                else:
                    raise TypeError("Unexpected dim {} in shape {}".format(dim, info))
            return dims

        return [_normalize(i) for i in inp]

    return [_normalize_input(inp) for inp in inputs]


def normalize_weights(
    t_weights: Dict[MSCTensor, tvm.nd.array], graph: MSCGraph
) -> Dict[str, tvm.nd.array]:
    """Normalize the weghts.

    Parameters
    ----------
    t_weights: dict of <MSCTensor, tvm.nd.array>
        The weights extracted from IRModule.
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.

    Returns
    -------
    weights: dict of <string:tvm.ndarray>
        The normalized weights.
    """

    def _to_data(ref_t, data):
        weight_t = graph.find_tensor(ref_t.name)
        if weight_t.ndim == 1:
            if ref_t.ndim != weight_t.ndim:
                return tvm.nd.array(data.asnumpy().reshape(weight_t.get_shape()))
            return data
        if ref_t.layout and weight_t.layout:
            ref_layout, weight_layout = ref_t.layout.name, weight_t.layout.name
            if ref_layout != weight_layout:
                assert all(
                    l in ref_layout for l in weight_layout
                ), "layout mismatch {} compare to {}".format(ref_t, weight_t)
                permute = [ref_layout.index(l) for l in weight_layout]
                return tvm.nd.array(data.asnumpy().transpose(*permute))
        return data

    weights = {t.name: _to_data(t, d) for t, d in t_weights.items() if graph.has_tensor(t.name)}
    # sort the weights by graph weights
    graph_weights = {}
    for weight in graph.get_weights():
        assert weight.name in weights, "Missing weight " + str(weight)
        graph_weights[weight.name] = weights[weight.name]
    return graph_weights


def from_relax(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    opt_config: Optional[Dict[str, str]] = None,
) -> Tuple[MSCGraph, Dict[str, tvm.nd.array]]:
    """Change IRModule to MSCGraph.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    trans_config: dict
        The config for transform IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize the relax before translate.

    Returns
    -------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    trans_config = msc_utils.copy_dict(trans_config)
    build_config = msc_utils.copy_dict(build_config)
    opt_config = msc_utils.copy_dict(opt_config)
    entry = trans_config.get("entry", "main")
    if params:
        mod = BindParams("main", params)(mod)
    opt_level = opt_config.get("opt_level", 1)
    if opt_level > 0:
        mod = tvm.transform.Sequential(
            [
                tvm.relax.transform.FoldConstant(),
            ]
        )(mod)
    patterns = get_patterns_with_prefix("msc.")
    passes = [
        tvm.relax.transform.ExpandTupleArguments(),
        msc_transform.SetExprName(),
        msc_transform.SetExprLayout(trans_config.get("allow_layout_missing", True)),
        tvm.relax.transform.FuseOpsByPattern(
            patterns, bind_constants=False, annotate_codegen=False
        ),
    ]
    mod = tvm.transform.Sequential(passes)(mod)
    graph = _ffi_api.BuildFromRelax(mod, entry, msc_utils.dump_dict(build_config))
    t_weights = _ffi_api.GetRelaxWeights(mod, entry)
    return graph, normalize_weights(t_weights, graph)


def get_relay_patterns(
    mod: tvm.IRModule,
    entry_name: str = "main",
) -> List[Tuple[str, relay_pattern.DFPattern, callable]]:
    """Filter relay patterns based on mod.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relay.
    entry_name: str
        The entry name.

    Returns
    -------
    patterns: list
        The useful patterns for relay
    """

    class OpExtractor(ExprVisitor):
        """Extract ops from expr."""

        def extract(self, expr):
            self._optypes = set()
            super().visit(expr)
            return self._optypes

        def visit_call(self, expr):
            super().visit_call(expr)
            if isinstance(expr.op, tvm.ir.Op):
                self._optypes.add(expr.op.name)

    op_names = OpExtractor().extract(mod[entry_name])
    skip_tags, patterns = set(), list(tvm.relay.op.contrib.get_pattern_table("msc"))
    if "nn.conv1d" not in op_names or "add" not in op_names:
        skip_tags.add("msc.conv1d_bias")
    if "nn.conv2d" not in op_names or "add" not in op_names:
        skip_tags.add("msc.conv2d_bias")
    if "nn.batch_matmul" not in op_names or "add" not in op_names:
        skip_tags.add("msc.linear_bias")
    if "nn.batch_matmul" not in op_names:
        skip_tags |= set(p[0] for p in patterns if p[0].startswith("msc.linear"))
        if "nn.dense" not in op_names:
            skip_tags |= set(p[0] for p in patterns if p[0].startswith("msc.matmul"))
    if "take" not in op_names:
        skip_tags |= set(p[0] for p in patterns if p[0].startswith("msc.embedding"))
    if "erf" not in op_names:
        skip_tags |= set(p[0] for p in patterns if p[0].startswith("msc.gelu"))
    valid_patterns = [p for p in patterns if p[0] not in skip_tags]
    return valid_patterns


def from_relay(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    opt_config: Optional[Dict[str, str]] = None,
) -> Tuple[MSCGraph, Dict[str, tvm.nd.array]]:
    """Change IRModule to MSCGraph.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relay.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    trans_config: dict
        The config for transform IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize the relay before translate.

    Returns
    -------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    trans_config = msc_utils.copy_dict(trans_config)
    build_config = msc_utils.copy_dict(build_config)
    opt_config = msc_utils.copy_dict(opt_config)
    # TODO(tong.meng): optimize before translate?
    opt_level = opt_config.get("opt_level", 0)
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    if opt_level > 0:
        target = opt_config.get("target", "llvm")
        disabled_pass = opt_config.get("disabled_pass", []) + [
            "SimplifyInference",
            "CanonicalizeOps",
            "FuseOps",
            "AlterOpLayout",
        ]
        with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass):
            mod, params = tvm.relay.optimize(mod, target=target, params=params)
    patterns = get_relay_patterns(mod)
    passes = [
        tvm.relay.transform.InferType(),
        tvm.relay.transform.MergeComposite(patterns),
        msc_transform.SetExprName(as_relax=False),
    ]
    mod = tvm.transform.Sequential(passes)(mod)
    graph = _ffi_api.BuildFromRelay(mod, "main", msc_utils.dump_dict(build_config))
    t_weights = _ffi_api.GetRelayWeights(mod, "main")
    return graph, normalize_weights(t_weights, graph)


@tvm.relax.expr_functor.visitor
class BYOCChecker(PyExprVisitor):
    """Checker to check if any non-target ops exist"""

    def check(self, func_names, expr):
        self._func_names = func_names
        self._non_target_exprs = []
        if isinstance(expr, tvm.relax.Expr):
            self.visit_expr(expr)
        elif isinstance(expr, tvm.relax.BindingBlock):
            self.visit_binding_block(expr)
        assert len(self._non_target_exprs) == 0, "Some exprs not on target {}".format(expr)

    def visit_var_binding_(self, binding) -> None:
        super().visit_var_binding_(binding)
        if isinstance(binding.value, tvm.relax.Call):
            if isinstance(binding.value.op, tvm.relax.GlobalVar):
                if binding.value.op.name_hint not in self._func_names:
                    self._non_target_exprs.append(binding.value)
            else:
                self._non_target_exprs.append(binding.value)
        elif not isinstance(binding.value, tvm.relax.DataflowVar):
            self._non_target_exprs.append(binding.value)


def byoc_partition(
    target: str,
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
) -> Tuple[tvm.IRModule, List[Tuple[MSCGraph, Dict[str, tvm.nd.array]]]]:
    """Partition module to target sub functions.

    Parameters
    ----------
    target: str
        The target for the BYOC.
    mod: IRModule
        The IRModule of relax.
    trans_config: dict
        The config for transform IRModule.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    build_config: dict
        The config for build MSCGraph.

    Returns
    -------
    mod: IRModule
        The IRModule of partitioned relax.
    graphs_info: list<<MSCGraph, weights>>
        The func <MSCGraph and weights> list, each element for a sub graph.
    """

    trans_config = msc_utils.copy_dict(trans_config)
    build_config = msc_utils.copy_dict(build_config)
    build_config["target"] = target
    for key in ["input_aliases", "output_aliases"]:
        if key in build_config:
            build_config.pop(key)
    entry = trans_config.get("entry", "main")
    if params:
        mod = BindParams("main", params)(mod)

    def _partition_mod(mod, as_msc=True):
        patterns = get_patterns_with_prefix(target)
        passes = [
            tvm.relax.transform.ExpandTupleArguments(),
            msc_transform.SetExprName(),
            msc_transform.SetExprLayout(trans_config.get("allow_layout_missing", True)),
            tvm.relax.transform.FuseOpsByPattern(patterns, bind_constants=not as_msc),
            msc_transform.InlineParams(),
            msc_transform.FuseTuple(target),
            tvm.relax.transform.MergeCompositeFunctions(),
            msc_transform.SetBYOCAttrs(target),
        ]
        return tvm.transform.Sequential(passes)(mod)

    def _is_target_func(func):
        if "Codegen" not in func.attrs:
            return False
        return func.attrs["Codegen"] == target

    msc_mod = _partition_mod(mod)
    func_names = [var.name_hint for var, func in msc_mod.functions.items() if _is_target_func(func)]

    if trans_config.get("as_complete", True):
        assert len(func_names) == 1, "More than 1 target func is found: " + str(msc_mod)
        BYOCChecker().check(func_names, msc_mod[entry])

    ref_weights = _ffi_api.GetRelaxWeights(msc_mod, entry)
    graphs, weights = [], {}
    for name in func_names:
        graph_name = msc_mod[name].attrs[_ffi_api.ToAttrKey("unique")]
        build_config.update({"graph_name": graph_name, "byoc_entry": name})
        graph = _ffi_api.BuildFromRelax(msc_mod, entry, msc_utils.dump_dict(build_config))
        graphs.append(graph)
        weights.update(normalize_weights(ref_weights, graph))
    return _partition_mod(mod, False), graphs, weights
