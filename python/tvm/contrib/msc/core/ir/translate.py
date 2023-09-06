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
"""tvm.contrib.msc.core.ir.translate"""

from typing import Dict, Optional, Tuple

import tvm
from tvm.relax.transform import BindParams
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.msc.core import transform as msc_transform
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from .graph import MSCGraph, MSCTensor


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
                    l.name in ref_layout for l in weight_layout
                ), "layout mismatch {} compare to {}".format(ref_t, weight_t)
                permute = [ref_layout.index(l) for l in weight_layout]
                return tvm.nd.array(data.asnumpy().transpose(*permute))
        return data

    weights = {t.name: _to_data(t, d) for t, d in t_weights.items()}
    return weights


def from_relax(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
) -> Tuple[MSCGraph, Dict[str, tvm.nd.array]]:
    """Change IRModule to MSCGraph.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    trans_config: dict
        The config for transfrorm IRModule.
    build_config: dict
        The config for build MSCGraph.

    Returns
    -------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    trans_config = trans_config or {}
    build_config = build_config or {}
    # TODO(tong.meng): optimize before translate?
    if params:
        mod = BindParams("main", params)(mod)
    patterns = get_patterns_with_prefix("msc")
    passes = [
        tvm.relax.transform.FuseOpsByPattern(
            patterns, bind_constants=False, annotate_codegen=False
        ),
        msc_transform.SetExprName(),
        msc_transform.SetExprLayout(trans_config.get("allow_layout_missing", True)),
    ]
    mod = tvm.transform.Sequential(passes)(mod)
    graph = _ffi_api.BuildFromRelax(mod, "main", msc_utils.dump_dict(build_config))
    t_weights = _ffi_api.GetRelaxWeights(mod, "main")
    return graph, normalize_weights(t_weights, graph)


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
        The config for transfrorm IRModule.
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

    trans_config = trans_config or {}
    build_config = build_config or {}
    opt_config = opt_config or {}
    # TODO(tong.meng): optimize before translate?
    opt_level = opt_config.get("opt_level", 0)
    if opt_level == 0:
        if params:
            mod["main"] = bind_params_by_name(mod["main"], params)
    else:
        target = opt_config.get("target", "llvm")
        disabled_pass = opt_config.get("disabled_pass", []) + [
            "SimplifyInference",
            "CanonicalizeOps",
            "FuseOps",
            "AlterOpLayout",
        ]
        with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass):
            mod, params = tvm.relay.optimize(mod, target=target, params=params)
    patterns = tvm.relay.op.contrib.get_pattern_table("msc")
    passes = [
        tvm.relay.transform.InferType(),
        tvm.relay.transform.MergeComposite(patterns),
        msc_transform.SetExprName(as_relax=False),
    ]
    mod = tvm.transform.Sequential(passes)(mod)
    graph = _ffi_api.BuildFromRelay(mod, "main", msc_utils.dump_dict(build_config))
    t_weights = _ffi_api.GetRelayWeights(mod, "main")
    return graph, normalize_weights(t_weights, graph)
