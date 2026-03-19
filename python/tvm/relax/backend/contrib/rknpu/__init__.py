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

"""RK3588 NPU (RKNPU) backend for TVM Relax.

Current project direction (see ``WORKING_DIRECTION.md``):
- Primary implementation path is Relax/TIR chain lowering + runtime bridge.
- Primary reverse-engineering reference is ``rknpu-compiler``.
- BYOC path is retained as legacy context/comparison, not the main delivery target.

Importing this module registers:
- Pattern table entries under the "rknpu" prefix
- ``relax.ext.rknpu`` Python codegen (overrides C++ stub)

Usage (BYOC path)::

    import tvm.relax.backend.contrib.rknpu as rknpu

    # For models with batch norm:
    mod = rknpu.fold_batch_norm(mod)
    mod = rknpu.partition_for_rknpu(mod)
    mod = relax.transform.RunCodegen()(mod)

Usage (experimental TIR path)::

    import tvm.relax.backend.contrib.rknpu as rknpu
    mod = rknpu.lower_to_rknpu_tir(mod)
    mod = rknpu.plan_rknpu_tir_memory(mod)
"""

import base64
import json
import os

import tvm
from tvm import relax
from tvm.ir import IRModule
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions

from ...pattern_registry import get_patterns_with_prefix

# Register patterns (side-effect import)
from . import patterns  # noqa: F401

# Register Python codegen as relax.ext.rknpu (side-effect import)
from . import codegen  # noqa: F401

from .transforms import DecomposeLayerNormForRKNPU, DecomposeSoftmaxForRKNPU
from .schedule_pretty import format_rknpu_schedule_report
from .tir_path import (
    build_rknpu_schedule_report,
    annotate_pc_chain_candidates,
    legalize_to_rknpu_tir_stages,
    lower_pc_chain_submits,
    lower_to_rknpu_tir,
    lower_to_rknpu_tir_with_pc_chain,
    plan_rknpu_tir_memory,
)


def fold_batch_norm(mod: IRModule) -> IRModule:
    """Fold batch_norm into preceding conv2d as conv2d + bias.

    Transforms ``conv2d → batch_norm → TupleGetItem(0)`` patterns into
    ``conv2d → add`` by absorbing the BN scale/shift into the conv2d
    weights and creating an explicit bias add.

    Prerequisites: conv2d weights and BN parameters (gamma, beta, mean,
    var) must be bound as constants.  Use ``relax.transform.BindParams``
    first if they are function parameters.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The module to transform.

    Returns
    -------
    mod : tvm.ir.IRModule
        Module with batch_norm folded into conv2d + add.
    """
    mod = relax.transform.FoldBatchnormToConv2D()(mod)
    mod = relax.transform.FoldConstant()(mod)
    return mod


def decompose_for_rknpu(mod: IRModule) -> IRModule:
    """Decompose high-level ops into NPU-compatible primitives.

    Currently decomposes:
    - ``nn.layer_norm`` into matmul / add / multiply / rsqrt
    - ``nn.softmax`` into max / matmul / add / exp / matmul / divide / matmul / multiply

    Must run BEFORE ``partition_for_rknpu``.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The module to transform.

    Returns
    -------
    mod : tvm.ir.IRModule
        Module with decomposed operations.
    """
    mod = DecomposeLayerNormForRKNPU()(mod)
    mod = DecomposeSoftmaxForRKNPU()(mod)
    return mod


def partition_for_rknpu(mod: IRModule, decompose=True, graph_level=False) -> IRModule:
    """Partition the graph, offloading supported operators to the RKNPU backend.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The module to run passes on.
    decompose : bool
        If True (default), decompose LayerNorm and other high-level ops
        into NPU-compatible primitives before pattern matching.
        Ignored when ``graph_level=True``.
    graph_level : bool
        If True, use high-level patterns (softmax, layer_norm) instead of
        decomposition, enabling graph-level compilation where all ops are
        merged into one function and intermediates stay in NPU layout.
        Default False for backward compatibility.

    Returns
    -------
    mod : tvm.ir.IRModule
        Annotated and partitioned module.
    """
    if graph_level:
        # Graph-level: use high-level patterns so softmax/layer_norm stay
        # as single composites.  annotate_codegen=True creates per-op
        # codegen wrappers; MergeCompositeFunctions is a no-op here
        # (it checks kComposite, not kCodegen).  For multi-op V8
        # merging, use partition_for_rknpu_v8() which requires the module
        # to contain at least one non-RKNPU op.
        rknpu_patterns = get_patterns_with_prefix("rknpu")
        mod = FuseOpsByPattern(rknpu_patterns, bind_constants=False, annotate_codegen=True)(mod)
        mod = MergeCompositeFunctions()(mod)
    else:
        # Per-op: decompose first, then pattern-match primitives
        if decompose:
            mod = decompose_for_rknpu(mod)
        # Filter out graph-level patterns to avoid partial matches
        all_patterns = get_patterns_with_prefix("rknpu")
        graph_only = {"rknpu.softmax", "rknpu.layer_norm"}
        rknpu_patterns = [p for p in all_patterns if p.name not in graph_only]
        mod = FuseOpsByPattern(rknpu_patterns, bind_constants=False, annotate_codegen=True)(mod)
        mod = MergeCompositeFunctions()(mod)
    return mod


def partition_for_rknpu_v8(mod: IRModule) -> IRModule:
    """Partition for V8 multi-op graph compilation.

    Uses ``annotate_codegen=False`` so that ``MergeCompositeFunctions``
    merges consecutive RKNPU composites into a single function compiled
    as one V8 binary with persistent NPU layout.

    **Requirement**: the module must contain at least one op that is NOT
    matched by RKNPU patterns (e.g. a residual add in float32).  If all
    ops match, ``MergeCompositeFunctions`` produces a degenerate main
    function that fails during ``relax.build()``.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The module to partition.

    Returns
    -------
    mod : tvm.ir.IRModule
        Module with RKNPU composites merged into V8 functions.
    """
    rknpu_patterns = get_patterns_with_prefix("rknpu")
    mod = FuseOpsByPattern(rknpu_patterns, bind_constants=False, annotate_codegen=False)(mod)
    mod = MergeCompositeFunctions()(mod)
    return mod


def get_bridge_chain_blob(mod: IRModule) -> bytes:
    """Return runtime-bridge chain blob from module attrs if present.

    The TIR chained path attaches `rknpu.bridge_chain_blob_b64` metadata.
    This helper decodes it to raw bytes for `runtime.rknpu_bridge_set_chain_blob`.
    """
    attrs = mod.attrs
    if attrs is None or "rknpu.bridge_chain_blob_b64" not in attrs:
        return b""
    blob_b64 = str(attrs["rknpu.bridge_chain_blob_b64"])
    return base64.b64decode(blob_b64.encode("ascii"))


def configure_runtime_bridge_from_mod(mod: IRModule) -> bool:
    """Preload runtime bridge chain blob from module metadata.

    Returns True when the runtime function exists and metadata was applied.
    """
    blob = get_bridge_chain_blob(mod)
    if not blob:
        return False
    set_chain = tvm.get_global_func("runtime.rknpu_bridge_set_chain_blob", allow_missing=True)
    if set_chain is None:
        # Backward-compatibility with older runtime symbol.
        set_chain = tvm.get_global_func(
            "runtime.rknpu_bridge_set_synthetic_chain", allow_missing=True
        )
    if set_chain is None:
        return False
    set_chain(blob)
    return True


def _attach_runtime_bridge_metadata_module(ex, mod: IRModule) -> bool:
    """Attach serialized RKNPU bridge metadata to an executable when available."""
    blob = get_bridge_chain_blob(mod)
    attrs = mod.attrs
    schedule_report_json = ""
    if attrs is not None and "rknpu.schedule_report_json" in attrs:
        schedule_report_json = str(attrs["rknpu.schedule_report_json"])
    if not blob and not schedule_report_json:
        return False
    create = tvm.get_global_func(
        "runtime.rknpu_bridge_metadata_module_create", allow_missing=True
    )
    if create is None or not hasattr(ex, "mod") or not hasattr(ex.mod, "import_module"):
        return False
    ex.mod.import_module(create(blob, schedule_report_json))
    return True


def _apply_runtime_bridge_metadata_from_executable(ex) -> bool:
    """Apply embedded RKNPU bridge metadata from a built executable when present."""
    mod = getattr(ex, "mod", None)
    if mod is None:
        return False
    get_function = getattr(mod, "get_function", None)
    if callable(get_function):
        try:
            fn = get_function("rknpu_bridge_apply_chain_blob", True)
        except TypeError:
            fn = get_function("rknpu_bridge_apply_chain_blob")
        if fn is not None:
            fn()
            return True
    return False


def get_rknpu_schedule_report(mod: IRModule) -> dict:
    """Return compile-time schedule report from module attrs if present."""
    attrs = mod.attrs
    if attrs is None or "rknpu.schedule_report_json" not in attrs:
        return {}
    return json.loads(str(attrs["rknpu.schedule_report_json"]))


def get_runtime_bridge_stats() -> dict:
    """Return runtime bridge counters as a JSON-decoded dict when available."""
    fn = tvm.get_global_func("runtime.rknpu_bridge_get_stats_json", allow_missing=True)
    if fn is None:
        return {}
    return json.loads(str(fn()))


def reset_runtime_bridge_stats() -> bool:
    """Reset runtime bridge counters for the current thread."""
    fn = tvm.get_global_func("runtime.rknpu_bridge_reset_stats", allow_missing=True)
    if fn is None:
        return False
    fn()
    return True


def build_with_runtime_bridge(mod: IRModule, *args, **kwargs):
    """Build a Relax executable and auto-apply RKNPU bridge metadata.

    This keeps bridge preload as part of the compile entry path for callers
    using the experimental chained TIR flow.
    """
    ex = relax.build(mod, *args, **kwargs)
    _attach_runtime_bridge_metadata_module(ex, mod)
    _apply_runtime_bridge_metadata_from_executable(ex)
    return ex


def build_vm_with_runtime_bridge(mod: IRModule, device, *args, **kwargs):
    """Build and construct a Relax VM while auto-applying bridge metadata."""
    if os.getenv("TVM_RKNPU_ENABLE_REAL_SUBMIT", "").lower() in ("1", "true", "yes", "on"):
        os.environ.setdefault("TVM_RKNPU_BRIDGE_REAL_SUBMIT", "1")
        os.environ.setdefault("TVM_RKNPU_BRIDGE_USE_RELOCS", "1")
        os.environ.setdefault("TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK", "1")
    ex = build_with_runtime_bridge(mod, *args, **kwargs)
    return relax.VirtualMachine(ex, device)
