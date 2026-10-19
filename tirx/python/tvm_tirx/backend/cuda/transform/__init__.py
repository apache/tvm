# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""CUDA-specific TIRx transformations."""

# pylint: disable=invalid-name

from tvm_ffi import get_global_func


def LowerIket():
    """Lower frontend-only NVIDIA IKET annotations.

    This pass must run after ``tvm_tirx.tirx.transform.SplitHostDevice`` and before
    ``tvm_tirx.tirx.transform.MakePackedAPI``.  It strips annotations for regular
    compilation and emits NVIDIA IKET metadata and NativeDump placeholders
    when the IRModule is explicitly IKET-enabled.  Trace collection and
    postprocessing are owned by the external ``run-iket`` process.

    Returns
    -------
    fpass : tvm_tirx.ir.transform.Pass
        The result pass.

    Examples
    --------
    Apply to an IRModule ``mod`` with split device functions::

        from tvm_tirx.backend.cuda.transform import LowerIket

        after = LowerIket()(mod)

    Without IKET enabled, a function-body TVMScript fragment changes as below
    (``T`` is ``tvm_tirx.script.tirx``). The store is preserved::

        # Before
        T.cuda.iket_mark("store")
        A[0] = 1
        # After
        A[0] = 1
    """
    return get_global_func("tirx.backend.cuda.transform.LowerIket")()


def BindBackendConfig():
    """Resolve CUDA function and device-region targets and compilation options.

    Run after ``BindTarget`` and before target-sensitive tile dispatch. Only
    CUDA-targeted functions are processed. Function defaults and entry overrides
    become resolved ``target`` and JSON ``backend_config`` attributes; tensor
    computations are unchanged.

    Examples
    --------
    Apply to an IRModule ``mod`` whose functions have CUDA targets::

        from tvm_tirx.backend.cuda.transform import BindBackendConfig

        after = BindBackendConfig()(mod)

    The device-region TVMScript changes as follows (``T`` is
    ``tvm_tirx.script.tirx``). Here ``entry_target`` denotes the resolved CUDA
    Target with architecture ``sm_90a``, and ``resolved_config`` denotes its
    canonical backend-configuration JSON string::

        # Before
        with T.device_entry(backend_config={"cuda": {"arch": "sm_90a"}}):
            A[0] = 1
        # After
        with T.device_entry(attrs={"target": entry_target,
                                   "backend_config": resolved_config}):
            A[0] = 1
    """
    return get_global_func("tirx.backend.cuda.transform.BindBackendConfig")("")


def SpecializeDeviceHelpers():
    """Specialize private device helpers for each CUDA compilation group.

    Run after ``SplitHostDevice``. Kernel target and backend configuration define
    each group. Called helpers receive those attributes and are cloned when
    their settings differ; helpers already in the same group can be reused.
    Unreferenced device-only helpers are omitted from the output module.

    Examples
    --------
    Apply to a split IRModule ``mod`` with resolved CUDA settings::

        from tvm_tirx.backend.cuda.transform import SpecializeDeviceHelpers

        after = SpecializeDeviceHelpers()(mod)

    For a kernel whose settings differ from its helper's, the call-site
    TVMScript changes as below (``Module`` denotes the surrounding IRModule).
    The new helper has the original body and the kernel's target/configuration;
    ``helper_cuda`` is a fresh generated name::

        # Before: inside the split kernel
        A[0] = Module.helper(A[0])
        # After: inside the same kernel
        A[0] = Module.helper_cuda(A[0])
    """
    return get_global_func("tirx.backend.cuda.transform.SpecializeDeviceHelpers")()


def SpecializeEntryHelpers():
    """Specialize helper calls using each CUDA device entry's resolved settings.

    Run after ``BindBackendConfig`` and before ``TileDispatch`` and
    ``SplitHostDevice``. Each entry's target and backend configuration propagate
    through its helper call graph. Helpers with different settings are cloned;
    original helpers are removed only when cloning makes them unreachable.

    Examples
    --------
    Apply to an IRModule ``mod`` with resolved entry settings::

        from tvm_tirx.backend.cuda.transform import SpecializeEntryHelpers

        after = SpecializeEntryHelpers()(mod)

    When an entry's configuration differs from the helper's, its TVMScript
    changes as below (``T`` is ``tvm_tirx.script.tirx``). ``entry_attrs`` stands
    for the unchanged resolved region attributes, and ``helper_cuda`` is a fresh
    helper name with the entry's settings and the original helper's body::

        # Before
        with T.device_entry(attrs=entry_attrs):
            A[0] = Module.helper(A[0])
        # After
        with T.device_entry(attrs=entry_attrs):
            A[0] = Module.helper_cuda(A[0])
    """
    return get_global_func("tirx.backend.cuda.transform.SpecializeEntryHelpers")()


__all__ = ["BindBackendConfig", "LowerIket", "SpecializeDeviceHelpers", "SpecializeEntryHelpers"]
