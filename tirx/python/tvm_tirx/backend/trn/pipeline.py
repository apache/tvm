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
"""Trainium TIRX pipeline entrypoints."""

import tvm_tirx
from tvm_tirx import tirx
from tvm_tirx.tirx.compilation_pipeline import _bind_target, finalize_host_passes

from . import transform as trn_transform


def trn_pipeline(*, target=None):
    """Construct Trainium lowering and host/device finalization stages.

    The lowering pass binds the target, prepares instruction workspaces and
    allocation addresses, dispatches tiles, lowers layouts, simplifies the IR,
    splits host/device functions, and creates packed host entry points.

    Parameters
    ----------
    target : Target or None
        Target to bind before lowering. If omitted, leave existing function
        target attributes unchanged.

    Returns
    -------
    stages : tuple
        The lowering ModulePass, host-finalization factory, and
        device-finalization factory consumed by ``tvm_tirx.tirx.compile``.

    Examples
    --------
    Apply the lowering stage to an IRModule ``mod``::

        from tvm_tirx.backend.trn.pipeline import trn_pipeline
        from tvm_tirx.target import Target

        lower, finalize_host, finalize_device = trn_pipeline(
            target=Target("aws/trn1/trn1.2xlarge"))
        after = lower(mod)

    A device-body TVMScript excerpt illustrates allocation and layout lowering.
    Surrounding function attributes and any generated host wrapper are omitted;
    ``T`` is ``tvm_tirx.script.tirx`` and ``P``/``F`` are Trainium axes::

        # Before: inside T.device_entry()
        A = T.alloc_tensor(ty_args=[T.Tensor(
            (128, 2, 2), "float32", scope="trn.sbuf",
            layout=T.TileLayout(T.S[(128, 4):(1 @ P, 1 @ F)]))])
        A[0, 1, 0] = T.float32(1)
        # After: lowered function body
        A = T.alloc_tensor((0,), ty_args=[T.Tensor((128, 4), "float32", scope="trn.sbuf")])
        A[0, 2] = T.float32(1)
    """

    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(
        mod: tvm_tirx.ir.IRModule, _ctx: tvm_tirx.ir.transform.PassContext
    ) -> tvm_tirx.ir.IRModule:
        """Lower TIRx for the Trainium backend."""
        mod = _bind_target(mod, target)
        passes = [
            trn_transform.TrnPrivateTensorAlloc(),
            trn_transform.TrnNaiveAllocator(),
            tirx.transform.TileDispatch(),
            trn_transform.LowerTrainiumLayout(),
            tirx.transform.DecorateDeviceScope(),
            tirx.transform.StmtSimplify(),
            tirx.transform.HoistIf(),
            tirx.transform.StmtSimplify(),
            tirx.transform.RemoveNoOp(),
            tirx.transform.AnnotateEntryFunc(),
            tirx.transform.SplitHostDevice(),
            tirx.transform.MakePackedAPI(),
        ]
        return tvm_tirx.ir.transform.Sequential(passes)(mod)

    return _pipeline, finalize_host_passes, finalize_device_passes_trn


def finalize_device_passes_trn():  # pylint: disable=unused-argument
    """Construct the statement-simplification stage for Trainium device IR.

    Examples
    --------
    Apply to an IRModule ``device_mod`` after host/device separation::

        from tvm_tirx.backend.trn.pipeline import finalize_device_passes_trn

        after = finalize_device_passes_trn()(device_mod)

    Function-body TVMScript can simplify as follows (``T`` is
    ``tvm_tirx.script.tirx``)::

        # Before
        for i in T.serial(4):
            A[i] = i // 4
        # After
        for i in T.serial(4):
            A[i] = 0
    """
    return tvm_tirx.ir.transform.Sequential([tirx.transform.StmtSimplify()])


__all__ = ["finalize_device_passes_trn", "trn_pipeline"]
