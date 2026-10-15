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
    """The Trainium pipeline used in tvm_tirx.tirx.compile."""

    @tvm_tirx.ir.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm_tirx.ir.IRModule, _ctx: tvm_tirx.ir.transform.PassContext) -> tvm_tirx.ir.IRModule:
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
    """The finalization passes for the Trainium backend."""
    return tvm_tirx.ir.transform.Sequential([tirx.transform.StmtSimplify()])


__all__ = ["finalize_device_passes_trn", "trn_pipeline"]
