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

import tvm
from tvm import tirx
from tvm.tirx.compilation_pipeline import finalize_host_passes

from . import transform as trn_transform


def trn_pipeline():
    """The Trainium pipeline used in tvm.tirx.build."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """Lower TIRx for the Trainium backend."""
        passes = [
            trn_transform.TrnPrivateBufferAlloc(),
            trn_transform.TrnNaiveAllocator(),
            tirx.transform.TilePrimitiveDispatch(),
            trn_transform.LowerTrainiumLayout(),
            tvm.s_tir.transform.DecorateDeviceScope(),
            tirx.transform.StmtSimplify(),
            tirx.transform.LowerTIRxOpaque(),
            tvm.s_tir.transform.LoopPartition(),
            tvm.s_tir.transform.HoistIfThenElse(),
            tirx.transform.StmtSimplify(),
            tirx.transform.RemoveNoOp(),
            tirx.transform.AnnotateEntryFunc(),
            tirx.transform.SplitHostDevice(),
            tirx.transform.MakePackedAPI(),
        ]
        return tvm.ir.transform.Sequential(passes)(mod)

    return _pipeline, finalize_host_passes, finalize_device_passes_trn


def finalize_device_passes_trn():  # pylint: disable=unused-argument
    """The finalization passes for the Trainium backend."""
    return tvm.ir.transform.Sequential([tirx.transform.StmtSimplify()])


__all__ = ["finalize_device_passes_trn", "trn_pipeline"]
