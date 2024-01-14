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
"""
Apply ScheduleRules onto an IRModule to generate default schedules without tuning,
or a space for MetaSchedule tuning
"""
from typing import List, Optional

from tvm import tir
from tvm import dlight as dl
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.target import Target
from .roller.policy import DefaultPolicy
from .roller.arch import CUDA
from .schedule_rule import ScheduleRule
from .analysis import get_root_block, get_reduction_blocks
from .utils import apply_and_build


def _is_scheduled(func: tir.PrimFunc) -> bool:
    if not isinstance(func, tir.PrimFunc):
        return False
    if not func.attrs:
        return False
    if "tir.is_scheduled" not in func.attrs:
        return False
    return func.attrs["tir.is_scheduled"] == 1


@module_pass(opt_level=0, name="ApplyDefaultSchedule")
class ApplyDefaultSchedule:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module."""

    def __init__(self, *rules: ScheduleRule):
        """Construct a new ApplyDefaultSchedule pass.

        Parameters
        ----------
        *rules : ScheduleRule
            The ScheduleRules to apply to all PrimFuncs in the module.
        """
        self.rules = list(rules)

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        target = Target.current(allow_none=False)
        updated_functions = {}
        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc) and not _is_scheduled(func):
                sch = _apply_rules(func, target, self.rules, tunable=False)
                if sch is not None:
                    assert len(sch) == 1
                    updated_functions[g_var] = sch[0].mod["main"].with_attr("tir.is_scheduled", 1)
        for g_var, func in updated_functions.items():
            mod[g_var] = func
        return mod


@module_pass(opt_level=0, name="ApplyFastTuning")
class ApplyFastTuning:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module."""

    def __init__(self, topk: int = 10):
        """Construct a new ApplyFastTuning pass.

        Parameters
        ----------
        *rules : ScheduleRule
            The ScheduleRules to apply to all PrimFuncs in the module.
        """
        self.topk = topk

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        target = Target.current(allow_none=False)
        updated_functions = {}

        def _apply_with_default(func: tir.PrimFunc, target: Target):
            _default_rules = [
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            ]
            sch = _apply_rules(func, target, _default_rules, tunable=False)
            if sch is not None:
                assert len(sch) == 1
                updated_functions[g_var] = sch[0].mod["main"].with_attr("tir.is_scheduled", 1)

        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc) and not _is_scheduled(func):
                arch = CUDA(target)
                print(f"[FastDlight] is scheduling {g_var}")
                # TODO(lei): should analysis the prim func to enable the right policy
                # (Default Policy for general or TensorCore Policy for tensorcore)
                policy = DefaultPolicy(func=func, arch=arch)
                configs = policy.emit_config(self.topk)
                if configs:
                    _, best = apply_and_build(
                        func=func, configs=configs, arch=arch, parallel_build=True
                    )
                    if best is not None:
                        updated_functions[g_var] = best.sch.mod["main"].with_attr(
                            "tir.is_scheduled", 1
                        )
                    else:
                        print(
                            f"[FastDlight] warnning: {g_var} has no valid config, fallback to default schedule"
                        )
                        _apply_with_default(func, target)
                else:
                    print(
                        f"[FastDlight] warnning: {g_var} has no valid config, fallback to default schedule"
                    )
                    _apply_with_default(func, target)

        for g_var, func in updated_functions.items():
            mod[g_var] = func
        return mod


def _apply_rules(
    func: tir.PrimFunc,
    target: Target,
    rules: List[ScheduleRule],
    tunable: bool,
) -> Optional[List[tir.Schedule]]:
    for rule in rules:
        space = rule.apply(func, target, tunable)
        if space is None:
            continue
        if isinstance(space, tir.Schedule):
            space = [space]
        return space
    return None
