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
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.target import Target

from .schedule_rule import ScheduleRule
from .roller.policy import DefaultPolicy
from .roller.arch import CUDA

def _is_scheduled(func: tir.PrimFunc) -> bool:
    if not isinstance(func, tir.PrimFunc):
        return False
    if not func.attrs:
        return False
    if "tir.is_scheduled" not in func.attrs:
        return False
    return func.attrs["tir.is_scheduled"] == 1


@module_pass(opt_level=0, name="ApplyFastTuning")
class ApplyFastTuning:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module."""

    def __init__(self, topk:int=10, *rules: ScheduleRule):
        """Construct a new ApplyFastTuning pass.

        Parameters
        ----------
        *rules : ScheduleRule
            The ScheduleRules to apply to all PrimFuncs in the module.
        """
        self.topk = topk
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
                # sch = _apply_rules(func, target, self.rules, tunable=False)
                policy = DefaultPolicy(func=func, arch=CUDA(target))
                configs = policy.emit_config(10)
                for config in configs:
                    sch = _apply_rules(func, target, self.rules, config)
                if sch is not None:
                    assert len(sch) == 1
                    updated_functions[g_var] = sch[0].mod["main"].with_attr("tir.is_scheduled", 1)
        for g_var, func in updated_functions.items():
            mod[g_var] = func
        return mod


def _apply_rules(
    func: tir.PrimFunc,
    target: Target,
    rules: List[ScheduleRule],
    config,
) -> Optional[List[tir.Schedule]]:
    for rule in rules:
        space = rule.apply_config(func, target, config)
        if space is None:
            continue
        if isinstance(space, tir.Schedule):
            space = [space]
        return space
    return None
