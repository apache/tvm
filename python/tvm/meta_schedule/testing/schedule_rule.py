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
"""Default schedule rules"""
from tvm.meta_schedule.schedule_rule import (
    AddRFactor,
    AutoInline,
    ScheduleRule,
)
from tvm.target import Target


def auto_inline(target: Target) -> ScheduleRule:
    """Default schedule rules for auto inline"""
    if target.kind.name == "llvm":
        return AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        )
    if target.kind.name == "cuda":
        return AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        )
    raise NotImplementedError(f"{target.kind.name} is not supported")


def add_rfactor(target: Target) -> ScheduleRule:
    """Default schedule rules for with add_rfactor"""
    if target.kind.name == "llvm":
        return AddRFactor(max_jobs_per_core=16, max_innermost_factor=64)
    raise NotImplementedError(f"{target.kind.name} is not supported")
