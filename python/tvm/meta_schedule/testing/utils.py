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
"""Testing utility functions in meta schedule"""
from typing import Callable, Dict, Optional, Union
from tvm.ir import IRModule, transform
from tvm.relay import Function as RelayFunc
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.tir import Schedule
from tvm import meta_schedule as ms


def apply_fixed_schedules(
    relay_mod: Union[RelayFunc, IRModule],
    target: Union[str, Target],
    params: Optional[Dict[str, NDArray]],
    schedule_fn: Callable[[ms.ExtractedTask, Schedule], bool],
    te_filter_func=None,
):
    """Apply fixed schedules (manually written, without any tunable knobs) as specified by
    schedule_fn to extracted tasks, and return a database that can be passed to ApplyHistoryBest.

    Parameters
    ----------
    mod : Union[RelayFunc, IRModule]
        The Relay module to apply fixed schedules.
    target : Union[str, Target]
        The target used to extract tasks.
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the module.
    schedule_fn : Callable[[ExtractedTask, Schedule], bool]
        A callable that is applied for each extracted task and the corresponding default schedule.
        Returns True if the given schedule should be committed to the database, False otherwise.
    te_filter_func : Union[str, None, Callable[[List[Tensor], List[NDArray]], PrimFunc]] = None
        The filtering function for TE computation
        If it's a string, it's the name of the filtering function. Built in functions are
          - "meta_schedule.DefaultTaskFilter"
          - "meta_schedule.DefaultTaskFilterAllowExtern"
        If it's None, it's the default filtering function
        If it's a callable, it's the filtering function

    Returns
    -------
    database : Database
        The database containing dummy tuning records for manually scheduled traces.
    """
    target = Target(target) if isinstance(target, str) else target
    config = {"relay.backend.use_meta_schedule": True}
    for k, v in transform.PassContext.current().config.items():
        config[k] = v

    extracted_tasks = ms.extract_task_from_relay(
        relay_mod, target, params, te_filter_func=te_filter_func, pass_config=config
    )
    database = ms.database.MemoryDatabase()
    for task in extracted_tasks:
        mod = ms.default_config.mod(task.dispatched[0])
        sch = Schedule(mod)

        if schedule_fn(task, sch):
            workload = database.commit_workload(mod)
            tune_rec = ms.database.TuningRecord(sch.trace, workload, [0.0], target, [])
            database.commit_tuning_record(tune_rec)

    return database
