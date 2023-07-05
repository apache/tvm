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
# pylint: disable=unused-import, redefined-builtin
""" Namespace for TVM Auto-scheduler. """

from . import (
    compute_dag,
    dispatcher,
    feature,
    loop_state,
    measure,
    measure_record,
    relay_integration,
    search_policy,
    search_task,
    task_scheduler,
    utils,
    workload_registry,
)

# Shortcut
from .compute_dag import (
    ComputeDAG,
    LayoutRewriteOption,
    get_shape_from_rewritten_layout,
)
from .cost_model import RandomModel, XGBModel
from .dispatcher import ApplyHistoryBest, ApplyHistoryBestOrSample, DispatchContext
from .measure import (
    LocalBuilder,
    LocalRPCMeasureContext,
    LocalRunner,
    MeasureInput,
    MeasureResult,
    RPCRunner,
    register_task_input_check_func,
)
from .measure_record import (
    RecordReader,
    RecordToFile,
    load_best_record,
    load_records,
    save_records,
)
from .relay_integration import (
    extract_tasks,
    is_auto_scheduler_enabled,
    remove_index_check,
    rewrite_compute_body,
    rewrite_tensor_shape,
)
from .search_policy import (
    EmptyPolicy,
    PreloadCustomSketchRule,
    PreloadMeasuredStates,
    SketchPolicy,
)
from .search_task import (
    HardwareParams,
    SearchTask,
    TuningOptions,
    auto_schedule,
    create_task,
)
from .task_scheduler import TaskScheduler
from .workload_registry import make_workload_key, register_workload
