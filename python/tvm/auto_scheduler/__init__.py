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

from . import compute_dag
from . import dispatcher
from . import feature
from . import loop_state
from . import measure
from . import measure_record
from . import relay_integration
from . import search_policy
from . import search_task
from . import task_scheduler
from . import utils
from . import workload_registry

# Shortcut
from .compute_dag import ComputeDAG, LayoutRewriteOption, get_shape_from_rewritten_layout
from .cost_model import RandomModel, XGBModel
from .dispatcher import DispatchContext, ApplyHistoryBest, ApplyHistoryBestOrSample
from .measure import (
    MeasureInput,
    MeasureResult,
    LocalBuilder,
    LocalRunner,
    RPCRunner,
    LocalRPCMeasureContext,
    register_task_input_check_func,
)
from .measure_record import RecordToFile, RecordReader, load_best_record, load_records, save_records
from .relay_integration import (
    extract_tasks,
    remove_index_check,
    rewrite_compute_body,
    is_auto_scheduler_enabled,
)
from .search_task import SearchTask, TuningOptions, HardwareParams, create_task, auto_schedule
from .search_policy import (
    EmptyPolicy,
    SketchPolicy,
    PreloadMeasuredStates,
    PreloadCustomSketchRule,
)
from .task_scheduler import TaskScheduler
from .workload_registry import register_workload, make_workload_key
