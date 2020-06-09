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
"""Namespace for Ansor auto-scheduler"""

from . import compute_dag
from . import measure
from . import serialization
from . import loop_state
from . import auto_schedule
from . import utils
from . import feature
from . import workload_registry
from . import task_scheduler

# Shortcut
from .compute_dag import ComputeDAG
from .auto_schedule import SearchTask, MetaTileRewritePolicy, TuneOption, HardwareParams
from .auto_schedule import auto_schedule
from .measure import MeasureInput, LocalBuilder, LocalRunner, RPCRunner, RPCRunnerWarpper
from .cost_model import RandomModel
from .cost_model.xgb_model import XGBModel
from .serialization import LogToFile, LogReader, best_measure_pair_in_file
from .workload_registry import register_auto_scheduler_workload_func, workload_key_to_dag
from .task_scheduler import TaskScheduler, SimpleTaskScheduler
