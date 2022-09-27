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
"""Package `tvm.meta_schedule`. The meta schedule infrastructure."""
from . import (
    arg_info,
    builder,
    cost_model,
    database,
    default_config,
    feature_extractor,
    measure_callback,
    mutator,
    postproc,
    runner,
    schedule_rule,
    search_strategy,
    space_generator,
)
from .extracted_task import ExtractedTask
from .profiler import Profiler
from .relay_integration import (
    extract_task_from_relay,
    is_meta_schedule_dispatch_enabled,
    is_meta_schedule_enabled,
)
from .search_strategy import MeasureCandidate
from .tune import TuneConfig, tune_extracted_tasks, tune_relay, tune_te, tune_tir
from .tune_context import TuneContext
from .utils import derived_object
