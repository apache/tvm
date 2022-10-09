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
    feature_extractor,
    measure_callback,
    mutator,
    postproc,
    relay_integration,
    runner,
    schedule_rule,
    search_strategy,
    space_generator,
    tir_integration,
)
from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .extracted_task import ExtractedTask
from .feature_extractor import FeatureExtractor
from .measure_callback import MeasureCallback
from .mutator import Mutator
from .postproc import Postproc
from .profiler import Profiler
from .relay_integration import (
    is_meta_schedule_dispatch_enabled,
    is_meta_schedule_enabled,
)
from .runner import Runner
from .schedule_rule import ScheduleRule
from .search_strategy import MeasureCandidate, SearchStrategy
from .space_generator import SpaceGenerator
from .tir_integration import tune_tir
from .tune import tune_tasks
from .tune_context import TuneContext
from .utils import derived_object
