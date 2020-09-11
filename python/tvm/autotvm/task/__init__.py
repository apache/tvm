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
"""Task is a tunable composition of template functions.

Tuner takes a tunable task and optimizes the joint configuration
space of all the template functions in the task.
This module defines the task data structure, as well as a collection(zoo)
of typical tasks of interest.
"""

from .task import (
    Task,
    create,
    get_config,
    args_to_workload,
    template,
    serialize_args,
    deserialize_args,
)
from .space import ConfigSpace, ConfigEntity
from .code_hash import attach_code_hash, attach_code_hash_to_arg
from .dispatcher import (
    DispatchContext,
    ApplyConfig,
    ApplyHistoryBest,
    FallbackContext,
    clear_fallback_cache,
    ApplyGraphBest,
)

from .topi_integration import (
    register_topi_compute,
    register_topi_schedule,
    TaskExtractEnv,
    get_workload,
)
from .relay_integration import extract_from_program, extract_from_multiple_program
