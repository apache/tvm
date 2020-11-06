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
"""The auto-tuning module of tvm

This module includes:

* Tuning space definition API

* Efficient auto-tuners

* Tuning result and database support

* Distributed measurement to scale up tuning
"""

from . import database
from . import feature
from . import measure
from . import record
from . import task
from . import tuner
from . import utils
from . import env
from . import tophub

# some shortcuts
from .measure import (
    measure_option,
    MeasureInput,
    MeasureResult,
    MeasureErrorNo,
    LocalBuilder,
    LocalRunner,
    RPCRunner,
)
from .tuner import callback
from .task import (
    get_config,
    create,
    ConfigSpace,
    ConfigEntity,
    register_topi_compute,
    register_topi_schedule,
    template,
    DispatchContext,
    FallbackContext,
    ApplyHistoryBest as apply_history_best,
    ApplyGraphBest as apply_graph_best,
)
from .env import GLOBAL_SCOPE
