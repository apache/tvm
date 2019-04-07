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
A tuner takes a task as input. It proposes some promising :any:`ConfigEntity`
in the :any:`ConfigSpace` and measure them on the real hardware. Then it
proposed the next batch of :any:`ConfigEntity` according to the measure results.
This tuning loop is repeated.
"""

from . import callback

from .tuner import Tuner

from .gridsearch_tuner import GridSearchTuner, RandomTuner
from .ga_tuner import GATuner
from .xgboost_tuner import XGBTuner
