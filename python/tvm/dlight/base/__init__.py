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
"""Base infra"""
from .analysis import (
    BlockInfo,
    IterInfo,
    normalize_prim_func,
    detect_dominant_read,
    is_broadcast_epilogue,
)
from .common_schedules import try_inline, try_inline_contiguous_spatial
from .schedule_rule import ScheduleRule
from .transform import ApplyDefaultSchedule
