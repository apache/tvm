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

# ruff: noqa: I001

# Op class declarations (Add, Sub, Gemm, ...) — must run first so their
# `op = Op.get("tirx.<name>")` registrations execute before any dispatch
# code refers to the same ops.
from .ops import *

# Dispatch infrastructure. Per-backend schedule registrations are loaded via
# ``tvm.backend.load(<name>)``.
from .dispatcher import fail, list_registered_schedules, predicate, register_dispatch
from .registry import DispatchContext

__all__ = ["DispatchContext", "fail", "list_registered_schedules", "predicate", "register_dispatch"]
