#!/bin/bash
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
set -o pipefail

echo "Checking MyPy Type defs in the schedule package."
mypy  --check-untyped-defs python/tvm/tir/schedule

echo "Checking MyPy Type defs in the analysis package."
mypy  --check-untyped-defs python/tvm/tir/analysis/

echo "Checking MyPy Type defs in the transofrm package."
mypy  --check-untyped-defs python/tvm/tir/transform/

#TODO(@mikepapadim): This is failing atm
# echo "Checking MyPy Type defs in the tvm.relay.backend.contrib.ethosu package."
# mypy  --check-untyped-defs python/tvm/relay/backend/contrib/ethosu/
