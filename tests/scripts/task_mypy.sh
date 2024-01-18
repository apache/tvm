#!/usr/bin/env bash
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

set -euxo pipefail

source tests/scripts/setup-pytest-env.sh

echo "Checking MyPy Type defs in the TensorIR schedule package."
mypy  --check-untyped-defs python/tvm/tir/schedule

echo "Checking MyPy Type defs in the meta schedule package."
mypy  --check-untyped-defs python/tvm/meta_schedule

echo "Checking MyPy Type defs in the analysis package."
mypy  --check-untyped-defs python/tvm/tir/analysis/

echo "Checking MyPy Type defs in the transform package."
mypy  --check-untyped-defs python/tvm/tir/transform/

echo "Checking MyPy Type defs in the tvmscript printer package."
mypy  --check-untyped-defs python/tvm/script/printer

echo "Checking MyPy Type defs in the TIR package with unittest"
MYPYPATH=$TVM_PATH/python mypy --check-untyped-defs tests/python/tvmscript/test_tvmscript_type.py

echo "Checking MyPy Type defs in tvm.relay.op.contrib"
mypy --disallow-untyped-defs python/tvm/relay/op/contrib/cublas.py
mypy --disallow-untyped-defs python/tvm/relay/op/contrib/cudnn.py
mypy --disallow-untyped-defs python/tvm/relay/op/contrib/te_target.py
mypy --disallow-untyped-defs python/tvm/relay/op/contrib/tensorrt.py

#TODO(@mikepapadim): This is failing atm
# echo "Checking MyPy Type defs in the tvm.relay.backend.contrib.ethosu package."
# mypy  --check-untyped-defs python/tvm/relay/backend/contrib/ethosu/

echo "Checking MyPy Type defs in the tvmscript IRBuilder package."
mypy  --check-untyped-defs python/tvm/script/ir_builder
