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

source ./tests/scripts/test_task_install_dist_wheels.sh
uv pip install --python python3 --target .local/python -r docs/requirements.txt
export TVM_TIRX_LIBRARY_PATH="$PWD/.local/python/tvm_tirx/lib"
rm -rf docs/_build docs/reference/python/generated
python3 -m sphinx -E -a -b html -d docs/_build/doctrees -W --keep-going docs docs/_build/html
git rev-parse HEAD > docs/_build/html/commit_hash
tar -czf docs.tgz -C docs/_build/html .
