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

python3 -m pylint python/tvm --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/tvmscript/test_tvmscript_type.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/ci --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/integration/ --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/conftest.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_cblas.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_tflite_runtime.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_thrust.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_util.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_sort.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_sparse.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_tedd.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_rpc_tracker.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_rpc_server_device.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_rpc_proxy.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_rocblas.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_random.py --rcfile="$(dirname "$0")"/pylintrc

# tests/python/contrib/test_hexagon tests
python3 -m pylint tests/python/contrib/test_hexagon/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_hexagon/conv2d/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_hexagon/topi/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_hexagon/metaschedule_e2e/*.py --rcfile="$(dirname "$0")"/pylintrc

# tests/python/contrib/test_msc tests
python3 -m pylint tests/python/contrib/test_msc/*.py --rcfile="$(dirname "$0")"/pylintrc
