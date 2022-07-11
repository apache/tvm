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
python3 -m pylint vta/python/vta --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_cmsisnn --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/relay/aot/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/ci --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/integration/ --rcfile="$(dirname "$0")"/pylintrc

# tests/python/unitest/
python3 -m pylint tests/python/unittest/test_crt.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_micro_model_library_format.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_micro_project_api.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_micro_transport.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_node_reflection.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_container.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_error.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_extension.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_graph.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_graph_cuda_graph.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_graph_debug.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_measure.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_module_based_interface.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_module_export.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_module_load.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_profiling.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_rpc.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_trace.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_runtime_vm_profiler.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_tvmscript_type.py --rcfile="$(dirname "$0")"/pylintrc
