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
python3 -m pylint tests/python/unittest/test_crt.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/unittest/test_tvmscript_type.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_cmsisnn --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_ethosn --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/relay/aot/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/ci --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/integration/ --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/conftest.py --rcfile="$(dirname "$0")"/pylintrc

# tests/python/contrib/test_hexagon tests
python3 -m pylint tests/python/contrib/test_hexagon/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_hexagon/conv2d/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_hexagon/topi/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/contrib/test_hexagon/metaschedule_e2e/*.py --rcfile="$(dirname "$0")"/pylintrc

# tests/python/frontend tests
python3 -m pylint tests/python/frontend/caffe/test_forward.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/caffe2/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/darknet/test_forward.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/coreml/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/keras/test_forward.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/darknet/test_forward.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/oneflow/*.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/tensorflow/test_forward.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/pytorch/test_forward.py --rcfile="$(dirname "$0")"/pylintrc
python3 -m pylint tests/python/frontend/tflite/test_forward.py --rcfile="$(dirname "$0")"/pylintrc
