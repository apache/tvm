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

cleanup()
{
  rm -rf /tmp/$$.*
}
trap cleanup 0


# These shards are solely for CI to enable the lint job to have some parallelism.

function shard1 {
  # echo "Check Jenkinsfile generation"
  # python3 ci/jenkins/generate.py --check

  echo "Checking file types..."
  python3 tests/lint/check_file_type.py

  echo "Checking CMake <-> LibInfo options mirroring"
  python3 tests/lint/check_cmake_options.py

  echo "black check..."
  tests/lint/git-black.sh

  echo "Linting the Python code with flake8..."
  tests/lint/flake8.sh

#  echo "Type checking with MyPy ..."
#  tests/scripts/task_mypy.sh

  echo "Checking for non-inclusive language with blocklint..."
  tests/lint/blocklint.sh

  echo "Linting the JNI code..."
  tests/lint/jnilint.sh
}

function shard2 {
  echo "check whitespace..."
  tests/lint/whitespace.sh

  echo "Linting the Python code with pylint..."
  tests/lint/pylint.sh

  echo "Checking C++ documentation..."
  tests/lint/cppdocs.sh

  echo "Checking ASF license headers..."
  tests/lint/check_asf_header.sh --local

  echo "Linting the C++ code..."
  tests/lint/cpplint.sh

  echo "clang-format check..."
  tests/lint/git-clang-format.sh

  echo "Rust check..."
  tests/lint/rust_format.sh

  echo "Docker check..."
  tests/lint/docker-format.sh
}


if [[ -n ${TVM_SHARD_INDEX+x} ]]; then
  if [[ "$TVM_SHARD_INDEX" == "0" ]]; then
    shard1
  else
    shard2
  fi
else
  shard1
  shard2
fi
