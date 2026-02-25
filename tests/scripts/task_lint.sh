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

cleanup() {
  rm -rf /tmp/$$.*
}
trap cleanup 0

# These shards are solely for CI to enable the lint job to have some parallelism.

function shard1 {
  echo "Installing pre-commit..."
  pip install pre-commit

  echo "Running pre-commit hooks..."
  pre-commit run --all-files

  echo "Checking C++ documentation..."
  tests/lint/cppdocs.sh

  echo "Linting the C++ code (regex header check)..."
  tests/lint/cpplint.sh

  echo "Docker check..."
  tests/lint/docker-format.sh
}

function shard2 {
  # shard2 is a no-op: the old lint checks have been replaced by pre-commit hooks in shard1.
  echo "Lint shard2: no-op (checks moved to pre-commit in shard1)"
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
