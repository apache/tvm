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

export PYTHONPATH=python
export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH:-}"

# to avoid CI CPU thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

CURR_DIR=$(cd `dirname $0`; pwd)
SCRIPT_DIR=$CURR_DIR/../../jvm/core/src/test/scripts
TEMP_DIR=$(mktemp -d)

cleanup()
{
  rm -rf "$TEMP_DIR"
}
trap cleanup 0

python3 "$SCRIPT_DIR"/test_add_cpu.py "$TEMP_DIR"
python3 "$SCRIPT_DIR"/test_add_gpu.py "$TEMP_DIR"
python3 "$SCRIPT_DIR"/test_graph_executor.py "$TEMP_DIR"

# Skip the Java RPC Unittests, see https://github.com/apache/tvm/issues/13168
# # start rpc proxy server
# PORT=$(( ( RANDOM % 1000 )  + 9000 ))
# python3 $SCRIPT_DIR/test_rpc_proxy_server.py $PORT 30 &

# make jvmpkg
# make jvmpkg JVM_TEST_ARGS="-DskipTests=false \
#   -Dtest.tempdir=$TEMP_DIR \
#   -Dtest.rpc.proxy.host=localhost \
#   -Dtest.rpc.proxy.port=$PORT"
