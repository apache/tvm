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

set -e

echo "Running 2 cpplints..."
python3 3rdparty/dmlc-core/scripts/lint.py --quiet tvm cpp \
	include src \
	examples/extension/src examples/graph_executor/src \
	tests/cpp tests/crt \
	--exclude_path  "src/runtime/hexagon/rpc/hexagon_rpc.h" \
			"src/runtime/hexagon/rpc/hexagon_rpc_skel.c" \
			"src/runtime/hexagon/rpc/hexagon_rpc_stub.c" \
			"src/relay/backend/contrib/libtorch/libtorch_codegen.cc"


if find src -name "*.cc" -exec grep -Hn '^#include <regex>$' {} +; then
    echo "The <regex> header file may not be used in TVM," 1>&2
    echo "because it causes ABI incompatibility with most pytorch installations." 1>&2
    echo "Pytorch packages on PyPI currently set `-DUSE_CXX11_ABI=0`," 1>&2
    echo "which causes ABI compatibility when calling <regex> functions." 1>&2
    echo "See https://github.com/pytorch/pytorch/issues/51039 for more details." 1>&2
    exit 1
fi
