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

LINT_ALL_FILES=true
REVISION=

while (( $# )); do
    case "$1" in
        --rev)
            LINT_ALL_FILES=false
            REVISION=$2
            shift 2
            ;;
        *)
            echo "Usage: tests/lint/cpplint.sh [--rev <commit>]"
            exit 1
            ;;
    esac
done

if [[ "$LINT_ALL_FILES" == "true" ]]; then
    echo "Running 2 cpplints..."
    python3 3rdparty/dmlc-core/scripts/lint.py --quiet tvm cpp \
        include src \
        examples/extension/src examples/graph_executor/src \
        tests/cpp tests/crt \
        --exclude_path  "src/runtime/hexagon/rpc/hexagon_rpc.h" \
                "src/runtime/hexagon/rpc/hexagon_rpc_skel.c" \
                "src/runtime/hexagon/rpc/hexagon_rpc_stub.c" \

else
    echo "Running cpplint on changed files..."
    # Get changed files, filtering by the directories we care about
    # We use git diff to find changes.
    # We filter for the directories: include, src, examples/extension/src, examples/graph_executor/src, tests/cpp, tests/crt

    # grep pattern construction
    DIRS="include|src|examples/extension/src|examples/graph_executor/src|tests/cpp|tests/crt"

    # Read files into array
    IFS=$'\n' read -a FILES -d'\n' < <(git diff --name-only --diff-filter=ACMRTUX $REVISION | grep -E "^($DIRS)/" ) || true

    # Filter out excluded files
    FILTERED_FILES=()
    for f in "${FILES[@]}"; do
        if [[ "$f" == "src/runtime/hexagon/rpc/hexagon_rpc.h" ]] || \
           [[ "$f" == "src/runtime/hexagon/rpc/hexagon_rpc_skel.c" ]] || \
           [[ "$f" == "src/runtime/hexagon/rpc/hexagon_rpc_stub.c" ]]; then
            continue
        fi
        FILTERED_FILES+=("$f")
    done

    if [ ${#FILTERED_FILES[@]} -eq 0 ]; then
        echo "No changes in C++ files"
    else
        python3 3rdparty/dmlc-core/scripts/lint.py --quiet tvm cpp "${FILTERED_FILES[@]}"
    fi
fi

if find src -name "*.cc" -exec grep -Hn '^#include <regex>$' {} +; then
    echo "The <regex> header file may not be used in TVM," 1>&2
    echo "because it causes ABI incompatibility with most pytorch installations." 1>&2
    echo "Pytorch packages on PyPI currently set \`-DUSE_CXX11_ABI=0\`," 1>&2
    echo "which causes ABI compatibility when calling <regex> functions." 1>&2
    echo "See https://github.com/pytorch/pytorch/issues/51039 for more details." 1>&2
    exit 1
fi
