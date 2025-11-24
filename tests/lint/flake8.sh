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

set -euo pipefail

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
            echo "Usage: tests/lint/flake8.sh [--rev <commit>]"
            echo ""
            echo "Run flake8 on Python files that changed since <commit> or on all files in the repo"
            echo "Examples:"
            echo "- Compare last one commit: tests/lint/flake8.sh --rev HEAD~1"
            echo "- Compare against upstream/main: tests/lint/flake8.sh --rev upstream/main"
            exit 1
            ;;
    esac
done

if [[ "$LINT_ALL_FILES" == "true" ]]; then
    echo "Running flake8 on all files"
    python3 -m flake8 . --count --select=E9,F63,F7 --show-source --statistics --exclude 3rdparty
else
    # Get changed Python files, excluding 3rdparty
    IFS=$'\n' read -a FILES -d'\n' < <(git diff --name-only --diff-filter=ACMRTUX $REVISION -- "*.py" "*.pyi" | grep -v "^3rdparty/") || true
    if [ -z ${FILES+x} ] || [ ${#FILES[@]} -eq 0 ]; then
        echo "No changes in Python files"
        exit 0
    fi
    echo "Running flake8 on changed files: ${FILES[@]}"
    python3 -m flake8 ${FILES[@]} --count --select=E9,F63,F7 --show-source --statistics
fi
