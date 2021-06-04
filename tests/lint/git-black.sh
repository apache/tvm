#!/bin/bash
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
set -u
set -o pipefail

if [[ "$1" == "-i" ]]; then
    INPLACE_FORMAT=1
    shift 1
else
    INPLACE_FORMAT=0
fi

if [[ "$#" -lt 1 ]]; then
    echo "Usage: tests/lint/git-black.sh [-i] <commit>"
    echo ""
    echo "Run black on Python files that changed since <commit>"
    echo "Examples:"
    echo "- Compare last one commit: tests/lint/git-black.sh HEAD~1"
    echo "- Compare against upstream/main: tests/lint/git-black.sh upstream/main"
    echo "The -i will use black to format files in-place instead of checking them."
    exit 1
fi

# required to make black's dep click to work
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [ ! -x "$(command -v black)" ]; then
    echo "Cannot find black"
    exit 1
fi

# Print out specific version
echo "Version Information: $(black --version)"

# Compute Python files which changed to compare.
IFS=$'\n' read -a FILES -d'\n' < <(git diff --name-only --diff-filter=ACMRTUX $1 -- "*.py" "*.pyi") || true
echo "Read returned $?"
if [ -z ${FILES+x} ]; then
    echo "No changes in Python files"
    exit 0
fi
echo "Files: $FILES"

if [[ ${INPLACE_FORMAT} -eq 1 ]]; then
    echo "Running black on Python files against revision" $1:
    CMD=( "black" "${FILES[@]}" )
    echo "${CMD[@]}"
    "${CMD[@]}"
else
    echo "Running black in checking mode"
    python3 -m black --diff --check ${FILES[@]}
fi
