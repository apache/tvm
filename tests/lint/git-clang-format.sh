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
set -u
set -o pipefail


INPLACE_FORMAT=${INPLACE_FORMAT:=false}
LINT_ALL_FILES=true
REVISION=$(git rev-list --max-parents=0 HEAD)

while (( $# )); do
    case "$1" in
        -i)
            INPLACE_FORMAT=true
            shift 1
            ;;
        --rev)
            LINT_ALL_FILES=false
            REVISION=$2
            shift 2
            ;;
        *)
            echo "Usage: tests/lint/git-clang-format.sh [-i] [--rev <commit>]"
            echo ""
            echo "Run clang-format on files that changed since <commit> or on all files in the repo"
            echo "Examples:"
            echo "- Compare last one commit: tests/lint/git-clang-format.sh --rev HEAD~1"
            echo "- Compare against upstream/main: tests/lint/git-clang-format.sh --rev upstream/main"
            echo "The -i will use black to format files in-place instead of checking them."
            exit 1
            ;;
    esac
done


cleanup()
{
    if [ -f /tmp/$$.clang-format.txt ]; then
       echo ""
       echo "---------clang-format log----------"
        cat /tmp/$$.clang-format.txt
    fi
    rm -rf /tmp/$$.clang-format.txt
}
trap cleanup 0

CLANG_FORMAT=clang-format-15

if [ -x "$(command -v clang-format-15)" ]; then
    CLANG_FORMAT=clang-format-15
elif [ -x "$(command -v clang-format)" ]; then
    echo "clang-format might be different from clang-format-15, expect potential difference."
    CLANG_FORMAT=clang-format
else
    echo "Cannot find clang-format-15"
    exit 1
fi

# Print out specific version
${CLANG_FORMAT} --version

if [[ "$INPLACE_FORMAT" == "true" ]]; then
    echo "Running inplace git-clang-format against $REVISION"
    git-${CLANG_FORMAT} --extensions h,mm,c,cc,cu --binary=${CLANG_FORMAT} "$REVISION"
    exit 0
fi

if [[ "$LINT_ALL_FILES" == "true" ]]; then
    echo "Running git-clang-format against all C++ files"
    git-${CLANG_FORMAT} --diff --extensions h,mm,c,cc,cu --binary=${CLANG_FORMAT} "$REVISION" 1> /tmp/$$.clang-format.txt
else
    echo "Running git-clang-format against $REVISION"
    git-${CLANG_FORMAT} --diff --extensions h,mm,c,cc,cu --binary=${CLANG_FORMAT} "$REVISION" 1> /tmp/$$.clang-format.txt
fi

if grep --quiet -E "diff" < /tmp/$$.clang-format.txt; then
    echo "clang-format lint error found. Consider running clang-format-15 on these files to fix them."
    exit 1
fi
