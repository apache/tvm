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
    echo "Usage: tests/lint/git-clang-format.sh [-i] <commit>"
    echo ""
    echo "Run clang-format on files that changed since <commit>"
    echo "Examples:"
    echo "- Compare last one commit: tests/lint/git-clang-format.sh HEAD~1"
    echo "- Compare against upstream/master: tests/lint/git-clang-format.sh upstream/master"
    echo "You can also add -i option to do inplace format"
    exit 1
fi

cleanup()
{
  rm -rf /tmp/$$.clang-format.txt
}
trap cleanup 0

CLANG_FORMAT=clang-format-10

if [ -x "$(command -v clang-format-10)" ]; then
    CLANG_FORMAT=clang-format-10
elif [ -x "$(command -v clang-format)" ]; then
    echo "clang-format might be different from clang-format-10, expect potential difference."
    CLANG_FORMAT=clang-format
else
    echo "Cannot find clang-format-10"
    exit 1
fi

# Print out specific version
${CLANG_FORMAT} --version

if [[ ${INPLACE_FORMAT} -eq 1 ]]; then
    echo "Running inplace git-clang-format against" $1
    git-${CLANG_FORMAT} --extensions h,mm,c,cc --binary=${CLANG_FORMAT} $1
    exit 0
fi

echo "Running git-clang-format against" $1
git-${CLANG_FORMAT} --diff --extensions h,mm,c,cc --binary=${CLANG_FORMAT} $1 1> /tmp/$$.clang-format.txt
echo "---------clang-format log----------"
cat /tmp/$$.clang-format.txt
echo ""
if grep --quiet -E "diff" < /tmp/$$.clang-format.txt; then
    echo "clang-format lint error found. Consider running clang-format-10 on these files to fix them."
    exit 1
fi
