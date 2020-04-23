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

# Precheck if sphinx docs build can fail.
set -e
set -u
set -o pipefail

cleanup()
{
    # cat error log if non zero exit
    if [ $? ]; then
        cat /tmp/$$.log.txt
    fi
    rm -rf /tmp/$$.*
}
trap cleanup 0

# cleanup cache
rm -rf docs/tutorials
rm -rf docs/vta/tutorials
find . -type f -path "*.pyc" | xargs rm -f
make cython3

echo "PreCheck sphinx doc generation WARNINGS.."
cd docs
make clean
TVM_TUTORIAL_EXEC_PATTERN=none make html 2>/tmp/$$.log.txt

grep -v -E "__mro__|RemovedIn|UserWarning|FutureWarning|Keras" < /tmp/$$.log.txt > /tmp/$$.logclean.txt || true
echo "---------Sphinx Log----------"
cat /tmp/$$.logclean.txt
echo "-----------------------------"
if grep --quiet -E "WARN" < /tmp/$$.logclean.txt; then
    echo "WARNINIG found in the log, please fix them."
    echo "You can reproduce locally by running ./tests/script/task_sphinx_precheck.sh"
    exit 1
fi
echo "No WARNINGS to be fixed."
