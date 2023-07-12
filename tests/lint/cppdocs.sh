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

function cleanup() {
    rm -f /tmp/$$.log.txt /tmp/$$.logclean.txt
}
trap cleanup EXIT

make cppdoc 2>/tmp/$$.log.txt

grep -v -E "ENABLE_PREPROCESSING|unsupported tag|Inheritance graph" < /tmp/$$.log.txt > /tmp/$$.logclean.txt || true
echo "---------Error Log----------"
cat /tmp/$$.logclean.txt
echo "----------------------------"
if grep --quiet -E "warning|error" < /tmp/$$.logclean.txt; then
    exit 1
fi
