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

cleanup()
{
  rm -rf /tmp/$$.*
}
trap cleanup 0

echo "Check codestyle of c++ code..."
make cpplint
echo "Check codestyle of python code..."
make pylint
echo "Check codestyle of jni code..."
make jnilint
echo "Check documentations of c++ code..."
make doc 2>/tmp/$$.log.txt

grep -v -E "ENABLE_PREPROCESSING|unsupported tag" < /tmp/$$.log.txt > /tmp/$$.logclean.txt || true
echo "---------Error Log----------"
cat /tmp/$$.logclean.txt
echo "----------------------------"
grep -E "warning|error" < /tmp/$$.logclean.txt || true
