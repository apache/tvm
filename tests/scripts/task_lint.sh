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


echo "Check file types..."
python3 tests/lint/check_file_type.py

echo "Check ASF license header..."
java -jar /bin/apache-rat.jar -E tests/lint/rat-excludes  -d . | (grep "== File" > /tmp/$$.apache-rat.txt || true)
if grep --quiet -E "File" /tmp/$$.apache-rat.txt; then
    echo "Need to add ASF header to the following files."
    echo "----------------File List----------------"
    cat /tmp/$$.apache-rat.txt
    echo "-----------------------------------------"
    echo "Use the following steps to add the headers:"
    echo "- Create file_list.txt in your text editor"
    echo "- Copy paste the above content in file-list into file_list.txt"
    echo "- python3 tests/lint/add_asf_header.py file_list.txt"
    exit 1
fi

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
if grep --quiet -E "warning|error" < /tmp/$$.logclean.txt; then
    exit 1
fi
