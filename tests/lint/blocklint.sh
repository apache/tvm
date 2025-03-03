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


for dir in $(ls)
do
  # Ignore the 3rdparty directory since we have no control of the language used there.
  if ! [ "$dir" == "3rdparty" ]; then
    for subdir in $(find $dir -type d -print)
    do
      blocklint --blocklist blacklist,whitelist,white\ box,master\ ,\ master,master_,_master,slave $subdir \
      --skip-files tests/lint/blocklint.sh,tests/lint/pylintrc,conda/recipe/meta.yaml,rust/tvm-sys/build.rs,src/target/source/codegen_vhls.cc,tests/micro/zephyr/test_utils.py
    done
  fi
done
