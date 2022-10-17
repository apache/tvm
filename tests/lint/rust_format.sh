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

TVM_HOME="$(git rev-parse --show-toplevel)"
RUST_DIR="$TVM_HOME/rust"

if [[ "$1" == "-i" ]]; then
    INPLACE_FORMAT=1
    shift 1
else
    INPLACE_FORMAT=0
fi

cd $RUST_DIR

if [[ ${INPLACE_FORMAT} -eq 1 ]]; then
    cargo fmt
else
    cargo fmt -- --check
fi
