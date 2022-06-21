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
sysctl kernel.core_pattern
ulimit -c
# mkdir -p /var/crash
ls /var/crash
# rm -f /var/crash/*
echo 'int main() { int* x = 0; return *x; }' > test.c
gcc -g test.c
./a.out || true
ls /var/crash

for file in $(find /var/crash -type f); do
echo 'backtrace' | gdb --quiet ./a.out "$file"
done
