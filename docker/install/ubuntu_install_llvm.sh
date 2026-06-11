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

wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/llvm-archive.gpg

cat > /etc/apt/sources.list.d/llvm.list <<'EOF'
deb [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-15 main
deb-src [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-15 main
deb [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main
deb-src [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main
deb [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
deb-src [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
deb [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy main
deb-src [signed-by=/usr/share/keyrings/llvm-archive.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy main
EOF

apt-get update && apt-install-and-clear -y \
     llvm-15 llvm-16 llvm-17\
     clang-15 libclang-15-dev \
     clang-16 libclang-16-dev libpolly-16-dev \
     clang-17 libclang-17-dev libpolly-17-dev libzstd-dev
