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

# Core runtime sources, extended by the backend modules.
tvm_file_glob(GLOB RUNTIME_SRCS
  src/runtime/*.cc
  src/runtime/rpc/minrpc/*.cc
)
# VM-owned runtime implementations, including external-module constant loading.
tvm_file_glob(GLOB RUNTIME_VM_SRCS src/runtime/vm/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_VM_SRCS})
# Note: src/runtime/extra/disco/** moves to libtvm_runtime_extra.
# Note: src/backend/*/runtime sources move to per-backend DSOs.
set(TVM_RUNTIME_EXT_OBJS "")

if(USE_RPC)
  message(STATUS "Build with RPC support...")
  tvm_file_glob(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)
