/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "../../../3rdparty/nvbench/l2_cache_flush.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "cuda_common.h"

namespace tvm {

namespace runtime {

typedef dmlc::ThreadLocalStore<L2Flush> L2FlushStore;

L2Flush* L2Flush::ThreadLocal() { return L2FlushStore::Get(); }

TVM_REGISTER_GLOBAL("l2_cache_flush_cuda").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK(L2Flush::ThreadLocal() != nullptr) << "L2Flush::ThreadLocal do not exist.";
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
  L2Flush::ThreadLocal()->Flush(stream);
});

}  // namespace runtime
}  // namespace tvm
