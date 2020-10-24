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

/*!
 * \file Use external nnpack library call.
 */
#include "nnpack_utils.h"

namespace tvm {
namespace contrib {
using namespace runtime;

typedef dmlc::ThreadLocalStore<NNPackThreadLocalEntry> NNPackThreadLocalStore;

NNPackThreadLocalEntry* NNPackThreadLocalEntry::ThreadLocal() {
  return NNPackThreadLocalStore::Get();
}

bool NNPackConfig(uint64_t nthreads) {
  NNPackThreadLocalEntry* entry = NNPackThreadLocalEntry::ThreadLocal();
  if (entry->threadpool && pthreadpool_get_threads_count(entry->threadpool) == nthreads) {
    ICHECK_NE(nthreads, 1);
    return true;
  }
  if (entry->threadpool) {
    pthreadpool_destroy(entry->threadpool);
    entry->threadpool = nullptr;
  }

  if (nthreads == 1) {
    // a null threadpool means the function is invoked on the calling thread,
    // which is the desired logic for nthreads == 1
    ICHECK(!entry->threadpool);
    return true;
  }

  entry->threadpool = pthreadpool_create(nthreads);
  return true;
}

TVM_REGISTER_GLOBAL("contrib.nnpack._initialize").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = nnp_initialize();
});

}  // namespace contrib
}  // namespace tvm
