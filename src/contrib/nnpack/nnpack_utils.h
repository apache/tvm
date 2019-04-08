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
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#ifndef TVM_CONTRIB_NNPACK_NNPACK_UTILS_H_
#define TVM_CONTRIB_NNPACK_NNPACK_UTILS_H_
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
#include <nnpack.h>

namespace tvm {
namespace contrib {
using namespace runtime;

struct NNPackThreadLocalEntry {
  pthreadpool_t threadpool{nullptr};
  static NNPackThreadLocalEntry* ThreadLocal();
};

bool NNPackConfig(uint64_t nthreads);
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_NNPACK_NNPACK_UTILS_H_
