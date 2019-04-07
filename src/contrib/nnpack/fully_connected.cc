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
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <nnpack.h>
#include "nnpack_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.nnpack.fully_connected_inference")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    NNPackThreadLocalEntry *entry = NNPackThreadLocalEntry::ThreadLocal();
    nnp_initialize();
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];
    NNPackConfig(args[3]);

    CHECK_EQ(A->ndim, 1);
    CHECK_EQ(B->ndim, 2);
    CHECK_EQ(C->ndim, 1);
    CHECK_EQ(B->shape[0], C->shape[0]);
    CHECK_EQ(B->shape[1], A->shape[0]);
    CHECK(C->strides == nullptr);
    CHECK(B->strides == nullptr);
    CHECK(A->strides == nullptr);
    CHECK(TypeMatch(A->dtype, kDLFloat, 32));
    CHECK(TypeMatch(B->dtype, kDLFloat, 32));
    CHECK(TypeMatch(C->dtype, kDLFloat, 32));

    nnp_fully_connected_inference(B->shape[1],
                                  B->shape[0],
                                  static_cast<float*>(A->data),
                                  static_cast<float*>(B->data),
                                  static_cast<float*>(C->data),
                                  entry->threadpool);
  });

}  // namespace contrib
}  // namespace tvm
