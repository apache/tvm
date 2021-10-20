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
 * \file Use external cblas library call.
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

extern "C" {
#include <dnnl.h>
}

#include "gemm_common.h"

namespace tvm {
namespace contrib {

using namespace runtime;
inline char MKLDNNBooleanToTransposeChar(bool trans) { return trans ? 'T' : 'N'; }

struct MKLDNNSgemmOp {
  typedef float TDatatype;
  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, float* A, int lda, float* B,
                  int ldb, float beta, float* C, int ldc) {
    dnnl_sgemm(MKLDNNBooleanToTransposeChar(tb), MKLDNNBooleanToTransposeChar(ta), N, M, K, alpha,
               B, ldb, A, lda, beta, C, ldc);
  }
};

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.mkldnn.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  ICHECK(TypeMatch(A->dtype, kDLFloat, 32));
  CallGemm(args, ret, MKLDNNSgemmOp());
});
}  // namespace contrib
}  // namespace tvm
