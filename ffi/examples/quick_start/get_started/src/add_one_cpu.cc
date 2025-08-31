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

#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

namespace tvm_ffi_example {

void AddOne(DLTensor* x, DLTensor* y) {
  // implementation of a library function
  TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
  DLDataType f32_dtype{kDLFloat, 32, 1};
  TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
  TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
  TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
  TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";
  for (int i = 0; i < x->shape[0]; ++i) {
    static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 1;
  }
}

// Expose global symbol `add_one_cpu` that follows tvm-ffi abi
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, tvm_ffi_example::AddOne);
}  // namespace tvm_ffi_example
