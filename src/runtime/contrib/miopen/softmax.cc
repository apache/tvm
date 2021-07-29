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
 * \file src/runtime/contrib/miopen/softmax.cc
 * \brief Use external miopen softmax function
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

#include "miopen_utils.h"

namespace tvm {
namespace contrib {
namespace miopen {

using namespace runtime;

void softmax_impl(TVMArgs args, TVMRetValue* ret, miopenSoftmaxAlgorithm_t alg) {
  DLTensor* x = args[0];
  DLTensor* y = args[1];
  int axis = args[2];
  int ndim = x->ndim;
  int64_t* shape = x->shape;
  if (axis < 0) axis += ndim;
  ICHECK(axis >= 0 && axis < ndim);
  // just fp32 for now
  ICHECK(TypeMatch(x->dtype, kDLFloat, 32));
  ICHECK(TypeMatch(y->dtype, kDLFloat, 32));

  MIOpenThreadEntry* entry_ptr = MIOpenThreadEntry::ThreadLocal();

  miopenSoftmaxMode_t mode;
  if (axis == ndim - 1) {
    int64_t N = 1;
    for (int i = 0; i < ndim - 1; ++i) {
      N *= shape[i];
    }
    mode = MIOPEN_SOFTMAX_MODE_INSTANCE;
    MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->softmax_entry.shape_desc, miopenFloat,
                                            static_cast<int>(N), static_cast<int>(shape[ndim - 1]),
                                            1, 1));
  } else {
    int64_t pre_axis_dim = 1;
    int64_t post_axis_dim = 1;
    for (int i = 0; i < ndim; ++i) {
      if (i < axis) {
        pre_axis_dim *= shape[i];
      } else if (i > axis) {
        post_axis_dim *= shape[i];
      }
    }
    mode = MIOPEN_SOFTMAX_MODE_CHANNEL;
    MIOPEN_CALL(miopenSet4dTensorDescriptor(
        entry_ptr->softmax_entry.shape_desc, miopenFloat, static_cast<int>(pre_axis_dim),
        static_cast<int>(shape[axis]), static_cast<int>(post_axis_dim), 1));
  }

  const float alpha = 1.f;
  const float beta = 0.f;
  MIOPEN_CALL(miopenSoftmaxForward_V2(entry_ptr->handle, &alpha,
                                      entry_ptr->softmax_entry.shape_desc, x->data, &beta,
                                      entry_ptr->softmax_entry.shape_desc, y->data, alg, mode));
}

TVM_REGISTER_GLOBAL("tvm.contrib.miopen.softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      softmax_impl(args, ret, MIOPEN_SOFTMAX_ACCURATE);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.miopen.log_softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) { softmax_impl(args, ret, MIOPEN_SOFTMAX_LOG); });

}  // namespace miopen
}  // namespace contrib
}  // namespace tvm
