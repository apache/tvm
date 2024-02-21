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
 * \file test_extern_module.cc
 * \brief Testing code to be compiled by Relax nn.SourceModule
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/packed_func.h>

namespace {

int _scalar_add(DLTensor* a, DLTensor* b, DLTensor* c) {
  using namespace tvm::runtime;
  ICHECK(a->ndim == 0);
  ICHECK(b->ndim == 0);
  ICHECK(c->ndim == 0);
  ICHECK(DataType(a->dtype) == DataType::Float(32));
  ICHECK(DataType(b->dtype) == DataType::Float(32));
  ICHECK(DataType(c->dtype) == DataType::Float(32));
  float* a_data = static_cast<float*>(a->data);
  float* b_data = static_cast<float*>(b->data);
  float* c_data = static_cast<float*>(c->data);
  *c_data = *a_data + *b_data;
  return 0;
}

int _test_sym(DLTensor* a, DLTensor* b, DLTensor* c) {
  using namespace tvm::runtime;
  ICHECK(a->ndim == 3);  // [x, y, 1]
  ICHECK(b->ndim == 3);  // [y, z, 5]
  ICHECK(c->ndim == 4);  // [x, y, z, 9]
  ICHECK(DataType(a->dtype) == DataType::Float(32));
  ICHECK(DataType(b->dtype) == DataType::Float(32));
  ICHECK(DataType(c->dtype) == DataType::Float(32));
  int x = a->shape[0];
  int y = a->shape[1];
  int z = b->shape[1];
  ICHECK(a->shape[0] == x);
  ICHECK(a->shape[1] == y);
  ICHECK(a->shape[2] == 1);
  ICHECK(b->shape[0] == y);
  ICHECK(b->shape[1] == z);
  ICHECK(b->shape[2] == 5);
  ICHECK(c->shape[0] == x);
  ICHECK(c->shape[1] == y);
  ICHECK(c->shape[2] == z);
  ICHECK(c->shape[3] == 9);
  return 0;
}
}  // namespace
TVM_DLL_EXPORT_TYPED_FUNC(ext_scalar_add, _scalar_add);
TVM_DLL_EXPORT_TYPED_FUNC(ext_test_sym, _test_sym);
