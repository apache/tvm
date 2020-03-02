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
 * \file tvm/runtime/crt/ndarray.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_CRT_NDARRAY_H_
#define TVM_RUNTIME_CRT_NDARRAY_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <dlpack/dlpack.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*! \brief Magic number for NDArray file */
static const uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

/*! \brief Magic number for NDArray list file  */
static const uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

typedef struct TVMNDArray {
  DLTensor dl_tensor;
} TVMNDArray;

TVMNDArray TVMNDArray_Create(uint32_t ndim, const tvm_index_t * shape,
                             DLDataType dtype, DLContext ctx);

TVMNDArray TVMNDArray_Empty(uint32_t ndim, const tvm_index_t * shape,
                            DLDataType dtype, DLContext ctx);

int TVMNDArray_Load(TVMNDArray * ret, const char ** strm);

TVMNDArray TVMNDArray_CreateView(TVMNDArray * arr, const tvm_index_t * shape,
                                 uint32_t ndim, DLDataType dtype);

int TVMNDArray_Release(TVMNDArray * arr);

#endif  // TVM_RUNTIME_CRT_NDARRAY_H_
