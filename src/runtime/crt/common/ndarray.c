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

// LINT_C_FILE

/*!
 * \file ndarray.c
 * \brief NDArray container infratructure.
 */

#include <tvm/runtime/crt/internal/common/ndarray.h>
#include <tvm/runtime/crt/memory.h>

#include "crt_config.h"

TVMNDArray TVMNDArray_Create(int32_t ndim, const tvm_index_t* shape, DLDataType dtype,
                             DLContext ctx) {
  TVMNDArray ret;
  memset(&ret, 0, sizeof(TVMNDArray));
  ret.dl_tensor.ndim = ndim;
  ret.dl_tensor.shape = (int64_t*)vmalloc(sizeof(int64_t) * ndim);  // NOLINT(*)
  memcpy(ret.dl_tensor.shape, shape, sizeof(int64_t) * ndim);
  ret.dl_tensor.dtype = dtype;
  ret.dl_tensor.ctx = ctx;
  ret.dl_tensor.data = 0;
  return ret;
}

TVMNDArray TVMNDArray_Empty(int32_t ndim, const tvm_index_t* shape, DLDataType dtype,
                            DLContext ctx) {
  TVMNDArray ret = TVMNDArray_Create(ndim, shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (dtype.bits + 7) / 8;
  int32_t idx;
  for (idx = 0; idx < ret.dl_tensor.ndim; ++idx) {
    num_elems *= shape[idx];
  }
  ret.dl_tensor.data = TVMBackendAllocWorkspace(kDLCPU, 0, num_elems, dtype.code, dtype.bits);
  memset(ret.dl_tensor.data, 0, num_elems * elem_bytes);
  return ret;
}

int TVMNDArray_Load(TVMNDArray* ret, const char** strm) {
  int32_t status = 0;
  uint64_t header, reserved;
  header = ((uint64_t*)*strm)[0];  // NOLINT(*)
  *strm += sizeof(header);
  if (header != kTVMNDArrayMagic) {
    fprintf(stderr, "Invalid DLTensor file format\n");
    status = -1;
  }
  reserved = ((uint64_t*)*strm)[0];  // NOLINT(*)
  *strm += sizeof(reserved);
  DLContext ctx;
  int ndim;  // sizeof ndim should match dlpack
  DLDataType dtype;
  ctx = ((DLContext*)*strm)[0];  // NOLINT(*)
  *strm += sizeof(ctx);
  ndim = ((int*)*strm)[0];  // NOLINT(*)
  *strm += sizeof(ndim);
  dtype = ((DLDataType*)*strm)[0];  // NOLINT(*)
  *strm += sizeof(dtype);
  if ((ndim < 0) || (ndim > TVM_CRT_MAX_NDIM)) {
    fprintf(stderr, "Invalid ndim=%d: expected to be 0 ~ %d.\n", ndim, TVM_CRT_MAX_NDIM);
    status = -1;
  }
  if (ctx.device_type != kDLCPU) {
    fprintf(stderr, "Invalid DLTensor context: can only save as CPU tensor\n");
    status = -1;
  }
  int64_t shape[TVM_CRT_MAX_NDIM] = {0};
  int32_t idx;
  if (ndim != 0) {
    for (idx = 0; idx < ndim; idx++) {
      shape[idx] = ((int64_t*)*strm)[0];  // NOLINT(*)
      *strm += sizeof(shape[idx]);
    }
  }
  *ret = TVMNDArray_Empty(ndim, shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dl_tensor.dtype.bits + 7) / 8;
  for (idx = 0; idx < ret->dl_tensor.ndim; ++idx) {
    num_elems *= ret->dl_tensor.shape[idx];
  }
  int64_t data_byte_size;
  data_byte_size = ((int64_t*)*strm)[0];  // NOLINT(*)
  *strm += sizeof(data_byte_size);
  if (!(data_byte_size == num_elems * elem_bytes)) {
    fprintf(stderr,
            "invalid DLTensor file format: data_byte_size=%d, "
            "while num_elems*elem_bytes=%d\n",
            (int)data_byte_size, (int)(num_elems * elem_bytes));  // NOLINT(*)
    status = -1;
  }
  memcpy(ret->dl_tensor.data, *strm, data_byte_size);
  *strm += data_byte_size;

  return status;
}

TVMNDArray TVMNDArray_CreateView(TVMNDArray* arr, const tvm_index_t* shape, int32_t ndim,
                                 DLDataType dtype) {
  TVMNDArray ret = TVMNDArray_Create(ndim, shape, dtype, arr->dl_tensor.ctx);
  ret.dl_tensor.data = arr->dl_tensor.data;
  return ret;
}

int TVMNDArray_Release(TVMNDArray* arr) {
  vfree(arr->dl_tensor.data);
  arr->dl_tensor.data = 0;
  vfree(arr->dl_tensor.shape);
  arr->dl_tensor.shape = 0;
  return 0;
}
