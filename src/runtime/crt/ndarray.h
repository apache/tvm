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
 * \file tvm/runtime/ndarray.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_NDARRAY_H_
#define TVM_RUNTIME_NDARRAY_H_

#include "c_runtime_api.h"
#include "c_backend_api.h"
#include "common.h"
#include "dlpack.h"

/*! \brief Magic number for NDArray file */
static const uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

typedef struct ndarray_t {
  DLTensor dl_tensor;
} NDArray;

NDArray NDArray_CreateView(NDArray * arr, int64_t * shape, DLDataType dtype);

static inline NDArray NDArray_Create(uint32_t ndim, int64_t * shape, DLDataType dtype, DLContext ctx) {
  NDArray ret;
  memset(&ret, 0, sizeof(NDArray));
  ret.dl_tensor.ndim = ndim;
  memcpy(ret.dl_tensor.shape, shape, sizeof(int64_t)*ndim);
  ret.dl_tensor.dtype = dtype;
  ret.dl_tensor.ctx = ctx;
  ret.dl_tensor.data = 0;
  return ret;
}

static inline NDArray NDArray_Empty(uint32_t ndim, int64_t * shape, DLDataType dtype, DLContext ctx) {
  NDArray ret = NDArray_Create(ndim, shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (dtype.bits + 7) / 8;
  uint32_t idx;
  for (idx = 0; idx < ret.dl_tensor.ndim; ++idx) {
    num_elems *= shape[idx];
  }
  ret.dl_tensor.data = TVMBackendAllocWorkspace(kDLCPU, 0, num_elems, dtype.code, dtype.bits);
  memset(ret.dl_tensor.data, 0, num_elems * elem_bytes);
  return ret;
}

static inline int NDArray_Load(NDArray * ret, const char ** strm) {
  API_BEGIN();
  uint64_t header, reserved;
  header = ((uint64_t*)*strm)[0]; *strm += sizeof(header);
  if (header != kTVMNDArrayMagic) {
    LOGE("Invalid DLTensor file format\n");
    status = TVM_STATUS_FAILURE;
  }
  reserved = ((uint64_t*)*strm)[0]; *strm += sizeof(reserved);
  DLContext ctx;
  uint32_t ndim;
  DLDataType dtype;
  ctx = ((DLContext*)*strm)[0]; *strm += sizeof(ctx);
  ndim = ((uint32_t*)*strm)[0]; *strm += sizeof(ndim);
  dtype = ((DLDataType*)*strm)[0]; *strm += sizeof(dtype);
  if ((ndim <= 0) || (ndim > TVM_CRT_MAX_NDIM)) {
    LOGE("Invalid ndim=%d: expected to be 1 ~ %d.\n", ndim, TVM_CRT_MAX_NDIM);
    status = TVM_STATUS_FAILURE;
  }
  if (ctx.device_type != kDLCPU) {
    LOGE("Invalid DLTensor context: can only save as CPU tensor\n");
    status = TVM_STATUS_FAILURE;
  }
  int64_t shape[TVM_CRT_MAX_NDIM]; // [ndim];
  uint32_t idx;
  if (ndim != 0) {
    for (idx = 0; idx < ndim; idx++){
      shape[idx] = ((int64_t*)*strm)[0]; *strm += sizeof(shape[idx]);
    }
  }
  *ret = NDArray_Empty(ndim, shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dl_tensor.dtype.bits + 7) / 8;
  for (idx = 0; idx < ret->dl_tensor.ndim; ++idx) {
    num_elems *= ret->dl_tensor.shape[idx];
  }
  int64_t data_byte_size;
  data_byte_size = ((int64_t*)*strm)[0]; *strm += sizeof(data_byte_size);
  if (!(data_byte_size == num_elems * elem_bytes)) {
    LOGE("invalid DLTensor file format: data_byte_size=%ld, while num_elems*elem_bytes=%ld",
         data_byte_size, (num_elems * elem_bytes));
    status = TVM_STATUS_FAILURE;
  }
  memcpy(ret->dl_tensor.data, *strm, data_byte_size);
  *strm += data_byte_size;
  API_END();
}

#endif  // TVM_RUNTIME_NDARRAY_H_
