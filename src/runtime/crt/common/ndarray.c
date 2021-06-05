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
#include <tvm/runtime/crt/page_allocator.h>
#include <tvm/runtime/crt/platform.h>

#include "crt_config.h"

int TVMNDArray_Create(int32_t ndim, const tvm_index_t* shape, DLDataType dtype, DLDevice dev,
                      TVMNDArray* array) {
  memset(array, 0, sizeof(TVMNDArray));
  array->dl_tensor.ndim = ndim;
  tvm_crt_error_t err;
  DLDevice dldev = {kDLCPU, 0};
  err = TVMPlatformMemoryAllocate(sizeof(int64_t) * ndim, dldev, (void*)&array->dl_tensor.shape);
  if (err != kTvmErrorNoError) {
    return -1;
  }
  memcpy(array->dl_tensor.shape, shape, sizeof(int64_t) * ndim);
  array->dl_tensor.dtype = dtype;
  array->dl_tensor.device = dev;
  array->dl_tensor.data = 0;
  return 0;
}

int TVMNDArray_Empty(int32_t ndim, const tvm_index_t* shape, DLDataType dtype, DLDevice dev,
                     TVMNDArray* array) {
  int status = TVMNDArray_Create(ndim, shape, dtype, dev, array);
  if (status != 0) {
    return status;
  }
  int64_t num_elems = 1;
  int32_t idx;
  for (idx = 0; idx < array->dl_tensor.ndim; ++idx) {
    num_elems *= shape[idx];
  }
  int total_elem_bytes = (num_elems * dtype.bits + 7) / 8;
  array->dl_tensor.data =
      TVMBackendAllocWorkspace(kDLCPU, 0, total_elem_bytes, dtype.code, dtype.bits);
  memset(array->dl_tensor.data, 0, total_elem_bytes);
  return 0;
}

int TVMNDArray_Load(TVMNDArray* ret, const char** strm) {
  int32_t status = 0;
  uint64_t header, reserved;
  memcpy(&header, *strm, sizeof(header));
  *strm += sizeof(header);
  if (header != kTVMNDArrayMagic) {
    fprintf(stderr, "Invalid DLTensor file format\n");
    status = -1;
  }
  memcpy(&reserved, *strm, sizeof(reserved));
  *strm += sizeof(reserved);
  DLDevice dev;
  int ndim;  // sizeof ndim should match dlpack
  DLDataType dtype;
  memcpy(&dev, *strm, sizeof(dev));
  *strm += sizeof(dev);
  memcpy(&ndim, *strm, sizeof(ndim));
  *strm += sizeof(ndim);
  memcpy(&dtype, *strm, sizeof(dtype));
  *strm += sizeof(dtype);
  if ((ndim < 0) || (ndim > TVM_CRT_MAX_NDIM)) {
    fprintf(stderr, "Invalid ndim=%d: expected to be 0 ~ %d.\n", ndim, TVM_CRT_MAX_NDIM);
    status = -1;
  }
  if (dev.device_type != kDLCPU) {
    fprintf(stderr, "Invalid DLTensor device: can only save as CPU tensor\n");
    status = -1;
  }
  int64_t shape[TVM_CRT_MAX_NDIM] = {0};
  int32_t idx;
  if (ndim != 0) {
    for (idx = 0; idx < ndim; idx++) {
      memcpy(&shape[idx], *strm, sizeof(int64_t));
      *strm += sizeof(shape[idx]);
    }
  }
  status = TVMNDArray_Empty(ndim, shape, dtype, dev, ret);
  if (status != 0) {
    return status;
  }
  int64_t num_elems = 1;
  int elem_bytes = (ret->dl_tensor.dtype.bits + 7) / 8;
  for (idx = 0; idx < ret->dl_tensor.ndim; ++idx) {
    num_elems *= ret->dl_tensor.shape[idx];
  }
  int64_t data_byte_size;
  memcpy(&data_byte_size, *strm, sizeof(data_byte_size));
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

int TVMNDArray_CreateView(TVMNDArray* arr, const tvm_index_t* shape, int32_t ndim, DLDataType dtype,
                          TVMNDArray* array_view) {
  int status = TVMNDArray_Create(ndim, shape, dtype, arr->dl_tensor.device, array_view);
  if (status != 0) {
    return status;
  }
  array_view->dl_tensor.data = arr->dl_tensor.data;
  return 0;
}

int TVMNDArray_Release(TVMNDArray* arr) {
  tvm_crt_error_t err;
  DLDevice dev = {kDLCPU, 0};

  err = TVMPlatformMemoryFree(arr->dl_tensor.data, dev);
  if (err != kTvmErrorNoError) {
    return err;
  }

  arr->dl_tensor.data = 0;
  err = TVMPlatformMemoryFree(arr->dl_tensor.shape, dev);
  if (err != kTvmErrorNoError) {
    return err;
  }

  arr->dl_tensor.shape = 0;
  return 0;
}
