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
 * \file ndarray.c
 * \brief NDArray container infratructure.
 */

#include "common.h"
#include "ndarray.h"
#include "c_runtime_api.h"
#include "c_backend_api.h"
#include "c_api_common.h"

NDArray NDArray_CreateView(NDArray * arr, int64_t * shape, DLDataType dtype) {
  uint32_t ndim = Shape_CountNonZero(shape);
  NDArray ret = NDArray_Create(ndim, shape, dtype, arr->dl_tensor.ctx);
  ret.dl_tensor.data = arr->dl_tensor.data;
  return ret;
}

int TVMArrayAlloc(const tvm_index_t* shape,
                  uint8_t ndim,
                  uint8_t dtype_code,
                  uint8_t dtype_bits,
                  uint16_t dtype_lanes,
                  uint8_t device_type,
                  uint8_t device_id,
                  TVMArrayHandle out) {
  API_BEGIN();
  uint32_t idx = 0;
  DLDataType dtype;
  dtype.code = dtype_code;
  dtype.bits = dtype_bits;
  dtype.lanes = dtype_lanes;
  DLContext ctx;
  ctx.device_type = device_type;
  ctx.device_id = device_id;
  out->ctx = ctx;
  out->ndim = ndim;
  out->dtype = dtype;
  uint32_t bytes = (dtype_bits + 7) / 8;
  uint32_t size = 1;
  for (idx = 0; idx < ndim; idx++) {
    size *= shape[idx];
  }
  out->data = TVMBackendAllocWorkspace(device_type, device_id, size, dtype_code, dtype_bits);
  memset(out->data, 0, size * bytes);
  for (idx = 0; idx < ndim; idx++) {
    out->shape[idx] = shape[idx];
    out->strides = 0;
  }
  out->byte_offset = 0;
  API_END();
}

int TVMArrayFree(TVMArrayHandle handle) {
  API_BEGIN();
  API_END();
}

