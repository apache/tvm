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
 * \file ndarray.c
 * \brief NDArray container infratructure.
 */

#include "ndarray.h"

NDArray NDArray_CreateView(NDArray * arr, int64_t * shape, uint32_t ndim, DLDataType dtype) {
  NDArray ret = NDArray_Create(ndim, shape, dtype, arr->dl_tensor.ctx);
  ret.dl_tensor.data = arr->dl_tensor.data;
  return ret;
}
