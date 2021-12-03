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

#ifndef TVM_RUNTIME_HEXAGON_PROXY_RPC_COMMON_H_
#define TVM_RUNTIME_HEXAGON_PROXY_RPC_COMMON_H_

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <string>
#include <vector>

struct HandlePacket {
  int ndim;
  uint32_t handles[];
  int size() const { return size(ndim); }
  static int size(int ndim) { return sizeof(HandlePacket) + ndim * sizeof(uint32_t); }
};

struct tensor_meta {
  int ndim;
  DLDataType dtype;
  int64_t shape[];

  int meta_size() const { return meta_size(ndim); }
  int data_size() const {
    int size = tvm::runtime::DataType(dtype).bytes();
    for (int d = 0; d != ndim; ++d) {
      size *= shape[d];
    }
    return size;
  }

  static int meta_size(int ndim) { return sizeof(tensor_meta) + ndim * sizeof(int64_t); }

  std::string to_string() const;
};

#endif  // TVM_RUNTIME_HEXAGON_PROXY_RPC_COMMON_H_
