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
#ifndef TVM_RUNTIME_DISCO_UTILS_H_
#define TVM_RUNTIME_DISCO_UTILS_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/disco/disco_worker.h>

#include <string>

namespace tvm {
namespace runtime {

inline Device UseDefaultDeviceIfNone(Device device) {
  if (device.device_type == 0 && device.device_id == 0) {
    return DiscoWorker::ThreadLocal()->default_device;
  }
  return device;
}

/*!
 * \brief Converts a 1-d shape tuple to an integer.
 * \note At the time of scaffolding Disco, RelaxVM has not provided mature support for standalone
 * integers. A common workaround is to use a 1-d shape tuple as an integer.
 */
inline int64_t IntegerFromShapeTuple(const ShapeTuple& shape) {
  CHECK_EQ(shape.size(), 1) << "ValueError: shape tuple must be 1-d to be converted to integer.";
  return shape[0];
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_UTILS_H_
