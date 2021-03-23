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
 * \file tvm/runtime/serializer.h
 * \brief Serializer extension to support TVM data types
 *  Include this file to enable serialization of DLDataType, DLDevice
 */
#ifndef TVM_RUNTIME_SERIALIZER_H_
#define TVM_RUNTIME_SERIALIZER_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>

namespace dmlc {
namespace serializer {

template <>
struct Handler<DLDataType> {
  inline static void Write(Stream* strm, const DLDataType& dtype) {
    Handler<uint8_t>::Write(strm, dtype.code);
    Handler<uint8_t>::Write(strm, dtype.bits);
    Handler<uint16_t>::Write(strm, dtype.lanes);
  }
  inline static bool Read(Stream* strm, DLDataType* dtype) {
    if (!Handler<uint8_t>::Read(strm, &(dtype->code))) return false;
    if (!Handler<uint8_t>::Read(strm, &(dtype->bits))) return false;
    if (!Handler<uint16_t>::Read(strm, &(dtype->lanes))) return false;
    return true;
  }
};

template <>
struct Handler<DLDevice> {
  inline static void Write(Stream* strm, const DLDevice& dev) {
    int32_t device_type = static_cast<int32_t>(dev.device_type);
    Handler<int32_t>::Write(strm, device_type);
    Handler<int32_t>::Write(strm, dev.device_id);
  }
  inline static bool Read(Stream* strm, DLDevice* dev) {
    int32_t device_type = 0;
    if (!Handler<int32_t>::Read(strm, &(device_type))) return false;
    dev->device_type = static_cast<DLDeviceType>(device_type);
    if (!Handler<int32_t>::Read(strm, &(dev->device_id))) return false;
    return true;
  }
};

}  // namespace serializer
}  // namespace dmlc
#endif  // TVM_RUNTIME_SERIALIZER_H_
