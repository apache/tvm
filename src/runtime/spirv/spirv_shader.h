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

#ifndef TVM_RUNTIME_SPIRV_SPIRV_SHADER_H_
#define TVM_RUNTIME_SPIRV_SPIRV_SHADER_H_

#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <vector>

namespace tvm {
namespace runtime {
namespace spirv {

struct SPIRVShader {
  /*! \brief header flag */
  uint32_t flag{0};
  /*! \brief Data segment */
  std::vector<uint32_t> data;

  void Save(support::Stream* writer) const {
    writer->Write(flag);
    writer->Write(data);
  }
  bool Load(support::Stream* reader) {
    if (!reader->Read(&flag)) return false;
    if (!reader->Read(&data)) return false;
    return true;
  }
};

}  // namespace spirv

using spirv::SPIRVShader;
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace support {
template <>
struct Serializer<::tvm::runtime::spirv::SPIRVShader> {
  static constexpr bool enabled = true;
  static void Write(Stream* strm, const ::tvm::runtime::spirv::SPIRVShader& data) {
    data.Save(strm);
  }
  static bool Read(Stream* strm, ::tvm::runtime::spirv::SPIRVShader* data) {
    return data->Load(strm);
  }
};
}  // namespace support
}  // namespace tvm
#endif  // TVM_RUNTIME_SPIRV_SPIRV_SHADER_H_
