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
 * \file spirv_support
 *
 * \brief Utility for determining which spirv capabilities a TVM
 * target supports.
 */

#include "spirv_support.h"

#include <spirv.hpp>

namespace tvm {
namespace codegen {

SPIRVSupport::SPIRVSupport(tvm::Target target) {
  ICHECK_EQ(target->kind->device_type, kDLVulkan)
      << "SPIRVSupport can only be checked for vulkan device type";

  // Currently, this codifies the assumptions that were present and
  // implicit in previous implementations.  In the future, this will
  // pull information from the specified `Target`.

  supports_storage_buffer_storage_class = (SPV_VERSION >= 0x10300);
  supports_storage_buffer_8bit_access = true;
  supports_storage_buffer_16bit_access = true;
  supports_float16 = true;
  supports_int8 = true;
  supports_int16 = true;
  supports_int64 = true;
}

}  // namespace codegen
}  // namespace tvm
