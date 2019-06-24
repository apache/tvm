/*!
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
 * This Contribution is being provided by Qualcomm Technologies, Inc.,
 * a Delaware corporation, or its subsidiary Qualcomm Innovation Center, Inc.,
 * a California corporation, under certain additional terms and conditions
 * pursuant to Section 5 of the Apache 2.0 license.  In this regard, with
 * respect to this Contribution, the term "Work" in Section 1 of the
 * Apache 2.0 license means only the specific subdirectory within the TVM repo
 * (currently at https://github.com/dmlc/tvm) to which this Contribution is
 * made.
 * In any case, this submission is "Not a Contribution" with respect to its
 * permitted use with any of the "vta" and "verilog" subdirectories in the TVM
 * repo.
 * Qualcomm Technologies, Inc. and Qualcomm Innovation Center, Inc. retain
 * copyright of their respective Contributions.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <cstring>

#include "hexagon_module.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace tvm {
namespace runtime {

class HexagonDeviceAPI : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       TVMType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from, size_t from_offset, void* to,
                      size_t to_offset, size_t num_bytes, TVMContext ctx_from,
                      TVMContext ctx_to, TVMType type_hint,
                      TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final;

  static const std::shared_ptr<HexagonDeviceAPI>& Global() {
    static std::shared_ptr<HexagonDeviceAPI> inst =
        std::make_shared<HexagonDeviceAPI>();
    return inst;
  }
};

// Debugging helpers.

std::string DeviceAttrName(DeviceAttrKind kind) {
  switch (kind) {
    case kExist:
      return "kExist";
    case kMaxThreadsPerBlock:
      return "kMaxThreadsPerBlock";
    case kWarpSize:
      return "kWarpSize";
    case kMaxSharedMemoryPerBlock:
      return "kMaxSharedMemoryPerBlock";
    case kComputeVersion:
      return "kComputeVersion";
    case kDeviceName:
      return "kDeviceName";
    case kMaxClockRate:
      return "kMaxClockRate";
    case kMultiProcessorCount:
      return "kMultiProcessorCount";
    case kMaxThreadDimensions:
      return "kMaxThreadDimensions";
    default:
      break;
  }

  std::stringstream ss;
  ss << "<unknown attr, kind=" << kind << '>';
  return ss.str();
}

// HexagonDeviceAPI.

inline void HexagonDeviceAPI::SetDevice(TVMContext ctx) {}

inline void HexagonDeviceAPI::GetAttr(TVMContext ctx, DeviceAttrKind kind,
                                      TVMRetValue* rv) {
  if (kind == kExist) *rv = 1;
}

inline void* HexagonDeviceAPI::AllocDataSpace(TVMContext ctx, size_t nbytes,
                                              size_t alignment,
                                              TVMType type_hint) {
#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_VERBOSE, "TVM",
                      "Allocating memory for bytes = %zu for device:%d\n",
                      nbytes, ctx.device_id);
#endif
  CHECK(hexagon::Device::ValidateDeviceId(ctx.device_id));
  return hexagon::Device::Global()->Alloc(nbytes, alignment);
}

inline void HexagonDeviceAPI::FreeDataSpace(TVMContext ctx, void* ptr) {
  CHECK(hexagon::Device::ValidateDeviceId(ctx.device_id));
  hexagon::Device::Global()->Free(ptr);
}

inline void HexagonDeviceAPI::CopyDataFromTo(
    const void* from, size_t from_offset, void* to, size_t to_offset,
    size_t num_bytes, TVMContext ctx_from, TVMContext ctx_to,
    TVMType type_hint, TVMStreamHandle stream) {
  const char* src = static_cast<const char*>(from) + from_offset;
  char* dst = static_cast<char*>(to) + to_offset;

  auto Is32bit = [](const void* p) {
    return p == reinterpret_cast<const void*>(uint32_t(uintptr_t(p)));
  };
  (void)Is32bit;

  if (ctx_from.device_type == ctx_to.device_type) {
    if (ctx_from.device_type == kDLCPU) {
      memmove(dst, src, num_bytes);
    } else if (static_cast<int>(ctx_from.device_type) == kDLHexagon) {
      CHECK(hexagon::Device::ValidateDeviceId(ctx_from.device_id));
      CHECK_EQ(ctx_from.device_id, ctx_to.device_id);
      CHECK(Is32bit(dst) && Is32bit(src));
      hexagon::Device::Global()->CopyDeviceToDevice(dst, src, num_bytes);
    }
  } else {
    if (ctx_from.device_type == kDLCPU) {
      CHECK_EQ(static_cast<int>(ctx_to.device_type), kDLHexagon);
      CHECK(Is32bit(dst));
      CHECK(hexagon::Device::ValidateDeviceId(ctx_to.device_id));
      hexagon::Device::Global()->CopyHostToDevice(dst, src, num_bytes);
    } else {
      CHECK_EQ(static_cast<int>(ctx_from.device_type), kDLHexagon);
      CHECK_EQ(ctx_to.device_type, kDLCPU);
      CHECK(Is32bit(src));
      CHECK(hexagon::Device::ValidateDeviceId(ctx_from.device_id));
      hexagon::Device::Global()->CopyDeviceToHost(dst, src, num_bytes);
    }
  }
}

inline void HexagonDeviceAPI::StreamSync(TVMContext ctx,
                                         TVMStreamHandle stream) {}

TVM_REGISTER_GLOBAL("device_api.hexagon")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      DeviceAPI* ptr = HexagonDeviceAPI::Global().get();
      *rv = ptr;
    });
}  // namespace runtime
}  // namespace tvm
