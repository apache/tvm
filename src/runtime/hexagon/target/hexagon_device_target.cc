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

#ifdef __ANDROID__

#include <unistd.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "../hexagon_module.h"
#include "AEEStdErr.h"
#include "fastrpc/include/tvm_remote.h"
#include "hexagon_dsprpcapi.h"
#include "hexagon_stubapi.h"
#include "hexagon_target_log.h"
#include "remote64.h"
#include "rpcmem.h"

#pragma weak remote_session_control

#define RPCMEM_HEAP 25

// All log messages start with "HexagonTarget::%s", where %s is replaced
// with the function name, so create macros that add that to avoid repetition.
// The downside is that the format string must be given as a string literal,
// but it seems to be a minor issue.
#define VA_EXPANDER(...) , ##__VA_ARGS__
#define TVM_LOGD_HT(fmt, ...) \
  TVM_LOGD("HexagonTarget::%s: " fmt, __func__ VA_EXPANDER(__VA_ARGS__))
#define TVM_LOGE_HT(fmt, ...) \
  TVM_LOGE("HexagonTarget::%s: " fmt, __func__ VA_EXPANDER(__VA_ARGS__))

namespace tvm {
namespace runtime {
namespace hexagon {

static constexpr int kStackSize = 128 * 1024;  // 128kB stack

class HexagonTarget : public tvm::runtime::hexagon::Device {
 public:
  HexagonTarget() {}
  ~HexagonTarget() final {}
  void* Alloc(unsigned size, unsigned align) final;
  void Free(void* ptr) final;
  void* AllocVtcm(unsigned size, unsigned align) final;
  void FreeVtcm(void* ptr) final;
  void CopyDeviceToDevice(void* dst, const void* src, unsigned len) final;
  void CopyDeviceToHost(void* host_dst, const void* src, unsigned len) final;
  void CopyHostToDevice(void* dst, const void* host_src, unsigned len) final;
  void* Load(const std::string& data, const std::string& fmt) final;
  void Unload(void* mod) final;
  void* Resolve(const std::string& sym) final;
  void Call(void* func, uint32_t* scalar, unsigned scalar_num, uint32_t* stack,
            unsigned stack_num) final;

 private:
  std::pair<void*, size_t> AddAddrMapping(const void* dsp_addr,
                                          void* apps_addr, size_t size);
  std::pair<void*, size_t> GetAppsAddr(const void* dsp_addr, bool exact) const;
  void RemoveAddrMapping(const void* dsp_addr);
  int OpenDomainChannel(bool set_unsigned_pd);
  int CloseDomainChannel();
  void ReleaseLibrary();
  void FreeMemoryBeforeChannelClose();

  // Mapping from a DSP address to a pair <apps address, buffer size>.
  // Using void* pointers is ok, since DSP pointers will always fit
  // in apps's pointers, i.e. sizeof_dsp(void*) <= sizeof_apps(void*).
  std::map<const void*, std::pair<void*, size_t>> dsp_to_apps_;
  remote_handle64 domain_channel_handle_ = AEE_EUNKNOWN;
  tvm_remote_handle_t module_pointer_ = AEE_EUNKNOWN;
  uint64_t count_channel_open_ = 0;
  // Global lock, used for all critical sections. This can be refined
  // in the future.
  mutable std::mutex crit_section_;

  // Don't use unsigned PDs by default. Change this to "true" to enable.
  static constexpr bool unsigned_pd = false;

  static void* const vtcm_mark_;
};

void* const HexagonTarget::vtcm_mark_ = reinterpret_cast<void*>(~0);

std::shared_ptr<Device> CreateHexagonTarget() {
  return std::make_shared<HexagonTarget>();
}

std::pair<void*, size_t> HexagonTarget::AddAddrMapping(const void* dsp_addr,
                                                       void* apps_addr,
                                                       size_t size) {
  crit_section_.lock();
  auto p = dsp_to_apps_.insert({dsp_addr, {apps_addr, size}});
  crit_section_.unlock();
  if (!p.second) {
    TVM_LOGE_HT(
        "failed to insert address mapping: dsp:%p -> apps:%p, size:%zu",
        dsp_addr, apps_addr, size);
    return std::make_pair(nullptr, 0);
  }
  TVM_LOGD_HT("added address mapping: dsp:%p -> apps:%p, size:%zu", dsp_addr,
              apps_addr, size);
  return p.first->second;
}

void HexagonTarget::RemoveAddrMapping(const void* dsp_addr) {
  crit_section_.lock();
  auto f = dsp_to_apps_.find(dsp_addr);
  if (f == dsp_to_apps_.end()) {
    TVM_LOGE_HT("failed to remove address mapping for dsp:%p", dsp_addr);
    crit_section_.unlock();
    return;
  }
  dsp_to_apps_.erase(f);
  crit_section_.unlock();
}

std::pair<void*, size_t> HexagonTarget::GetAppsAddr(const void* dsp_addr,
                                                    bool exact) const {
  struct AutoUnlock {
    explicit AutoUnlock(std::mutex& m) : m(m) {}
    ~AutoUnlock() { m.unlock(); }
    std::mutex& m;
  };

  crit_section_.lock();
  AutoUnlock u(crit_section_);

  // If the address is in the map, simply return the result.
  auto f = dsp_to_apps_.find(dsp_addr);
  if (f != dsp_to_apps_.end()) return f->second;
  // If exact mapping is requested, then it hasn't been found.
  if (exact) return std::make_pair(nullptr, 0);

  // If the address is not in the map, maybe it points to somewhere in the
  // interior of a mapped buffer.
  uintptr_t dsp_v = reinterpret_cast<uintptr_t>(dsp_addr);
  for (const auto& v : dsp_to_apps_) {
    uintptr_t dsp_k = reinterpret_cast<uintptr_t>(v.first);
    size_t size = v.second.second;
    if (dsp_v >= dsp_k && dsp_v < dsp_k + size) {
      uintptr_t apps_k = reinterpret_cast<uintptr_t>(v.second.first);
      size_t offset = dsp_v - dsp_k;
      uintptr_t apps_v = apps_k + offset;
      return std::make_pair(reinterpret_cast<void*>(apps_v), size - offset);
    }
  }
  TVM_LOGE_HT("failed to locate apps address for dsp:%p", dsp_addr);
  return std::make_pair(nullptr, 0);
}

int HexagonTarget::OpenDomainChannel(bool use_unsigned_pd) {
  if (domain_channel_handle_ != AEE_EUNKNOWN) return AEE_SUCCESS;

  const DspRpcAPI* dsp_api = DspRpcAPI::Global();
  const StubAPI* stub_api = StubAPI::Global();

  stub_api->rpcmem_init_ptr()();

  if (auto* rsc_ptr = dsp_api->remote_session_control_ptr(true)) {
    remote_rpc_thread_params th_data;
    th_data.domain = CDSP_DOMAIN_ID;
    th_data.stack_size = kStackSize;
    th_data.prio = -1;  // Default priority.
    int rc = rsc_ptr(FASTRPC_THREAD_PARAMS, &th_data, sizeof(th_data));
    if (rc != AEE_SUCCESS) {
      TVM_LOGE_HT("remote_session_control failed rc=%08x for stack size", rc);
    }
    if (use_unsigned_pd) {
      remote_rpc_control_unsigned_module data;
      data.enable = 1;
      data.domain = CDSP_DOMAIN_ID;
      int rc = rsc_ptr(DSPRPC_CONTROL_UNSIGNED_MODULE, &data, sizeof(data));
      if (rc != AEE_SUCCESS) {
        TVM_LOGE_HT("remote_session_control failed rc=%08x for unsigned PD",
                    rc);
      }
    }
  } else {
    TVM_LOGD_HT("remote_session_control not available");
  }

  int rc = stub_api->tvm_remote_open(tvm_remote_URI "&_dom=cdsp",
                                     &domain_channel_handle_);
  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT("failed to open channel rc=0x%x", rc);
  } else {
    count_channel_open_++;
    TVM_LOGD_HT("channel open success and rpcmem_init done");
  }
  return rc;
}

int HexagonTarget::CloseDomainChannel() {
  if (domain_channel_handle_ == AEE_EUNKNOWN) return AEE_SUCCESS;

  const StubAPI* stub_api = StubAPI::Global();

  int rc = stub_api->tvm_remote_close(domain_channel_handle_);
  if (rc == AEE_SUCCESS) {
    domain_channel_handle_ = AEE_EUNKNOWN;
    stub_api->rpcmem_deinit_ptr()();
    TVM_LOGD_HT("channel close success and rpcmem_deinit done");
  } else {
    TVM_LOGE_HT("failed to close domain channel rc=0x%x", rc);
  }
  return rc;
}

void HexagonTarget::ReleaseLibrary() {
  crit_section_.lock();
  if (module_pointer_ != AEE_EUNKNOWN) {
    const StubAPI* stub_api = StubAPI::Global();
    int rc = stub_api->tvm_remote_release_library(domain_channel_handle_,
                                                  module_pointer_);
    if (rc != AEE_SUCCESS) {
      TVM_LOGE_HT("failed to unload device library rc=0x%x", rc);
    } else {
      module_pointer_ = AEE_EUNKNOWN;
    }
  }
  crit_section_.unlock();
}

void HexagonTarget::FreeMemoryBeforeChannelClose() {
  while (!dsp_to_apps_.empty()) {
    void* dsp_addr = const_cast<void*>((dsp_to_apps_.begin()->first));
    TVM_LOGD_HT("Freeing up dsp_addr %p", dsp_addr);
    HexagonTarget::Free(dsp_addr);
  }
}

void* HexagonTarget::Alloc(unsigned size, unsigned align) {
  const DspRpcAPI* dsp_api = DspRpcAPI::Global();
  const StubAPI* stub_api = StubAPI::Global();

  // Opening the domain channel should be done once.
  crit_section_.lock();
  int rc_oc = OpenDomainChannel(/*use_unsigned_pd*/ unsigned_pd);
  crit_section_.unlock();
  if (rc_oc != AEE_SUCCESS) {
    TVM_LOGE_HT("mem alloc failed: unable to open domain channel");
    return nullptr;
  }

  // This is a workaround. If HexagonTarget::Alloc is called from a different
  // thread then remote_mmap64 fails. FastRPC expects one call to be made to
  // DSP before calling remote_map64. Hence this call is needed for now untill
  // FastRPC comes up with a fix.
  int rc_call_mmap_64 =
      stub_api->tvm_remote_call_mmap64(domain_channel_handle_);
  if (rc_call_mmap_64 != AEE_SUCCESS) {
    TVM_LOGE_HT("mmap64 failed for domain channel %lu",
                domain_channel_handle_);
    return nullptr;
  }

  void* mem =
      stub_api->rpcmem_alloc_ptr()(RPCMEM_HEAP, RPCMEM_DEFAULT_FLAGS, size);
  if (mem == nullptr) {
    TVM_LOGE_HT("mem alloc failed for size=0x%x alignment=0x%x", size, align);
    return nullptr;
  }
  int mem_fd = stub_api->rpcmem_to_fd_ptr()(mem);
  uintptr_t dsp_va = 0;
  int rc = dsp_api->remote_mmap64_ptr()(
      mem_fd, 0, reinterpret_cast<uintptr_t>(mem), size, &dsp_va);
  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT(
        "buffer mapping failed for remote_map64 fd=0x%x rc=0x%x "
        "apps_addr=0x%lx",
        mem_fd, rc, reinterpret_cast<uintptr_t>(mem));
    return nullptr;
  }

  void* dsp_addr = reinterpret_cast<void*>(dsp_va);
  AddAddrMapping(dsp_addr, mem, size);
  return dsp_addr;
}

void HexagonTarget::Free(void* ptr) {
  const DspRpcAPI* dsp_api = DspRpcAPI::Global();
  const StubAPI* stub_api = StubAPI::Global();
  auto bb = GetAppsAddr(ptr, true);
  if (bb.first == vtcm_mark_) {
    TVM_LOGD_HT("VTCM mapping found. dsp_addr=0x%p", ptr);
    RemoveAddrMapping(ptr);
    FreeVtcm(ptr);
    return;
  }

  TVM_LOGD_HT("VTCM mapping not found. dsp_addr=0x%p", ptr);
  auto aa = GetAppsAddr(ptr, true);
  if (aa.first == nullptr) return;

  int rc = dsp_api->remote_munmap64_ptr()(reinterpret_cast<uintptr_t>(ptr),
                                          aa.second);
  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT("buffer unmapping failed rc=0x%x", rc);
  }
  RemoveAddrMapping(ptr);
  stub_api->rpcmem_free_ptr()(aa.first);
}

void* HexagonTarget::AllocVtcm(unsigned size, unsigned align) {
  const StubAPI* stub_api = StubAPI::Global();

  unsigned int dsp_va = 0;
  int rc = stub_api->tvm_remote_alloc_vtcm(domain_channel_handle_, size, align,
                                           &dsp_va);
  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT("VTCM allocation failed size=%u, align=%u", size, align);
    return nullptr;
  }
  void* dsp_addr = reinterpret_cast<void*>(dsp_va);
  TVM_LOGD_HT("Done vtcm alloc dsp:%p", dsp_addr);
  AddAddrMapping(dsp_addr, vtcm_mark_, size);
  return dsp_addr;
}

void HexagonTarget::FreeVtcm(void* ptr) {
  const StubAPI* stub_api = StubAPI::Global();

  TVM_LOGD_HT("%s:Calling vtcm free. ptr=%p", __func__, ptr);
  uintptr_t dsp_va = reinterpret_cast<uintptr_t>(ptr);
  int rc = stub_api->tvm_remote_free_vtcm(domain_channel_handle_, dsp_va);
  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT("VTCM deallocation failed");
  }
  TVM_LOGD_HT("Done VTCM free from HexagonTarget::FreeVtcm");
}

void HexagonTarget::CopyDeviceToDevice(void* dst, const void* src,
                                       unsigned len) {
  auto aa_src = GetAppsAddr(src, false);
  auto aa_dst = GetAppsAddr(dst, false);
  if (aa_src.first == vtcm_mark_ || aa_dst.first == vtcm_mark_) {
    TVM_LOGE_HT("VTCM address. Copy operation not supported");
    return;
  }
  if (!aa_src.first || !aa_dst.first) {
    TVM_LOGE_HT("copy failed, dsp:%p -> dsp:%p, len:%u", src, dst, len);
    return;
  }
  if (aa_src.second < len) {
    TVM_LOGD_HT(
        "specified length:%u larger than source buffer size:%zu, copy "
        "truncated",
        len, aa_src.second);
  }
  if (aa_dst.second < len) {
    TVM_LOGD_HT(
        "specified length:%u larger than dest buffer size:%zu, copy "
        "truncated",
        len, aa_dst.second);
  }
  len = std::min({size_t(len), aa_src.second, aa_dst.second});
  TVM_LOGD_HT("copy, dsp:%p(apps:%p) -> dsp:%p(apps:%p), len:%u", src,
              aa_src.first, dst, aa_dst.first, len);
  std::memcpy(aa_dst.first, aa_src.first, len);
}

void HexagonTarget::CopyDeviceToHost(void* host_dst, const void* src,
                                     unsigned len) {
  auto aa = GetAppsAddr(src, false);
  if (aa.first == vtcm_mark_) {
    TVM_LOGE_HT("VTCM address. Copy operation not supported");
    return;
  }
  if (!aa.first) {
    TVM_LOGE_HT("copy failed, dsp:%p -> apps:%p, len:%u", src, host_dst, len);
    return;
  }
  if (aa.second < len) {
    TVM_LOGD_HT(
        "specified length:%u larger than buffer size:%zu, copy truncated", len,
        aa.second);
    len = aa.second;
  }
  TVM_LOGD_HT("copy, dsp:%p(apps:%p) -> apps:%p, len:%u", src, aa.first,
              host_dst, len);
  std::memcpy(host_dst, aa.first, len);
}

void HexagonTarget::CopyHostToDevice(void* dst, const void* host_src,
                                     unsigned len) {
  auto aa = GetAppsAddr(dst, false);
  if (aa.first == vtcm_mark_) {
    TVM_LOGE_HT("VTCM address. Copy operation not supported");
    return;
  }
  if (!aa.first) {
    TVM_LOGE_HT("copy failed, dsp:%p <- apps:%p, len:%u", dst, host_src, len);
    return;
  }
  if (aa.second < len) {
    TVM_LOGD_HT(
        "specified length:%u larger than buffer size:%zu, copy truncated", len,
        aa.second);
    len = aa.second;
  }
  TVM_LOGD_HT("copy, dsp:%p(apps:%p) <- apps:%p, len:%u", dst, aa.first,
              host_src, len);
  std::memcpy(aa.first, host_src, len);
}

void* HexagonTarget::Load(const std::string& data, const std::string& fmt) {
  crit_section_.lock();
  int rc_oc = OpenDomainChannel(/*use_unsigned_pd*/ unsigned_pd);
  crit_section_.unlock();
  if (rc_oc != AEE_SUCCESS) {
    TVM_LOGE_HT("loading of %s failed: unable to open domain channel",
                data.c_str());
    return nullptr;
  }

  if (domain_channel_handle_ == AEE_EUNKNOWN) return nullptr;
  ReleaseLibrary();

  crit_section_.lock();
  TVM_LOGD_HT("loading library %s ", data.c_str());
  const StubAPI* stub_api = StubAPI::Global();
  int rc = stub_api->tvm_remote_load_library(
      domain_channel_handle_, data.c_str(), data.size() + 1, &module_pointer_);
  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT("failed to load device library rc=0x%x", rc);
  }
  crit_section_.unlock();

  if (module_pointer_ != AEE_EUNKNOWN) {
    return reinterpret_cast<void*>(module_pointer_);
  } else {
    return nullptr;
  }
}

void HexagonTarget::Unload(void* mod) {
  crit_section_.lock();
  count_channel_open_--;
  crit_section_.unlock();
  if (count_channel_open_ == 0) FreeMemoryBeforeChannelClose();

  ReleaseLibrary();
  if (module_pointer_ != AEE_EUNKNOWN) return;

  crit_section_.lock();
  if (count_channel_open_ == 0) CloseDomainChannel();
  crit_section_.unlock();
}

void* HexagonTarget::Resolve(const std::string& sym) {
  const StubAPI* stub_api = StubAPI::Global();

  tvm_remote_handle_t pf;
  TVM_LOGD_HT("resolving symbol %s", sym.c_str());
  int rc =
      stub_api->tvm_remote_get_symbol(domain_channel_handle_, module_pointer_,
                                      sym.c_str(), sym.size() + 1, &pf);
  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT("failed to get symbol from CDSP rc=0x%x", rc);
    return nullptr;
  }
  void* addr = reinterpret_cast<void*>(pf);
  TVM_LOGD_HT("resolved %s -> %p", sym.c_str(), addr);
  return addr;
}

void HexagonTarget::Call(void* func, uint32_t* scalar, unsigned scalar_num,
                         uint32_t* stack, unsigned stack_num) {
  uint64 pcycles = 0, execution_time_usec = 0;
  auto scalar_octet =
      std::unique_ptr<tvm_remote_buffer[]>(new tvm_remote_buffer[scalar_num]);
  auto stack_octet =
      std::unique_ptr<tvm_remote_buffer[]>(new tvm_remote_buffer[stack_num]);
  TVM_LOGD_HT("scalars=%p, stack=%p", scalar, stack);

  if (scalar_octet == nullptr || stack_octet == nullptr) {
    TVM_LOGE_HT("mem alloc failed for scalar/stack octets");
    return;
  }
  std::memset(scalar_octet.get(), 0, scalar_num * sizeof(tvm_remote_buffer));
  std::memset(stack_octet.get(), 0, stack_num * sizeof(tvm_remote_buffer));

  auto ProcessInputs = [this](uint32_t* inputs, tvm_remote_buffer* buffers,
                              unsigned num) {
    for (unsigned i = 0; i != num; ++i) {
      void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(inputs[i]));
      auto aa = GetAppsAddr(ptr, false);
      if (aa.first == vtcm_mark_) {
        buffers[i].data = nullptr;
        buffers[i].dataLen = 0;
      } else if (aa.first) {
        buffers[i].data = static_cast<unsigned char*>(aa.first);
        buffers[i].dataLen = aa.second;
      }
    }
  };

  ProcessInputs(scalar, scalar_octet.get(), scalar_num);
  ProcessInputs(stack, stack_octet.get(), stack_num);

  auto ToString = [](const char* title, uint32_t* data, unsigned num) {
    std::ostringstream log;
    log << "  " << title << ':' << num << " {" << std::hex;
    for (unsigned i = 0; i != num; ++i) log << ' ' << data[i];
    log << " }";
    return log.str();
  };

  TVM_LOGD_HT("%s", ToString("scalars", scalar, scalar_num).c_str());
  TVM_LOGD_HT("%s", ToString("  stack", stack, stack_num).c_str());

  const StubAPI* stub_api = StubAPI::Global();
  int rc = stub_api->tvm_remote_kernel(
      domain_channel_handle_, module_pointer_,
      static_cast<tvm_remote_handle_t>(reinterpret_cast<uintptr_t>(func)),
      reinterpret_cast<int*>(scalar), scalar_num,
      reinterpret_cast<int*>(stack), stack_num, scalar_octet.get(), scalar_num,
      scalar_octet.get(), scalar_num, stack_octet.get(), stack_num,
      stack_octet.get(), stack_num, &pcycles, &execution_time_usec);

  if (rc != AEE_SUCCESS) {
    TVM_LOGE_HT("failed to run kernel on CDSP rc=0x%x", rc);
  } else {
    TVM_LOGD_HT("kernel execution: %llu pcycles, %llu usec, scalar_num=%d",
                pcycles, execution_time_usec, scalar_num);
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // #ifdef __ANDROID__
