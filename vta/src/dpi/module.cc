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

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <vta/dpi/module.h>
#include <vta/dpi/tsim.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>

namespace vta {
namespace dpi {

using namespace tvm::runtime;

typedef void * DeviceHandle;

struct HostRequest {
  uint8_t opcode;
  uint8_t addr;
  uint32_t value;
};

struct HostResponse {
  uint32_t value;
};

struct MemResponse {
  uint8_t valid;
  uint64_t value;
};

template <typename T>
class ThreadSafeQueue {
 public:
  void Push(const T item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    cond_.notify_one();
  }

  void WaitPop(T *item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]{return !queue_.empty();});
    *item = std::move(queue_.front());
    queue_.pop();
  }

  bool TryPop(T *item, bool pop) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    *item = std::move(queue_.front());
    if (pop) queue_.pop();
    return true;
  }

 private:
  mutable std::mutex mutex_;
  std::queue<T> queue_;
  std::condition_variable cond_;
};

class HostDevice {
 public:
  void PushRequest(uint8_t opcode, uint8_t addr, uint32_t value);
  bool TryPopRequest(HostRequest *r, bool pop);
  void PushResponse(uint32_t value);
  void WaitPopResponse(HostResponse *r);
  void Exit();
  uint8_t GetExitStatus();

 private:
  uint8_t exit_{0};
  mutable std::mutex mutex_;
  ThreadSafeQueue<HostRequest> req_;
  ThreadSafeQueue<HostResponse> resp_;
};

class MemDevice {
 public:
  void SetRequest(uint8_t opcode, uint64_t addr, uint32_t len);
  MemResponse ReadData(uint8_t ready);
  void WriteData(uint64_t value);

 private:
  uint64_t *raddr_{0};
  uint64_t *waddr_{0};
  uint32_t rlen_{0};
  uint32_t wlen_{0};
  std::mutex mutex_;
};

void HostDevice::PushRequest(uint8_t opcode, uint8_t addr, uint32_t value) {
  HostRequest r;
  r.opcode = opcode;
  r.addr = addr;
  r.value = value;
  req_.Push(r);
}

bool HostDevice::TryPopRequest(HostRequest *r, bool pop) {
  r->opcode = 0xad;
  r->addr = 0xad;
  r->value = 0xbad;
  return req_.TryPop(r, pop);
}

void HostDevice::PushResponse(uint32_t value) {
  HostResponse r;
  r.value = value;
  resp_.Push(r);
}

void HostDevice::WaitPopResponse(HostResponse *r) {
  resp_.WaitPop(r);
}

void HostDevice::Exit() {
  std::unique_lock<std::mutex> lock(mutex_);
  exit_ = 1;
}

uint8_t HostDevice::GetExitStatus() {
  std::unique_lock<std::mutex> lock(mutex_);
  return exit_;
}

void MemDevice::SetRequest(uint8_t opcode, uint64_t addr, uint32_t len) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (opcode == 1) {
    wlen_ = len + 1;
    waddr_ = reinterpret_cast<uint64_t*>(addr);
  } else {
    rlen_ = len + 1;
    raddr_ = reinterpret_cast<uint64_t*>(addr);
  }
}

MemResponse MemDevice::ReadData(uint8_t ready) {
  std::lock_guard<std::mutex> lock(mutex_);
  MemResponse r;
  r.valid = rlen_ > 0;
  r.value = rlen_ > 0 ? *raddr_ : 0xdeadbeefdeadbeef;
  if (ready == 1 && rlen_ > 0) {
    raddr_++;
    rlen_ -= 1;
  }
  return r;
}

void MemDevice::WriteData(uint64_t value) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (wlen_ > 0) {
    *waddr_ = value;
    waddr_++;
    wlen_ -= 1;
  }
}

class DPIModule final : public DPIModuleNode {
 public:
  ~DPIModule() {
    if (lib_handle_) Unload();
  }

  const char* type_key() const final {
    return "vta-tsim";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    if (name == "WriteReg") {
      return TypedPackedFunc<void(int, int)>(
          [this](int addr, int value){
            this->WriteReg(addr, value);
          });
    } else {
      LOG(FATAL) << "Member " << name << "does not exists";
      return nullptr;
    }
  }

  void Init(const std::string& name) {
    Load(name);
    VTADPIInitFunc finit =  reinterpret_cast<VTADPIInitFunc>(
        GetSymbol("VTADPIInit"));
    CHECK(finit != nullptr);
    finit(this, VTAHostDPI, VTAMemDPI);
    fvsim_ = reinterpret_cast<VTADPISimFunc>(GetSymbol("VTADPISim"));
    CHECK(fvsim_ != nullptr);
  }

  void Launch(uint64_t max_cycles) {
    auto frun = [this, max_cycles]() {
      (*fvsim_)(max_cycles);
    };
    vsim_thread_ = std::thread(frun);
  }

  void WriteReg(int addr, uint32_t value) {
    host_device_.PushRequest(1, addr, value);
  }

  uint32_t ReadReg(int addr) {
    uint32_t value;
    HostResponse *r = new HostResponse;
    host_device_.PushRequest(0, addr, 0);
    host_device_.WaitPopResponse(r);
    value = r->value;
    delete r;
    return value;
  }

  void Finish(uint32_t length) {
    host_device_.Exit();
    vsim_thread_.join();
  }

 protected:
  VTADPISimFunc fvsim_;
  HostDevice host_device_;
  MemDevice mem_device_;
  std::thread vsim_thread_;

  void HostDPI(unsigned char *exit,
               unsigned char *req_valid,
               unsigned char *req_opcode,
               unsigned char *req_addr,
               unsigned int *req_value,
               unsigned char req_deq,
               unsigned char resp_valid,
               unsigned int resp_value) {
    HostRequest *r = new HostRequest;
    *exit = host_device_.GetExitStatus();
    *req_valid = host_device_.TryPopRequest(r, req_deq);
    *req_opcode = r->opcode;
    *req_addr = r->addr;
    *req_value = r->value;
    if (resp_valid) {
      host_device_.PushResponse(resp_value);
    }
    delete r;
  }

  void MemDPI(
      unsigned char req_valid,
      unsigned char req_opcode,
      unsigned char req_len,
      unsigned long long req_addr,
      unsigned char wr_valid,
      unsigned long long wr_value,
      unsigned char *rd_valid,
      unsigned long long *rd_value,
      unsigned char rd_ready) {
    MemResponse r = mem_device_.ReadData(rd_ready);
    *rd_valid = r.valid;
    *rd_value = r.value;
    if (wr_valid) {
      mem_device_.WriteData(wr_value);
    }
    if (req_valid) {
      mem_device_.SetRequest(req_opcode, req_addr, req_len);
    }
  }

  static void VTAHostDPI(
      VTAContextHandle self,
      unsigned char *exit,
      unsigned char *req_valid,
      unsigned char *req_opcode,
      unsigned char *req_addr,
      unsigned int *req_value,
      unsigned char req_deq,
      unsigned char resp_valid,
      unsigned int resp_value) {
    static_cast<DPIModule*>(self)->HostDPI(
        exit, req_valid, req_opcode, req_addr,
        req_value, req_deq, resp_valid, resp_value);
  }

  static void VTAMemDPI(
    VTAContextHandle self,
    unsigned char req_valid,
    unsigned char req_opcode,
    unsigned char req_len,
    unsigned long long req_addr,
    unsigned char wr_valid,
    unsigned long long wr_value,
    unsigned char *rd_valid,
    unsigned long long *rd_value,
    unsigned char rd_ready) {
    static_cast<DPIModule*>(self)->MemDPI(
        req_valid, req_opcode, req_len,
        req_addr, wr_valid, wr_value,
        rd_valid, rd_value, rd_ready);
  }

 private:
  // Platform dependent handling.
#if defined(_WIN32)
  // library handle
  HMODULE lib_handle_{nullptr};
  // Load the library
  void Load(const std::string& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name;
  }
  void* GetSymbol(const char* name) {
    return reinterpret_cast<void*>(
        GetProcAddress(lib_handle_, (LPCSTR)name)); // NOLINT(*)
  }
  void Unload() {
    FreeLibrary(lib_handle_);
  }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const std::string& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name
        << " " << dlerror();
  }
  void* GetSymbol(const char* name) {
    return dlsym(lib_handle_, name);
  }
  void Unload() {
    dlclose(lib_handle_);
  }
#endif
};

Module DPIModuleNode::Load(std::string dll_name) {
  std::shared_ptr<DPIModule> n =
      std::make_shared<DPIModule>();
  n->Init(dll_name);
  return Module(n);
}

TVM_REGISTER_GLOBAL("module.loadfile_vta-tsim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = DPIModuleNode::Load(args[0]);
  });
}  // namespace dpi
}  // namespace vta
