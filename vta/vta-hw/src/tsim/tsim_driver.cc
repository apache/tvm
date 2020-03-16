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
#include <tvm/runtime/registry.h>
#include <vta/driver.h>
#include <vta/dpi/module.h>

#include "../vmem/virtual_memory.h"

namespace vta {
namespace tsim {

using tvm::runtime::Module;
using vta::dpi::DPIModuleNode;

class Profiler {
 public:
  Profiler() {
    counters_ = new int[num_counters_];
    this->ClearAll();
  }

  ~Profiler() {
    delete [] counters_;
  }

  /*! \brief update one event counter */
  void Update(uint32_t idx, uint32_t value) {
    counters_[idx] += value;
  }

  /*! \brief clear one event counter*/
  void Clear(uint32_t idx) {
    counters_[idx] = 0;
  }

  /*! \brief clear all event counters */
  void ClearAll() {
    for (uint32_t i = 0; i < num_counters_; i++) {
      counters_[i] = 0;
    }
  }

  /*! \brief return counters as json */
  std::string AsJSON() {
    std::ostringstream os;
    os << "{\n"
       << " \"cycle_count\":" << counters_[0] << "\n"
       <<"}\n";
    return os.str();
  }

  static Profiler* Global() {
    static Profiler inst;
    return &inst;
  }

 private:
  /*! \brief total number of event counters */
  uint32_t num_counters_{1};
  /*! \brief event counters */
  int* counters_{nullptr};
};

class DPILoader {
 public:
  ~DPILoader() {
    dpi_->SimResume();
    dpi_->SimFinish();
  }

  void Init(Module module) {
    mod_ = module;
    dpi_ = this->Get();
    dpi_->SimLaunch();
    dpi_->SimWait();
  }

  DPIModuleNode* Get() {
    return static_cast<DPIModuleNode*>(mod_.operator->());
  }

  static DPILoader* Global() {
    static DPILoader inst;
    return &inst;
  }

  // TVM module
  Module mod_;
  // DPI Module
  DPIModuleNode* dpi_{nullptr};
};

class Device {
 public:
  Device() {
    loader_ = DPILoader::Global();
    prof_ = Profiler::Global();
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    this->Init();
    this->Launch(insn_phy_addr,
                 insn_count,
                 wait_cycles);
    this->WaitForCompletion(wait_cycles);
    return 0;
  }

 private:
  void Init() {
    dpi_ = loader_->Get();
    dpi_->SimResume();
  }

  void Launch(vta_phy_addr_t insn_phy_addr,
              uint32_t insn_count,
              uint32_t wait_cycles) {
    dpi_->WriteReg(0x08, insn_count);
    dpi_->WriteReg(0x0c, insn_phy_addr);
    dpi_->WriteReg(0x10, 0);
    dpi_->WriteReg(0x14, 0);
    dpi_->WriteReg(0x18, 0);
    dpi_->WriteReg(0x1c, 0);
    dpi_->WriteReg(0x20, 0);
    // start
    dpi_->WriteReg(0x00, 0x1);
  }

  void WaitForCompletion(uint32_t wait_cycles) {
    uint32_t i, val;
    for (i = 0; i < wait_cycles; i++) {
      val = dpi_->ReadReg(0x00);
      val &= 0x2;
      if (val == 0x2) break;  // finish
    }
    prof_->Update(0, dpi_->ReadReg(0x04));
    dpi_->SimWait();
  }

  // Profiler
  Profiler* prof_;
  // DPI loader
  DPILoader* loader_;
  // DPI Module
  DPIModuleNode* dpi_;
};

using tvm::runtime::TVMRetValue;
using tvm::runtime::TVMArgs;

TVM_REGISTER_GLOBAL("vta.tsim.init")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    DPILoader::Global()->Init(m);
  });

TVM_REGISTER_GLOBAL("vta.tsim.profiler_clear")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Profiler::Global()->ClearAll();
  });

TVM_REGISTER_GLOBAL("vta.tsim.profiler_status")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = Profiler::Global()->AsJSON();
  });

}  // namespace tsim
}  // namespace vta

void* VTAMemAlloc(size_t size, int cached) {
  return vta::vmem::VirtualMemoryManager::Global()->Alloc(size);
}

void VTAMemFree(void* buf) {
  vta::vmem::VirtualMemoryManager::Global()->Free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return vta::vmem::VirtualMemoryManager::Global()->GetPhyAddr(buf);
}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
}

VTADeviceHandle VTADeviceAlloc() {
  return new vta::tsim::Device();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<vta::tsim::Device*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<vta::tsim::Device*>(handle)->Run(
      insn_phy_addr,
      insn_count,
      wait_cycles);
}
