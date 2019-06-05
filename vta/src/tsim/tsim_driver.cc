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

#include <vta/driver.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <vta/dpi/module.h>

namespace vta {
namespace tsim {

using vta::dpi::DPIModuleNode;
using tvm::runtime::Module;

class DPILoader {
 public:
  void Init(Module module) {
    mod_ = module;
  }

  DPIModuleNode* Get() {
    return static_cast<DPIModuleNode*>(mod_.operator->());
  }

  static DPILoader* Global() {
    static DPILoader inst;
    return &inst;
  }

  Module mod_;
};

class Device {
 public:
  Device() {
    dpi_ = DPILoader::Global();
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          vta_phy_addr_t uop_phy_addr,
          vta_phy_addr_t inp_phy_addr,
          vta_phy_addr_t wgt_phy_addr,
          vta_phy_addr_t acc_phy_addr,
          vta_phy_addr_t out_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    this->Init();
    this->Launch(insn_phy_addr,
                 uop_phy_addr,
                 inp_phy_addr,
                 wgt_phy_addr,
                 acc_phy_addr,
                 out_phy_addr,
                 insn_count,
                 wait_cycles);
    this->WaitForCompletion(wait_cycles);
    dev_->Finish();
    return 0;
  }

 private:
  void Init() {
    dev_ = dpi_->Get();
  }

  void Launch(vta_phy_addr_t insn_phy_addr,
              vta_phy_addr_t uop_phy_addr,
              vta_phy_addr_t inp_phy_addr,
              vta_phy_addr_t wgt_phy_addr,
              vta_phy_addr_t acc_phy_addr,
              vta_phy_addr_t out_phy_addr,
              uint32_t insn_count,
              uint32_t wait_cycles) {
    // launch simulation thread
    dev_->Launch(wait_cycles);
    dev_->WriteReg(0x10, insn_count);
    dev_->WriteReg(0x14, insn_phy_addr);
    dev_->WriteReg(0x18, insn_phy_addr >> 32);
    dev_->WriteReg(0x1c, 0);
    dev_->WriteReg(0x20, uop_phy_addr >> 32);
    dev_->WriteReg(0x24, 0);
    dev_->WriteReg(0x28, inp_phy_addr >> 32);
    dev_->WriteReg(0x2c, 0);
    dev_->WriteReg(0x30, wgt_phy_addr >> 32);
    dev_->WriteReg(0x34, 0);
    dev_->WriteReg(0x38, acc_phy_addr >> 32);
    dev_->WriteReg(0x3c, 0);
    dev_->WriteReg(0x40, out_phy_addr >> 32);
    // start
    dev_->WriteReg(0x00, 0x1);
  }

  void WaitForCompletion(uint32_t wait_cycles) {
    uint32_t i, val;
    for (i = 0; i < wait_cycles; i++) {
      val = dev_->ReadReg(0x00);
      val &= 0x2;
      if (val == 0x2) break;  // finish
    }
  }

  DPILoader* dpi_;
  DPIModuleNode* dev_;
};

using tvm::runtime::TVMRetValue;
using tvm::runtime::TVMArgs;

TVM_REGISTER_GLOBAL("tvm.vta.tsim.init")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    DPILoader::Global()->Init(m);
  });

}  // namespace tsim
}  // namespace vta

void* VTAMemAlloc(size_t size, int cached) {
  void *p = malloc(size);
  return p;
}

void VTAMemFree(void* buf) {
  free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return reinterpret_cast<uint64_t>(reinterpret_cast<uint64_t*>(buf));
}

void VTAFlushCache(vta_phy_addr_t buf, int size) {
}

void VTAInvalidateCache(vta_phy_addr_t buf, int size) {
}

VTADeviceHandle VTADeviceAlloc() {
  return new vta::tsim::Device();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<vta::tsim::Device*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 vta_phy_addr_t uop_phy_addr,
                 vta_phy_addr_t inp_phy_addr,
                 vta_phy_addr_t wgt_phy_addr,
                 vta_phy_addr_t acc_phy_addr,
                 vta_phy_addr_t out_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<vta::tsim::Device*>(handle)->Run(
      insn_phy_addr,
      uop_phy_addr,
      inp_phy_addr,
      wgt_phy_addr,
      acc_phy_addr,
      out_phy_addr,
      insn_count,
      wait_cycles);
}
