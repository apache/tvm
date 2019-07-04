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
#include <vta/dpi/module.h>
#include <unistd.h>

namespace vta {
namespace driver {

uint32_t get_half_addr(void *p, bool upper) {
  if (upper) {
    return ((uint64_t) ((uint64_t*) p)) >> 32;
  } else {
    return ((uint64_t) ((uint64_t*) p));
  }
}

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

  uint32_t Run(uint32_t c, uint32_t length, void* inp, void* out) {
    uint32_t cycles;
    this->Init();
    this->Launch(c, length, inp, out);
    cycles = this->WaitForCompletion();
    dev_->Wait();
    sleep(1);
    dev_->Resume();
    dev_->Finish();
    return cycles;
  }

 private:
  void Init() {
    dev_ = dpi_->Get();
  }

  void Launch(uint32_t c, uint32_t length, void* inp, void* out) {
    dev_->Launch(wait_cycles_);
    dev_->WriteReg(0x08, c);
    dev_->WriteReg(0x0c, length);
    dev_->WriteReg(0x10, get_half_addr(inp, false));
    dev_->WriteReg(0x14, get_half_addr(inp, true));
    dev_->WriteReg(0x18, get_half_addr(out, false));
    dev_->WriteReg(0x1c, get_half_addr(out, true));
    dev_->WriteReg(0x00, 0x1); // launch
  }

  uint32_t WaitForCompletion() {
    uint32_t i, val;
    for (i = 0; i < wait_cycles_; i++) {
      val = dev_->ReadReg(0x00);
      if (val == 2) break; // finish
    }
    val = dev_->ReadReg(0x04);
    return val;
  }

  // wait cycles
  uint32_t wait_cycles_{100000000};
  // DPI loader
  DPILoader* dpi_;
  // DPI Module
  DPIModuleNode* dev_;
};

using tvm::runtime::TVMRetValue;
using tvm::runtime::TVMArgs;

TVM_REGISTER_GLOBAL("tvm.vta.tsim.init")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    DPILoader::Global()->Init(m);
  });

TVM_REGISTER_GLOBAL("tvm.vta.driver")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    Device dev_;
    uint32_t cycles = dev_.Run(static_cast<int>(args[2]), A->shape[0], A->data, B->data);
    *rv = static_cast<int>(cycles);
  });

}  // namespace driver
}  // namespace vta
