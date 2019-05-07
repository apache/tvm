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

class TestDriver {
 public:
  TestDriver(Module module)
      : module_(module) {
    dpi_ = static_cast<DPIModuleNode*>(
        module.operator->());
  }

  int Run(uint32_t length, void* inp, void* out) {
    uint32_t wait_cycles = 100000000;
    this->Launch(wait_cycles, length, inp, out);
    this->WaitForCompletion(wait_cycles);
    dpi_->Finish();
    return 0;
  }

 private:
  void Launch(uint32_t wait_cycles, uint32_t length, void* inp, void* out) {
    dpi_->Launch(wait_cycles);
    // write registers
    dpi_->WriteReg(0x04, length);
    dpi_->WriteReg(0x08, get_half_addr(inp, false));
    dpi_->WriteReg(0x0c, get_half_addr(inp, true));
    dpi_->WriteReg(0x10, get_half_addr(out, false));
    dpi_->WriteReg(0x14, get_half_addr(out, true));
    dpi_->WriteReg(0x00, 0x1); // launch
  }

  void WaitForCompletion(uint32_t wait_cycles) {
    uint32_t i, val;
    for (i = 0; i < wait_cycles; i++) {
      val = dpi_->ReadReg(0x00);
      if (val == 2) break; // finish
    }
  }

 private:
  DPIModuleNode* dpi_;
  Module module_;
};

using tvm::runtime::TVMRetValue;
using tvm::runtime::TVMArgs;

TVM_REGISTER_GLOBAL("tvm.vta.driver")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module dev_mod = args[0];
    DLTensor* A = args[1];
    DLTensor* B = args[2];
    TestDriver dev_(dev_mod);
    dev_.Run(A->shape[0], A->data, B->data);
  });

}  // namespace driver
}  // namespace vta
