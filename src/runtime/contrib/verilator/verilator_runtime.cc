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
 * \file src/runtime/contrib/verilator/verilator_runtime.cc
 * \brief A runtime for Verilator.
 */

#include "verilator_runtime.h"

#include <dlfcn.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../../library_module.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "verilator_device.h"
#include "verilator_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::contrib;
using namespace tvm::runtime::json;

VerilatorLibrary::~VerilatorLibrary() {
  if (lib_handle_) {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
}

void VerilatorLibrary::Load(const std::string& name) {
  lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  ICHECK(lib_handle_ != nullptr) << "Failed to load dynamic shared library " << name << " "
                                 << dlerror();
}

void* VerilatorLibrary::GetSymbol(const char* name) { return dlsym(lib_handle_, name); }

void VerilatorProfiler::Clear() { cycle_counter = 0; }

std::string VerilatorProfiler::AsJSON() {
  std::ostringstream os;
  os << "{\n"
     << " \"cycle_counter\":" << cycle_counter << "\n"
     << "}\n";
  return os.str();
}

VerilatorProfiler* VerilatorProfiler::ThreadLocal() {
  static thread_local VerilatorProfiler inst;
  return &inst;
}

VerilatorRuntime::~VerilatorRuntime() {
  VLOG(0) << "destroying verilator runtime";
  if (lib_ == nullptr) {
    // Never initialized. This can happen if the runtime was created during compilation of
    // a BYOC function but the resulting runtime module was never invoked.
    return;
  }
  auto dealloc = reinterpret_cast<VerilatorDeallocFunc>(lib_->GetSymbol("VerilatorDealloc"));
  ICHECK(dealloc != nullptr);
  ICHECK(device_ != nullptr);
  dealloc(device_);
  device_ = nullptr;
  lib_->~VerilatorLibrary();
  lib_ = nullptr;
}

void VerilatorRuntime::SetLibrary(const std::string& lib_path) { lib_path_ = lib_path; }

void VerilatorRuntime::SetResetCycles(const int cycles) { reset_cycles_ = cycles; }

void VerilatorRuntime::EnableProfiler() { prof_enable_ = true; }

void VerilatorRuntime::SetProfilerCycleCounterId(const int id) { prof_cycle_counter_id_ = id; }

void VerilatorRuntime::Init(const Array<NDArray>& consts) {
  VLOG(0) << "initializing verilator runtime";
  lib_ = new VerilatorLibrary();
  lib_->Load(lib_path_);
  auto alloc = reinterpret_cast<VerilatorAllocFunc>(lib_->GetSymbol("VerilatorAlloc"));
  ICHECK(alloc != nullptr);
  auto reset = reinterpret_cast<VerilatorResetFunc>(lib_->GetSymbol("VerilatorReset"));
  ICHECK(reset != nullptr);
  read_ = reinterpret_cast<VerilatorReadFunc>(lib_->GetSymbol("VerilatorRead"));
  ICHECK(read_ != nullptr);

  // alloc verilator device
  device_ = alloc();

  // enable profiler
  if (prof_enable_) prof_ = VerilatorProfiler::ThreadLocal();

  // reset verilator device
  reset(device_, reset_cycles_);

  CHECK_EQ(consts.size(), const_idx_.size())
      << "The number of input constants must match the number of required.";

  // Setup constants entries for weights.
  SetupConstants(consts);
}

void VerilatorRuntime::Run() {
  std::vector<int*> in_ptr;
  std::vector<int*> out_ptr;
  for (size_t i = 0; i < input_nodes_.size(); ++i) {
    uint32_t eid = EntryID(input_nodes_[i], 0);
    int* data = static_cast<int*>(data_entry_[eid]->data);
    in_ptr.push_back(data);
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    uint32_t eid = EntryID(outputs_[i]);
    int* data = static_cast<int*>(data_entry_[eid]->data);
    out_ptr.push_back(data);
  }
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    const auto& node = nodes_[nid];
    if (node.GetOpType() == "kernel") {
      CHECK_EQ(node.GetOpType(), "kernel");
      auto op_name = node.GetOpName();
      auto entry = node.GetInputs()[0];
      auto shape = node.GetOpShape()[entry.index_];
      if ("add" == op_name) {
        auto add = reinterpret_cast<VerilatorAddFunc>(lib_->GetSymbol("verilator_add"));
        ICHECK(add != nullptr);
        add(device_, in_ptr[0], in_ptr[1], out_ptr[0], shape[0], shape[1]);
      } else if ("nn.bias_add" == op_name) {
        auto bias_add =
            reinterpret_cast<VerilatorBiasAddFunc>(lib_->GetSymbol("verilator_bias_add"));
        ICHECK(bias_add != nullptr);
        bias_add(device_, in_ptr[0], in_ptr[1], out_ptr[0], shape[0], shape[3], shape[1], shape[2]);
      } else {
        LOG(FATAL) << "Unsupported op: " << op_name;
      }
    }
  }
  if (prof_enable_) {
    int cycles = read_(device_, prof_cycle_counter_id_, 0);
    prof_->cycle_counter += cycles;
  }
}

TVM_REGISTER_GLOBAL("verilator.profiler_clear").set_body([](TVMArgs args, TVMRetValue* rv) {
  VerilatorProfiler::ThreadLocal()->Clear();
});

TVM_REGISTER_GLOBAL("verilator.profiler_status").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = VerilatorProfiler::ThreadLocal()->AsJSON();
});

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
