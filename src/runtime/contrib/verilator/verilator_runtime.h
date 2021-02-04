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
 * \file src/runtime/contrib/verilator/verilator_runtime.h
 * \brief A runtime for Verilator.
 */

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

typedef VerilatorHandle (*VerilatorAllocFunc)();
typedef void (*VerilatorResetFunc)(VerilatorHandle, int);
typedef void (*VerilatorAddFunc)(VerilatorHandle, int*, int*, int*, int, int);

class VerilatorLibrary : public Library {
 public:
  ~VerilatorLibrary() {
    if (lib_handle_) Unload();
  }
  void Init(const std::string& name) { Load(name); }

  void* GetSymbol(const char* name) final { return GetSymbol_(name); }

 private:
  void Load(const std::string& name);

  void* GetSymbol_(const char* name);

  void Unload();

  void* lib_handle_{nullptr};
};

class VerilatorRuntime : public JSONRuntimeBase {
 public:
  VerilatorRuntime(const std::string& symbol_name, const std::string& graph_json,
                       const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "verilator"; }

  void LoadLibrary(const std::string& lib_name);

  void SetResetCycles(const int cycles);

  void EnableProfiler();

  void SetProfilerCycleCounterId(const int id);

  void Init(const Array<NDArray>& consts) override;

  void Run() override;

 private:
  /* Device handle. */
  VerilatorHandle device_{nullptr};
  /* Library handle. */
  VerilatorLibrary* lib_{nullptr};
  /* Add operator. */
  VerilatorAddFunc add_func_{nullptr};
  /* Number of reset cycles. */
  int reset_cycles_{1};
  /* Profiler status. */
  bool prof_enable_{false};
  /* Profiler cycle counter id. */
  int prof_cycle_counter_id_{0};
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
