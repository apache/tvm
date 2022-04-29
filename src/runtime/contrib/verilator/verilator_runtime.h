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

#ifndef TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_RUNTIME_H_

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
typedef void (*VerilatorDeallocFunc)(VerilatorHandle);
typedef void (*VerilatorResetFunc)(VerilatorHandle, int);
typedef int (*VerilatorReadFunc)(VerilatorHandle, int, int);
typedef void (*VerilatorAddFunc)(VerilatorHandle, int*, int*, int*, int, int);
typedef void (*VerilatorBiasAddFunc)(VerilatorHandle, int*, int*, int*, int, int, int, int);

class VerilatorLibrary : public Library {
 public:
  ~VerilatorLibrary();

  /*! \brief load library */
  void Load(const std::string& name);

  /*! \brief get symbol from libray */
  void* GetSymbol(const char* name) final;

 private:
  /*! \brief the library handle */
  void* lib_handle_{nullptr};
};

class VerilatorProfiler {
 public:
  /*! \brief the number of cycle counter */
  uint32_t cycle_counter{0};

  /*! \brief clear the profiler */
  void Clear();

  /*! \brief get profiler data */
  std::string AsJSON();

  /*! \brief profiler constructor */
  static VerilatorProfiler* ThreadLocal();
};

class VerilatorRuntime : public JSONRuntimeBase {
 public:
  VerilatorRuntime(const std::string& symbol_name, const std::string& graph_json,
                   const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {
    VLOG(0) << "creating verilator runtime";
  }

  ~VerilatorRuntime();

  const char* type_key() const final { return "verilator"; }

  /*! \brief set verilator library */
  void SetLibrary(const std::string& lib_name);

  /*! \brief set the number of reset cycles */
  void SetResetCycles(const int cycles);

  /*! \brief enable profiler */
  void EnableProfiler();

  /*! \brief set cycle counter register id */
  void SetProfilerCycleCounterId(const int id);

  /*! \brief init verilator runtime */
  void Init(const Array<NDArray>& consts) override;

  /*! \brief run verilator runtime */
  void Run() override;

 private:
  /*! \brief the verilator library path */
  String lib_path_;
  /*! \brief the verilator device */
  VerilatorHandle device_{nullptr};
  /*! \brief the verilator library */
  VerilatorLibrary* lib_{nullptr};
  /*! \brief the verilator profiler */
  VerilatorProfiler* prof_{nullptr};
  /*! \brief the verilator read function */
  VerilatorReadFunc read_{nullptr};
  /*! \brief the verilator reset cycles */
  int reset_cycles_{1};
  /*! \brief the verilator profiler status */
  bool prof_enable_{false};
  /*! \brief the verilator profiler cycle counter id */
  int prof_cycle_counter_id_{0};
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_RUNTIME_H_
