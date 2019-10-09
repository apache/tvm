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
 *  Copyright (c) 2019 by Contributors
 * \file src/runtime/vm/profiler/vm.cc
 * \brief The Relay debug virtual machine.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "vm.h"

namespace tvm {
namespace runtime {
namespace vm {

PackedFunc VirtualMachineDebug::GetFunction(
    const std::string& name, const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "get_stat") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      double total_duration = 0.0;
      std::ostringstream os;
      os << std::setw(30) << std::left << "#OpName"
         << "\t" << std::setw(10) << std::left << "#InvokeCount"
         << "\t"
         << "#Duration(us): Sum/Mean/Min/Max" << std::endl;

      for (auto kv : op_durations) {
        auto vals = op_durations[kv.first];
        auto sum = std::accumulate(vals.begin(), vals.end(), 0.0);;
        auto mean = sum / static_cast<double>(vals.size());
        auto min_value = *std::min_element(vals.begin(), vals.end());
        auto max_value = *std::max_element(vals.begin(), vals.end());

        os << std::setw(30) << std::left << packed_index_map[kv.first] << "\t"
           << std::setw(10) << std::left << op_invokes[kv.first] << "\t"
           <<  sum << "/" << mean << "/" << min_value << "/" << max_value << std::endl;

        total_duration += sum;
      }
      os << "Total Duration " << total_duration << " us" << std::endl;
      *rv = os.str();
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

void VirtualMachineDebug::Init(const Executable* exec) {
  VirtualMachine::Init(exec);
  CHECK(this->exec);
  for (auto kv : this->exec->primitive_map) {
    packed_index_map[kv.second] = kv.first;
    op_invokes[kv.second] = 0;
  }
}

void VirtualMachineDebug::InvokePacked(Index packed_index,
                                       const PackedFunc& func, Index arg_count,
                                       Index output_size,
                                       const std::vector<ObjectRef>& args) {
  CHECK(this->exec);
  auto ctx = this->exec->GetParamsContext();
  // warmup
  VirtualMachine::InvokePacked(packed_index, func, arg_count, output_size,
                               args);
  TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);

  auto op_begin = std::chrono::high_resolution_clock::now();
  VirtualMachine::InvokePacked(packed_index, func, arg_count, output_size,
                               args);
  TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
  auto op_end = std::chrono::high_resolution_clock::now();
  double op_duration =
      std::chrono::duration_cast<std::chrono::duration<double> >(op_end -
                                                                 op_begin)
          .count();

  op_durations[packed_index].push_back(op_duration * 1e6);
  op_invokes[packed_index] += 1;
}

runtime::Module CreateVirtualMachineDebug(const Executable* exec) {
  std::shared_ptr<VirtualMachineDebug> vm = std::make_shared<VirtualMachineDebug>();
  vm->Init(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("relay._vm._VirtualMachineDebug")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec) << "Virtual machine has not been defined yet."
              << "\n";
  *rv = CreateVirtualMachineDebug(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
