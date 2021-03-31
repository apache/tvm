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
 * \file src/runtime/vm/profiler/vm.cc
 * \brief The Relay debug virtual machine.
 */

#include "vm.h"

#include <tvm/runtime/registry.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

PackedFunc VirtualMachineDebug::GetFunction(const std::string& name,
                                            const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_stat") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 1U);
      std::vector<std::pair<Index, double>> op_acc_time;
      std::unordered_map<Index, std::vector<double>> op_durations;
      for (auto kv : op_timers_) {
        std::vector<double> durations_us;
        for (auto t : kv.second) {
          durations_us.push_back(t->SyncAndGetElapsedNanos() / 1e3);
        }
        op_durations[kv.first] = durations_us;
      }
      for (auto kv : op_durations) {
        auto val =
            std::make_pair(kv.first, std::accumulate(kv.second.begin(), kv.second.end(), 0.0));
        op_acc_time.push_back(val);
      }
      bool sort_by_time = args[0];
      if (sort_by_time) {
        auto comp = [](const std::pair<Index, double>& lhs, const std::pair<Index, double>& rhs) {
          return lhs.second > rhs.second;
        };
        std::sort(op_acc_time.begin(), op_acc_time.end(), comp);
      }
      double total_duration = 0.0;
      int64_t total_packed_funcs = 0;
      std::ostringstream os;
      os << std::setw(30) << std::left << "#OpName"
         << "\t" << std::setw(10) << std::left << "#InvokeCount"
         << "\t"
         << "#Duration(us): Sum/Mean/Min/Max" << std::endl;

      for (auto kv : op_acc_time) {
        auto vals = op_durations[kv.first];
        auto sum = kv.second;
        auto mean = sum / static_cast<double>(vals.size());
        auto min_value = *std::min_element(vals.begin(), vals.end());
        auto max_value = *std::max_element(vals.begin(), vals.end());

        os << std::setw(30) << std::left << packed_index_map_[kv.first] << "\t" << std::setw(10)
           << std::left << op_invokes_[kv.first] << "\t" << sum << "/" << mean << "/" << min_value
           << "/" << max_value << std::endl;

        total_duration += sum;
        total_packed_funcs += op_invokes_[kv.first];
      }
      os << "\nTotal Duration: " << total_duration << " us.\t"
         << "Total Packed Functions: " << total_packed_funcs << std::endl;
      *rv = os.str();
    });
  } else if (name == "reset") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      op_timers_.clear();
      op_invokes_.clear();
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

void VirtualMachineDebug::LoadExecutable(const Executable* exec) {
  VirtualMachine::LoadExecutable(exec);
  ICHECK(exec_);
  for (auto kv : exec_->primitive_map) {
    packed_index_map_[kv.second] = kv.first;
    op_invokes_[kv.second] = 0;
  }
}

void VirtualMachineDebug::InvokePacked(Index packed_index, const PackedFunc& func, Index arg_count,
                                       Index output_size, const std::vector<ObjectRef>& args) {
  ICHECK(exec_);
  ICHECK(!devices_.empty()) << "Device has not been initialized yet.";
  // The device of any input of the operator is used for synchronization.
  ICHECK_GT(arg_count, 0U);
  ObjectRef arg = args[0];
  while (arg->IsInstance<ADTObj>()) {
    ADT adt = Downcast<ADT>(arg);
    arg = adt[0];
  }
  ICHECK(arg->IsInstance<NDArray::ContainerType>());
  auto nd_array = Downcast<NDArray>(arg);
  auto dev = nd_array->device;

  Timer t = Timer::Start(dev);
  VirtualMachine::InvokePacked(packed_index, func, arg_count, output_size, args);
  t->Stop();

  op_timers_[packed_index].push_back(t);
  op_invokes_[packed_index] += 1;
}

runtime::Module CreateVirtualMachineDebug(const Executable* exec) {
  auto vm = make_object<VirtualMachineDebug>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._VirtualMachineDebug").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec) << "Virtual machine has not been defined yet."
               << "\n";
  *rv = CreateVirtualMachineDebug(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
