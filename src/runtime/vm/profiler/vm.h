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
 * \file src/runtime/vm/profiler/vm.h
 * \brief The Relay debug virtual machine.
 */

#ifndef TVM_RUNTIME_VM_PROFILER_VM_H_
#define TVM_RUNTIME_VM_PROFILER_VM_H_

#include <tvm/runtime/vm/vm.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

class VirtualMachineDebug : public VirtualMachine {
 public:
  VirtualMachineDebug() : VirtualMachine() {}

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void LoadExecutable(const Executable* exec) final;

  ~VirtualMachineDebug() {}

 private:
  void InvokePacked(Index packed_index, const PackedFunc& func, Index arg_count, Index output_size,
                    const std::vector<ObjectRef>& args) final;

  std::unordered_map<Index, std::string> packed_index_map_;
  std::unordered_map<Index, std::vector<double>> op_durations_;
  std::unordered_map<Index, int> op_invokes_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_PROFILER_VM_H_
