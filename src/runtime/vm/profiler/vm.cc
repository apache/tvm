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

#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

PackedFunc VirtualMachineDebug::GetFunction(const String& name,
                                            const ObjectPtr<Object>& sptr_to_self) {
  if (name == "profile") {
    return TypedPackedFunc<profiling::Report(String, Array<profiling::MetricCollector>)>(
        [sptr_to_self, this](String arg_name, Array<profiling::MetricCollector> collectors) {
          std::vector<Device> devices;
          for (auto dev : devices_) {
            if (dev.device_type > 0) {
              devices.push_back(dev);
            }
          }

          // We cannot send Arrays over rpc, so in order to support profiling
          // on remotes, we accept a nullptr for collectors.
          if (collectors.defined()) {
            std::vector<profiling::MetricCollector> cs(collectors.begin(), collectors.end());
            prof_ = profiling::Profiler(devices, cs, {{String("Executor"), String("VM")}});
          } else {
            prof_ = profiling::Profiler(devices, {}, {{String("Executor"), String("VM")}});
          }

          auto invoke = VirtualMachine::GetFunction("invoke", sptr_to_self);
          // warmup
          for (int i = 0; i < 3; i++) {
            invoke(arg_name);
          }

          prof_.operator*().Start();
          invoke(arg_name);
          prof_.operator*().Stop();
          auto report = prof_.operator*().Report();
          prof_ = std::nullopt;  // releases hardware counters
          return report;
        });
  } else if (name == "profile_rpc") {
    // We cannot return a Report over RPC because TVM RPC mechanism only
    // supports a subset of Object classes. Instead we serialize it on the
    // remote (here) and deserialize it on the other end.
    return TypedPackedFunc<std::string(std::string)>([sptr_to_self, this](std::string arg_name) {
      PackedFunc profile = GetFunction("profile", sptr_to_self);
      profiling::Report report = profile(arg_name, Array<profiling::MetricCollector>());
      return report->AsJSON();
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

void VirtualMachineDebug::LoadExecutable(const ObjectPtr<Executable>& exec) {
  VirtualMachine::LoadExecutable(exec);
  for (auto kv : exec_->primitive_map) {
    packed_index_map_[kv.second] = kv.first;
  }
}

void VirtualMachineDebug::OpStartHook(Instruction instr) {
  if (prof_ && prof_.operator*().IsRunning()) {
    if (instr.op == Opcode::LoadConst) {
      Device dev = GetDevice(exec_->const_device_indexes[instr.const_index]);
      prof_.operator*().StartCall("VM::LoadConst", dev, {});
    } else if (instr.op == Opcode::DeviceCopy) {
      Device dst_dev = GetDevice(instr.device_copy.dst_device_index);
      prof_.operator*().StartCall("VM::DeviceCopy", dst_dev, {});
    } else if (instr.op == Opcode::ReshapeTensor) {
      prof_.operator*().StartCall("VM::ReshapeTensor", devices_[exec_->host_device_index], {});
    } else if (instr.op == Opcode::AllocTensor) {
      auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

      for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
        shape[i] = instr.alloc_tensor.shape[i];
      }
      auto storage_obj = ReadRegister(instr.alloc_tensor.storage);
      auto storage = Downcast<Storage>(storage_obj);
      prof_.operator*().StartCall(
          "VM::AllocTensor", storage->buffer.device,
          {{"Argument Shapes", profiling::ShapeString(shape, instr.alloc_tensor.dtype)}});
    } else if (instr.op == Opcode::AllocTensorReg) {
      auto storage_obj = ReadRegister(instr.alloc_tensor_reg.storage);
      auto storage = Downcast<Storage>(storage_obj);
      Device cpu_dev = GetDevice(exec_->host_device_index);
      auto shape_obj = ReadRegister(instr.alloc_tensor_reg.shape_register);
      NDArray shape_tensor = Downcast<NDArray>(shape_obj).CopyTo(cpu_dev);
      prof_.operator*().StartCall(
          "VM::AllocTensorReg", storage->buffer.device,
          {{"Argument Shapes",
            profiling::ShapeString(shape_tensor, instr.alloc_tensor_reg.dtype)}});
    } else if (instr.op == Opcode::AllocStorage) {
      std::ostringstream shape;
      if (instr.alloc_storage.ndim > 0) {
        std::string shape_str = "[";
        for (uint32_t i = 0; i < instr.alloc_storage.ndim; ++i) {
          if (i > 0) {
            shape_str += ", ";
          }
          shape_str += std::to_string(instr.alloc_storage.shape[i]);
        }
        shape_str += "]";
        shape << DLDataType2String(instr.alloc_storage.dtype_hint) << shape_str;
      } else {
        auto size = LoadScalarInt(instr.alloc_storage.allocation_size);
        shape << DLDataType2String(instr.alloc_storage.dtype_hint) << "[" << size << "]";
      }
      Device dev = GetDevice(instr.alloc_storage.device_index);
      prof_.operator*().StartCall("VM::AllocStorage", dev,
                                  {{"VM::Argument Shapes", String(shape.str())}});
    } else {
      prof_.operator*().StartCall("VM::UnknownOp", GetDevice(exec_->host_device_index), {});
    }
  }
}

void VirtualMachineDebug::OpStopHook() {
  if (prof_ && prof_.operator*().IsRunning()) {
    prof_.operator*().StopCall();
  }
}

void VirtualMachineDebug::InvokePacked(Index packed_index, const PackedFunc& func, Index arg_count,
                                       Index output_size, const std::vector<ObjectRef>& args) {
  ICHECK(exec_);
  ICHECK(!devices_.empty()) << "Device has not been initialized yet.";
  if (prof_ && prof_.operator*().IsRunning()) {
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

    // get argument sizes
    std::vector<NDArray> shapes;
    for (Index i = 0; i < arg_count; i++) {
      if (const auto* obj = args[i].as<ADTObj>()) {
        for (size_t fi = 0; fi < obj->size; ++fi) {
          auto o = (*obj)[fi];
          shapes.push_back(Downcast<NDArray>(o));
        }
      } else {
        shapes.push_back(Downcast<NDArray>(args[i]));
      }
    }

    std::unordered_map<std::string, ObjectRef> metrics;

    ICHECK(exec_->op_attrs.find(packed_index) != exec_->op_attrs.end())
        << packed_index_map_[packed_index] << " not found in op attrs";

    auto& op_attrs = exec_->op_attrs.at(packed_index);
    for (auto p : op_attrs) {
      if (std::string(p.first).find("layout") != std::string::npos) {
        metrics[p.first] = p.second;
      }
    }
    auto it = op_attrs.find("hash");
    if (it != op_attrs.end()) {
      metrics["Hash"] = Downcast<String>((*it).second);
    }
    metrics["Argument Shapes"] = profiling::ShapeString(shapes);

    prof_.operator*().StartCall(packed_index_map_[packed_index], dev, metrics);
  }
  VirtualMachine::InvokePacked(packed_index, func, arg_count, output_size, args);
  if (prof_ && prof_.operator*().IsRunning()) {
    prof_.operator*().StopCall();
  }
}

runtime::Module CreateVirtualMachineDebug(Executable* exec) {
  auto vm = make_object<VirtualMachineDebug>();
  vm->LoadExecutable(GetObjectPtr<Executable>(exec));
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._VirtualMachineDebug").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  auto* exec = dynamic_cast<Executable*>(mod.operator->());
  *rv = CreateVirtualMachineDebug(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
