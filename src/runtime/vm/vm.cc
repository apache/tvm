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
 * \file src/runtime/vm/vm.cc
 * \brief The Relay virtual machine runtime.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/debug.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm/vm.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../file_utils.h"

using namespace tvm::runtime;

namespace tvm {
namespace runtime {
namespace vm {

TVM_REGISTER_OBJECT_TYPE(VMClosureObj);

VMClosure::VMClosure(size_t func_index, std::vector<ObjectRef> free_vars) {
  auto ptr = make_object<VMClosureObj>();
  ptr->func_index = func_index;
  ptr->free_vars = std::move(free_vars);
  data_ = std::move(ptr);
}

void VMFunctionPrint(std::ostream& os, const VMFunction& vm_func) {
  os << vm_func.name << ": " << std::endl;
  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    os << i << ": " << vm_func.instructions[i] << ";" << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, const VMFunction& vm_func) {
  VMFunctionPrint(os, vm_func);
  return os;
}

inline ObjectRef CopyTo(ObjectRef src, const DLDevice& dev, Optional<String> mem_scope = NullOpt) {
  if (src->IsInstance<NDArray::ContainerType>()) {
    auto nd_array = Downcast<NDArray>(src);
    // TODO(mbs): Should respect device id also.
    // TODO(vvchernov): it still does not work for different device id
    // due to simple implementation of Get() and AllocDataSpace() methods
    // see tvm/src/runtime/c_runtime_api.cc: L139
    // tvm/src/runtime/cpu_device_api.cc: L47
    if (nd_array->device.device_type != dev.device_type ||
        nd_array->device.device_id != dev.device_id) {
      VLOG(2) << "copying from " << nd_array->device.device_type << "["
              << nd_array->device.device_id << "] to " << dev.device_type << "[" << dev.device_id
              << "]";
      return nd_array.CopyTo(dev, mem_scope);
    }
    return src;
  } else {
    ICHECK(src->IsInstance<ADTObj>())
        << "VM data must be NDArray or a list of NDArray, but received: " << src->_type_key;
    std::vector<ObjectRef> ret;
    ADT adt = Downcast<ADT>(src);
    for (size_t i = 0; i < adt.size(); i++) {
      ret.push_back(CopyTo(adt[i], dev, mem_scope));
    }
    return ADT(adt->tag, ret.begin(), ret.end());
  }
}

ShapeTuple ToShape(NDArray shape_tensor) {
  std::vector<ShapeTuple::index_type> shape;
  auto rank = shape_tensor.Shape().size();
  auto dtype = shape_tensor.DataType();

  // For 0-rank shapes we need to allocate a single scalar.
  if (rank == 0) {
    return shape;
  }

  // Otherwise we should be rank-1, and we will extract the number of dimensions
  // for the output vector.
  ICHECK_EQ(rank, 1U) << "shape tensor should be a k-length vector, found " << rank;
  int64_t ndim = shape_tensor.Shape().at(0);
  shape.resize(ndim);

  const DLTensor* dl_tensor = shape_tensor.operator->();
  if (dtype.is_int() && dtype.bits() == 32 && dtype.lanes() == 1) {
    int32_t* dims = reinterpret_cast<int32_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else if (dtype.is_int() && dtype.bits() == 64 && dtype.lanes() == 1) {
    int64_t* dims = reinterpret_cast<int64_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else {
    LOG(FATAL) << "invalid shape tensor datatype: " << dtype;
  }

  return ShapeTuple(shape);
}

void VirtualMachine::OpStartHook(Instruction instr) {}
void VirtualMachine::OpStopHook() {}

PackedFunc VirtualMachine::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK(exec_) << "The executable is not created yet.";

      std::string func_name = args[0];
      auto git = exec_->global_map.find(func_name);
      ICHECK(git != exec_->global_map.end())
          << "Cannot find function " << func_name << " in the executable";
      auto func = exec_->functions[git->second];
      if (func.params.empty()) {
        *rv = Invoke(func, {});
      } else {
        auto it = inputs_.find(func_name);
        ICHECK(it != inputs_.end()) << "Input has not been set for function " << func_name;
        const std::vector<ObjectRef>& input_args = it->second;
        if (set_outputs_enabled_.count(func_name) && set_outputs_enabled_[func_name]) {
          ICHECK(outputs_.count(func_name))
              << "Outputs have not been set for function " << func_name;
          *rv = Invoke(func, input_args, outputs_[func_name]);
          outputs_[func_name].clear();
          set_outputs_enabled_[func_name] = false;
        } else {
          *rv = Invoke(func, input_args);
        }
      }
    });
  } else if (name == "invoke_stateful") {
    // TODO(tkonolige, jroesch, tqchen): invoke_stateful and get_output are
    // stop-gap measure to allow using vm over a remote connection.
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      PackedFunc invoke = GetFunction("invoke", sptr_to_self);
      TVMRetValue rv_;
      invoke.CallPacked(args, &rv_);
    });
  } else if (name == "invoke_return_to_device") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Device host{static_cast<DLDeviceType>(args[1].operator int()), args[2].operator int()};

      SetInput(args[0].operator std::string(), args, 3);
      PackedFunc invoke = GetFunction("invoke", sptr_to_self);
      TVMRetValue rv_;
      invoke.CallPacked(args, &rv_);  // Invoke only uses the first arg, so the rest of the args
                                      // should not cause an issue
      if (rv_.type_code() == kTVMObjectHandle) {
        ADT adt = Downcast<ADT>(rv_.operator ObjectRef());
        std::vector<ObjectRef> transfered;
        for (size_t i = 0; i < adt.size(); i++) {
          transfered.push_back(CopyTo(adt[i], host));
        }
        *rv = ADT(adt.tag(), transfered);
      } else {
        *rv = CopyTo(rv_, host);
      }
    });
  } else if (name == "get_output") {
    return TypedPackedFunc<NDArray(int64_t)>([this](int64_t index) {
      if (this->return_register_.as<ADTObj>()) {
        return Downcast<NDArray>(Downcast<ADT>(this->return_register_)[index]);
      } else {
        CHECK_EQ(index, 0) << "VM output contains only one item, but you are trying to get the "
                           << index << "th.";
        return Downcast<NDArray>(this->return_register_);
      }
    });
  } else if (name == "get_num_outputs") {
    return TypedPackedFunc<int64_t(void)>([this]() -> int64_t {
      // single output is an NDArray not an ADT
      if (this->return_register_.as<ADTObj>()) {
        return Downcast<ADT>(this->return_register_).size();
      } else {
        return 1;
      }
    });
  } else if (name == "get_input_index") {
    return TypedPackedFunc<int64_t(std::string, std::string)>(
        [this](std::string input_name, std::string func_name) {
          return GetInputIndexFromVMFunction(func_name, input_name);
        });
  } else if (name == "init") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size() % 3, 0);
      std::vector<Device> devices;
      std::vector<AllocatorType> alloc_types;
      for (int i = 0; i < args.size() / 3; ++i) {
        Device dev;
        int device_type = args[i * 3];
        dev.device_type = DLDeviceType(device_type);
        dev.device_id = args[i * 3 + 1];
        int type = args[i * 3 + 2];
        devices.push_back(dev);
        alloc_types.push_back(AllocatorType(type));
      }
      this->Init(devices, alloc_types);
    });
  } else if (name == "set_input") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { SetInput(args[0], args, 1); });
  } else if (name == "set_one_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 3) << "The expected number of arguments is 3 "
                                << "(func_name, index or name, tensor)";
      SetOneInput(args[0], args[1], args[2]);
    });
  } else if (name == "set_outputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { SetOutputs(args[0], args); });
  } else if (name == "load_late_bound_consts") {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 1);
      std::string path = args[0];
      exec_->LoadLateBoundConstantsFromFile(path);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
  }
}

void VirtualMachine::SetInput(std::string func_name, TVMArgs args, int offset) {
  const auto& vm_func = CheckAndGetVMFunction(func_name);
  size_t params_num = vm_func.params.size();
  ICHECK_EQ(args.size() - offset, params_num)
      << "The number of provided parameters doesn't match the number of arguments";
  std::vector<ObjectRef> func_args(params_num);
  for (int i = offset; i < args.size(); ++i) {
    int index = i - offset;
    Device dev = GetDevice(vm_func.param_device_indexes[index]);
    SetInputTensorWithIndex(func_args, args[i], index, dev);
  }
  inputs_.erase(func_name);
  inputs_.emplace(func_name, func_args);
}

void VirtualMachine::SetOneInput(std::string func_name, const TVMArgValue& tag,
                                 const TVMArgValue& tensor) {
  const auto& vm_func = CheckAndGetVMFunction(func_name);
  size_t params_num = vm_func.params.size();

  int inp_index = 0;
  if (tag.type_code() == kTVMArgInt) {
    inp_index = tag;
  } else if (tag.type_code() == kTVMStr) {
    inp_index = static_cast<int>(GetInputIndexFromName(vm_func.params, tag));
  } else {
    LOG(FATAL) << "The type of input tensor tag (" << tag.type_code()
               << ") doesn't match integer or string";
  }
  ICHECK_LT(inp_index, params_num);

  CreateInputsOrCheckSize(func_name, params_num);
  Device dev = GetDevice(vm_func.param_device_indexes[inp_index]);
  SetInputTensorWithIndex(inputs_[func_name], tensor, inp_index, dev);
}

void VirtualMachine::SetOutputs(std::string func_name, TVMArgs args) {
  set_outputs_enabled_[func_name] = true;
  size_t outputs_size = args.size();
  // First args is func_name
  ICHECK_GT(outputs_size, 1) << "There is no output arguments set";

  std::vector<ObjectRef> func_args(outputs_size - 1);
  for (size_t i = 1; i < outputs_size; ++i) {
    // TODO(vvchernov): device?
    func_args[i - 1] = TensorFromTVMArgValueToObjectRef(args[i]);
  }
  outputs_.erase(func_name);
  outputs_.emplace(func_name, func_args);
}

void VirtualMachine::PrintInfoAndSetInputArgs(const VMFunction& func,
                                              const std::vector<ObjectRef>& args) {
  VLOG(2) << "Executing Function: " << std::endl << func;
  for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
    VLOG(2) << "Device " << i << " has device type " << devices_[i].device_type << " and device id "
            << devices_[i].device_id
            << (i == exec_->host_device_index ? " (using as host device)" : "");
  }

  InvokeGlobal(func, args);
}

void VirtualMachine::SetOutputTensorsToRegister(const std::string& func_name,
                                                const std::vector<ObjectRef>& outputs) {
  size_t size = outputs.size();

  if (output_tensor_reg_indices_[func_name].empty()) {
    output_tensor_reg_indices_[func_name] = GetOutputTensorRegIndices();
  }
  auto& reg_indices = output_tensor_reg_indices_[func_name];
  ICHECK_EQ(reg_indices.size(), size)
      << "Number of outside output tensors should be equal to model outputs number";
  size_t i = 0;
  for (auto it = reg_indices.begin(); it != reg_indices.end(); ++it, ++i) {
    WriteRegister(*it, outputs[i]);
  }
}

ObjectRef VirtualMachine::TensorFromTVMArgValueToObjectRef(const TVMArgValue& output_tensor) const {
  if (output_tensor.type_code() == kTVMDLTensorHandle) {
    DLTensor* dl_tensor = output_tensor;
    return NDArray::FromExternalDLTensor(*dl_tensor);
  } else if (output_tensor.type_code() == kTVMNDArrayHandle) {
    return output_tensor.AsObjectRef<tvm::runtime::NDArray>();
  } else {
    LOG(FATAL) << "It supports tensor of DLTensor or NDArray type only! Given type is "
               << output_tensor.type_code();
  }
  return ObjectRef();
}

int64_t VirtualMachine::GetInputIndexFromVMFunction(const std::string& func_name,
                                                    const std::string& input_name) const {
  const auto& vm_func = CheckAndGetVMFunction(func_name);
  return GetInputIndexFromName(vm_func.params, input_name);
}

int64_t VirtualMachine::GetInputIndexFromName(const std::vector<std::string>& params,
                                              const std::string& input_name) const {
  // TODO(vvchernov): excess integer type?
  for (uint64_t i = 0; i < params.size(); i++) {
    if (input_name == params[i]) {
      return static_cast<int64_t>(i);
    }
  }
  return static_cast<int64_t>(-1);
}

const VMFunction& VirtualMachine::CheckAndGetVMFunction(const std::string& func_name) const {
  ICHECK(exec_) << "The executable is not created yet.";
  return exec_->GetVMFunctionWithName(func_name);
}

void VirtualMachine::CreateInputsOrCheckSize(const std::string& func_name, size_t size) {
  if (inputs_.count(func_name)) {
    ICHECK_EQ(inputs_[func_name].size(), size)
        << "The size of function" << func_name
        << " doesn't match the number of provided parameters";
  } else {
    std::vector<ObjectRef> func_args(size);
    inputs_.emplace(func_name, func_args);
  }
}

void VirtualMachine::SetInputTensorWithIndex(std::vector<ObjectRef>& tensors,
                                             const TVMArgValue& inp_tensor, int index, Device dev) {
  if (inp_tensor.type_code() == kTVMDLTensorHandle) {
    if (NDArray::AbilityOfZeroCopyForDLTensor(inp_tensor, dev)) {
      tensors[index] = NDArray::FromExternalDLTensor(*inp_tensor);
    } else {
      tensors[index] = NDArray::NewFromDLTensor(inp_tensor, dev);
    }
  } else {
    tensors[index] = CopyTo(inp_tensor, dev);
  }
}

inline Device VirtualMachine::GetDevice(Index device_index) const {
  ICHECK_GE(devices_.size(), device_index) << "invalid device index: " << device_index;
  return devices_[device_index];
}

inline Allocator* VirtualMachine::GetAllocator(Index device_index) const {
  ICHECK_GE(allocators_.size(), device_index) << "invalid device index: " << device_index;
  return allocators_[device_index];
}

void VirtualMachine::PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, func_index_, arg_count, code_, vm_func.register_file_size);
  frames_.push_back(frame);
}

Index VirtualMachine::PopFrame() {
  ICHECK_GT(frames_.size(), 0);
  const VMFrame& fr = frames_.back();
  func_index_ = fr.func_index;
  code_ = fr.code;
  pc_ = fr.pc;
  auto call_stack_size = frames_.size();
  frames_.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<ObjectRef>& args) {
  VLOG(2) << "Invoking global " << func.name << " with " << args.size() << " args";

  PushFrame(func.params.size(), this->pc_ + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
    VLOG(2) << "arg " << i << " = "
            << RuntimeObject2String(args[i], GetDevice(exec_->host_device_index));
  }

  code_ = func.instructions.data();
  pc_ = 0;
}

ObjectRef VirtualMachine::Invoke(const VMFunction& func, const std::vector<ObjectRef>& args) {
  PrintInfoAndSetInputArgs(func, args);
  RunLoop();
  return return_register_;
}

ObjectRef VirtualMachine::Invoke(const std::string& name, const std::vector<ObjectRef>& args) {
  ICHECK(exec_) << "The executable has not been created yet.";
  auto it = exec_->global_map.find(name);
  ICHECK(it != exec_->global_map.end()) << "Cannot find function " << name << " in the executable";
  Index func_index = it->second;
  VLOG(2) << "Invoke Global " << name << " at index " << func_index;
  return Invoke(exec_->functions[func_index], args);
}

ObjectRef VirtualMachine::Invoke(const VMFunction& func, const std::vector<ObjectRef>& input_args,
                                 const std::vector<ObjectRef>& output_args) {
  PrintInfoAndSetInputArgs(func, input_args);
  SetOutputTensorsToRegister(func.name, output_args);
  RunLoop(output_tensor_reg_indices_[func.name]);
  return return_register_;
}

void VirtualMachine::InvokePacked(Index packed_index, const PackedFunc& func, Index arg_count,
                                  Index output_size, const std::vector<ObjectRef>& args) {
  size_t arity = 0;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* obj = args[i].as<ADTObj>()) {
      arity += obj->size;
    } else {
      ++arity;
    }
  }

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  int idx = 0;
  bool is_empty_output = false;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* dt_cell = args[i].as<ADTObj>()) {
      for (size_t fi = 0; fi < dt_cell->size; ++fi) {
        auto obj = (*dt_cell)[fi];
        auto nd_array = Downcast<NDArray>(obj);
        setter(idx++, nd_array);
      }
    } else {
      auto nd_array = Downcast<NDArray>(args[i]);
      // We can safely skip CallPacked if there is only one
      // output and it is empty.
      if (i == arg_count - 1 && output_size == 1) {
        for (const auto& dim : nd_array.Shape()) {
          if (!dim) {
            is_empty_output = true;
            break;
          }
        }
      }
      setter(idx++, nd_array);
    }
  }

  if (!is_empty_output) {
    TVMRetValue rv;
    func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
  }
}

void VirtualMachine::LoadExecutable(const ObjectPtr<Executable>& exec) {
  ICHECK(exec) << "The executable is not created yet.";
  ICHECK(exec->late_bound_constant_names.empty())
      << "Need to load late-bound-constants before creating VM";
  exec_ = exec;

  runtime::Module lib = exec_->GetLib();

  ICHECK(exec_->primitive_map.empty() || lib.operator->())
      << "If the executable has declared primitive functions, the "
      << "generated kernel library must non-be null.";

  for (const auto& it : exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (packed_funcs_.size() <= packed_index) {
      packed_funcs_.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, /*query_imports=*/true);
    ICHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    packed_funcs_[packed_index] = pf;
  }
  for (size_t i = 0; i < packed_funcs_.size(); ++i) {
    ICHECK(packed_funcs_[i] != nullptr) << "Packed function " << i << " is not initialized";
  }
}

void VirtualMachine::Init(const std::vector<Device>& physical_devices,
                          const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(physical_devices.size(), alloc_types.size());

  // Find a physical device to represent each virtual device the VM code requires.
  // (Recall the VM instructions refer to devices by "device index" into this vector of
  // virtual devices.)
  const size_t num_virtual_devices = exec_->virtual_devices.size();
  devices_.reserve(num_virtual_devices);
  allocators_.reserve(num_virtual_devices);

  for (size_t device_index = 0; device_index < num_virtual_devices; ++device_index) {
    // We'll retain the legacy behaviour and just match by device type.
    // TODO(mbs): Generalize.
    DLDeviceType virtual_device_type = exec_->virtual_devices[device_index].first.device_type;
    auto itr = std::find_if(physical_devices.begin(), physical_devices.end(),
                            [virtual_device_type](const Device& physical_device) {
                              return physical_device.device_type == virtual_device_type;
                            });
    CHECK(itr != physical_devices.end())
        << "Unable to find a physical device (from among the " << physical_devices.size()
        << " given) to match the virtual device with device type " << virtual_device_type;
    const size_t i = std::distance(physical_devices.begin(), itr);
    devices_.push_back(*itr);
    allocators_.push_back(MemoryManager::GetOrCreateAllocator(*itr, alloc_types[i]));
  }
}

inline void VirtualMachine::WriteRegister(Index r, const ObjectRef& val) {
  frames_.back().register_file[r] = val;
}

ObjectRef VirtualMachine::ReadRegister(Index r) const { return frames_.back().register_file[r]; }

int64_t VirtualMachine::LoadScalarInt(Index r) const {
  int64_t result = 0;
  const auto& obj = ReadRegister(r);
  NDArray array = Downcast<NDArray>(CopyTo(obj, GetDevice(exec_->host_device_index)));

  switch (array->dtype.bits) {
    case 1: {
      result = reinterpret_cast<bool*>(array->data)[0];
      break;
    }
    case 8: {
      result = reinterpret_cast<int8_t*>(array->data)[0];
      break;
    }
    case 16: {
      result = reinterpret_cast<int16_t*>(array->data)[0];
      break;
    }
    case 32: {
      result = reinterpret_cast<int32_t*>(array->data)[0];
      break;
    }
    case 64: {
      result = reinterpret_cast<int64_t*>(array->data)[0];
      break;
    }
    default:
      LOG(FATAL) << "Unknown scalar int type: " << DLDataType2String(array->dtype);
  }
  return result;
}

Index VirtualMachine::GetResultRegisterIndex() const {
  Index op_index = 0;
  while (code_[op_index].op != Opcode::Ret) {
    ++op_index;
  }

  return code_[op_index].result;
}

void VirtualMachine::CalculatePreResultOpIndex(Index res_index) {
  if (preresult_op_index_ == -1) {
    preresult_op_index_ = 0;
    while (code_[preresult_op_index_].dst != res_index) {
      ++preresult_op_index_;
    }
  }
}

std::vector<Index> VirtualMachine::GetOutputTensorRegIndices() {
  std::vector<Index> reg_indices;
  Index res_index = GetResultRegisterIndex();
  CalculatePreResultOpIndex(res_index);
  auto& preres_instr = code_[preresult_op_index_];
  auto op_code = preres_instr.op;
  if (op_code == Opcode::AllocTensor) {
    reg_indices.emplace_back(res_index);
  } else if (op_code == Opcode::AllocADT) {
    for (Index i = 0; i < preres_instr.num_fields; ++i) {
      reg_indices.push_back(preres_instr.datatype_fields[i]);
    }
  } else if (op_code == Opcode::ReshapeTensor) {
    reg_indices.push_back(preres_instr.reshape_tensor.tensor);
  } else {
    LOG(FATAL) << "Operation " << size_t(op_code) << " is not supported for set_outputs method";
  }
  return reg_indices;
}

void VirtualMachine::RunLoop(const std::vector<Index>& output_tensor_reg_indices) {
  ICHECK(this->exec_);
  ICHECK(this->code_);
  pc_ = 0;
  Index frame_start = frames_.size();
  while (true) {
  main_loop:
    auto const& instr = code_[this->pc_];
    VLOG(2) << "Executing(" << pc_ << "): " << instr;

    switch (instr.op) {
      case Opcode::Move: {
        ObjectRef from_obj;
        from_obj = ReadRegister(instr.from);
        WriteRegister(instr.dst, from_obj);
        pc_++;
        goto main_loop;
      }
      case Opcode::Fatal: {
        throw std::runtime_error("VM encountered fatal error");
      }
      case Opcode::LoadConst: {
        bool is_not_cached = const_pool_.size() <= static_cast<size_t>(instr.const_index) ||
                             !const_pool_[instr.const_index].defined();
        if (is_not_cached) {
          OpStartHook(instr);
        }
        auto constant_obj = exec_->constants[instr.const_index];
        // We cache the allocated object in the constant pool. To measure, the
        // first iteration will set the pool up. The other iterations will
        // directly reuse the allocated objects.
        if (const_pool_.size() <= static_cast<size_t>(instr.const_index)) {
          const_pool_.resize(instr.const_index + 1);
        }

        if (!const_pool_[instr.const_index].defined()) {
          auto& [dev, mem_scope] =
              exec_->virtual_devices[exec_->const_device_indexes[instr.const_index]];
          const_pool_[instr.const_index] = CopyTo(constant_obj, dev, String(mem_scope));
        }
        WriteRegister(instr.dst, const_pool_[instr.const_index]);
        if (is_not_cached) {
          OpStopHook();
        }
        pc_++;
        goto main_loop;
      }
      case Opcode::LoadConsti: {
        auto tensor = NDArray::Empty({1}, {kDLInt, 64, 1}, GetDevice(exec_->host_device_index));
        reinterpret_cast<int64_t*>(tensor->data)[0] = instr.load_consti.val;
        WriteRegister(instr.dst, tensor);
        pc_++;
        goto main_loop;
      }
      case Opcode::Invoke: {
        std::vector<ObjectRef> args;
        for (Index i = 0; i < instr.num_args; ++i) {
          args.push_back(ReadRegister(instr.invoke_args_registers[i]));
        }
        InvokeGlobal(exec_->functions[instr.func_index], args);
        frames_.back().caller_return_register = instr.dst;
        goto main_loop;
      }
      case Opcode::InvokePacked: {
        ICHECK_LE(instr.packed_index, packed_funcs_.size());
        const auto& func = packed_funcs_[instr.packed_index];
        const auto& arity = instr.arity;
        std::vector<ObjectRef> args;
        for (Index i = 0; i < arity; ++i) {
          auto arg = ReadRegister(instr.packed_args[i]);
          args.push_back(arg);
#if TVM_LOG_DEBUG
          if (i < arity) {
            const bool is_input = i < arity - instr.output_size;
            VLOG(2) << (is_input ? "input" : "placeholder") << " arg " << i << " = "
                    << RuntimeObject2String(arg, GetDevice(exec_->host_device_index),
                                            /*show_contents=*/is_input);
          }
#endif
        }

        // We no longer need to write the registers back, we write directly
        // through the registers mutably.
        InvokePacked(instr.packed_index, func, arity, instr.output_size, args);

#if TVM_LOG_DEBUG
        for (Index i = arity - instr.output_size; i < arity; ++i) {
          auto arg = ReadRegister(instr.packed_args[i]);
          VLOG(2) << "output arg " << i << " = "
                  << RuntimeObject2String(arg, GetDevice(exec_->host_device_index));
        }
#endif

        pc_++;
        goto main_loop;
      }
      case Opcode::InvokeClosure: {
        auto object = ReadRegister(instr.closure);
        const auto* closure = object.as<VMClosureObj>();
        ICHECK(closure);
        std::vector<ObjectRef> args;
        for (auto free_var : closure->free_vars) {
          args.push_back(free_var);
        }
        for (Index i = 0; i < instr.num_closure_args; ++i) {
          args.push_back(ReadRegister(instr.closure_args[i]));
        }
        InvokeGlobal(exec_->functions[closure->func_index], args);
        frames_.back().caller_return_register = instr.dst;
        goto main_loop;
      }
      case Opcode::GetField: {
        auto object = ReadRegister(instr.object);
        const auto& tuple = Downcast<ADT>(object);
        auto field = tuple[instr.field_index];
        WriteRegister(instr.dst, field);
        pc_++;
        goto main_loop;
      }
      case Opcode::GetTag: {
        auto object = ReadRegister(instr.get_tag.object);
        const auto& adt = Downcast<ADT>(object);
        auto tag = adt.tag();
        auto tag_tensor = NDArray::Empty({1}, {kDLInt, 32, 1}, GetDevice(exec_->host_device_index));
        reinterpret_cast<int32_t*>(tag_tensor->data)[0] = tag;
        WriteRegister(instr.dst, tag_tensor);
        pc_++;
        goto main_loop;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        goto main_loop;
      }
      case Opcode::If: {
        int32_t test_val = LoadScalarInt(instr.if_op.test);
        int32_t target_val = LoadScalarInt(instr.if_op.target);

        if (test_val == target_val) {
          ICHECK_NE(instr.if_op.true_offset, 0);
          pc_ += instr.if_op.true_offset;
        } else {
          ICHECK_NE(instr.if_op.false_offset, 0);
          pc_ += instr.if_op.false_offset;
        }

        goto main_loop;
      }
      case Opcode::AllocTensor: {
        OpStartHook(instr);
        if (!output_tensor_reg_indices.empty() && FindIndex(output_tensor_reg_indices, instr.dst)) {
          WriteAllocatedTensorFromOutside(instr);
        } else {
          WriteAllocatedTensor(instr);
        }
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocTensorReg: {
        OpStartHook(instr);
        Device cpu_dev = GetDevice(exec_->host_device_index);
        auto shape_obj = ReadRegister(instr.alloc_tensor_reg.shape_register);
        NDArray shape_tensor = Downcast<NDArray>(CopyTo(shape_obj, cpu_dev));
        auto shape = ToShape(shape_tensor);
        auto storage_obj = ReadRegister(instr.alloc_tensor_reg.storage);
        auto storage = Downcast<Storage>(storage_obj);
        auto offset = LoadScalarInt(instr.alloc_tensor.offset);
        auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor_reg.dtype);
        VLOG(2) << "allocated "
                << RuntimeObject2String(obj, GetDevice(exec_->host_device_index),
                                        /*show_contents=*/false);

        WriteRegister(instr.dst, obj);
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocADT: {
        std::vector<ObjectRef> fields;
        for (Index i = 0; i < instr.num_fields; ++i) {
          fields.push_back(ReadRegister(instr.datatype_fields[i]));
        }
        ObjectRef obj = ADT(instr.constructor_tag, fields);
        WriteRegister(instr.dst, obj);
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocClosure: {
        std::vector<ObjectRef> free_vars;
        for (Index i = 0; i < instr.num_freevar; i++) {
          free_vars.push_back(ReadRegister(instr.free_vars[i]));
        }
        WriteRegister(instr.dst, VMClosure(instr.func_index, free_vars));
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocStorage: {
        OpStartHook(instr);

        auto storage_obj = SimpleObjAllocator().make_object<StorageObj>();
        Allocator* allocator = GetAllocator(instr.alloc_storage.device_index);
        Device device = devices_[instr.alloc_storage.device_index];
        ICHECK(allocator) << "Did you forget to init the VirtualMachine with devices?";

        if (instr.alloc_storage.ndim > 0) {
          std::string shape = "[";
          for (uint32_t i = 0; i < instr.alloc_storage.ndim; ++i) {
            if (i > 0) {
              shape += ", ";
            }
            shape += std::to_string(instr.alloc_storage.shape[i]);
          }
          shape += "]";
          std::string mem_scope = exec_->virtual_devices[instr.alloc_storage.device_index].second;
          VLOG(2) << "allocating with ndims=" << instr.alloc_storage.ndim << ", shape=" << shape
                  << ", dtype_hint=" << DLDataType2String(instr.alloc_storage.dtype_hint)
                  << ", device_index=" << instr.alloc_storage.device_index
                  << ", memory_scope=" << mem_scope;

          std::vector<ShapeTuple::index_type> shape_;
          shape_.resize(instr.alloc_storage.ndim);
          shape_.assign(instr.alloc_storage.shape,
                        instr.alloc_storage.shape + instr.alloc_storage.ndim);
          storage_obj->buffer = allocator->Alloc(device, ShapeTuple(shape_),
                                                 instr.alloc_storage.dtype_hint, mem_scope);
          storage_obj->allocator = allocator;
        } else {
          auto size = LoadScalarInt(instr.alloc_storage.allocation_size);
          auto alignment = instr.alloc_storage.alignment;
          VLOG(2) << "allocating with allocation_size=" << size << ", alignment=" << alignment
                  << ", dtype_hint=" << DLDataType2String(instr.alloc_storage.dtype_hint)
                  << ", device_index=" << instr.alloc_storage.device_index;
          storage_obj->buffer =
              allocator->Alloc(device, size, alignment, instr.alloc_storage.dtype_hint);
          storage_obj->allocator = allocator;
        }
        Storage storage(storage_obj);
        WriteRegister(instr.dst, storage);
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      case Opcode::ShapeOf: {
        auto input = ReadRegister(instr.shape_of.tensor);
        NDArray input_array = Downcast<NDArray>(input);
        int ndim = input_array->ndim;
        auto out_tensor =
            NDArray::Empty({ndim}, {kDLInt, 64, 1}, GetDevice(exec_->host_device_index));
        for (int i = 0; i < ndim; ++i) {
          reinterpret_cast<int64_t*>(out_tensor->data)[i] = input_array->shape[i];
        }
        VLOG(2) << "shape = "
                << RuntimeObject2String(out_tensor, GetDevice(exec_->host_device_index));
        WriteRegister(instr.dst, out_tensor);
        pc_++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_register_ = ReadRegister(instr.result);
        auto caller_return_register = frames_.back().caller_return_register;

        if (PopFrame() == frame_start) {
          return;
          // Otherwise we are just returning from a local call.
        } else {
          WriteRegister(caller_return_register, return_register_);
          goto main_loop;
        }
      }
      case Opcode::ReshapeTensor: {
        OpStartHook(instr);
        Device cpu_dev = GetDevice(exec_->host_device_index);
        auto tensor_obj = ReadRegister(instr.reshape_tensor.tensor);
        NDArray tensor_arr = Downcast<NDArray>(tensor_obj);
        // Read the shape from shape tensor
        auto shape_obj = ReadRegister(instr.reshape_tensor.newshape);
        NDArray shape_tensor = Downcast<NDArray>(CopyTo(shape_obj, cpu_dev));
        const DLTensor* dl_tensor = shape_tensor.operator->();
        ICHECK_EQ(dl_tensor->dtype.code, 0u);
        ICHECK_EQ(dl_tensor->dtype.bits, 64u);
        int64_t* dims = reinterpret_cast<int64_t*>(dl_tensor->data);
        int64_t ndim = shape_tensor->shape[0];
        std::vector<int64_t> shape(dims, dims + ndim);
        // Reshape the input tensor
        auto out_tensor = tensor_arr.CreateView(shape, tensor_arr->dtype);
        VLOG(2) << "reshaped "
                << RuntimeObject2String(tensor_obj, GetDevice(exec_->host_device_index)) << " to "
                << RuntimeObject2String(out_tensor, GetDevice(exec_->host_device_index));
        WriteRegister(instr.dst, out_tensor);
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      case Opcode::DeviceCopy: {
        OpStartHook(instr);
        auto tensor_src = ReadRegister(instr.device_copy.src);
        NDArray src_data = Downcast<NDArray>(tensor_src);
        Device actual_src_dev = src_data->device;
        Device inst_src_dev = GetDevice(instr.device_copy.src_device_index);
        ICHECK_EQ(actual_src_dev.device_type, inst_src_dev.device_type);
        ICHECK_EQ(actual_src_dev.device_id, inst_src_dev.device_id);
        Device dst_dev = GetDevice(instr.device_copy.dst_device_index);
        auto mem_scope = exec_->virtual_devices[instr.device_copy.dst_device_index].second;

        NDArray dst_data = src_data.CopyTo(dst_dev, String(mem_scope));
        WriteRegister(instr.dst, dst_data);
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      case Opcode::KillRegister: {
        OpStartHook(instr);
        WriteRegister(instr.dst, ObjectRef());
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      default:
        LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
    }
  }
}

void VirtualMachine::WriteAllocatedTensor(const Instruction& instr) {
  auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

  for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
    shape[i] = instr.alloc_tensor.shape[i];
  }

  auto storage_obj = ReadRegister(instr.alloc_tensor.storage);
  auto offset = LoadScalarInt(instr.alloc_tensor.offset);
  auto storage = Downcast<Storage>(storage_obj);
  auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor.dtype);
  VLOG(2) << "allocated "
          << RuntimeObject2String(obj, GetDevice(exec_->host_device_index),
                                  /*show_contents=*/false);

  WriteRegister(instr.dst, obj);
}

void VirtualMachine::WriteAllocatedTensorFromOutside(const Instruction& instr) {
  // External tensor(s) has been already written to the register (instr.dst)
  auto ex_arr = Downcast<NDArray>(ReadRegister(instr.dst));
  auto ex_shape = ex_arr.Shape();
  auto ex_size = ex_shape.size();
  auto ex_dtype = ex_arr->dtype;

  auto in_size = instr.alloc_tensor.ndim;
  auto in_dtype = instr.alloc_tensor.dtype;
  ICHECK_EQ(TypeEqual(in_dtype, ex_dtype), true)
      << "Data types mismatching for internal and external output tensors";

  bool size_check = false;
  if (ex_size != in_size) {
    size_check = true;
  } else {
    for (size_t i = 0; i < in_size; ++i) {
      if (ex_shape[i] != instr.alloc_tensor.shape[i]) {
        size_check = true;
        break;
      }
    }
  }

  if (size_check) {
    // Match element number
    size_t in_el_num = 1, ex_el_num = 1;
    for (size_t i = 0; i < ex_size; ++i) {
      ex_el_num *= ex_shape[i];
    }
    for (size_t i = 0; i < in_size; ++i) {
      in_el_num *= instr.alloc_tensor.shape[i];
    }
    ICHECK_EQ(in_el_num, ex_el_num)
        << "Element number mismatching of internal and external output tensors";
    if (code_[preresult_op_index_].op == Opcode::ReshapeTensor) {
      int64_t* dims = instr.alloc_tensor.shape;
      std::vector<int64_t> ref_shape(dims, dims + int64_t(in_size));
      auto reshaped_tensor = ex_arr.CreateView(ref_shape, ex_dtype);
      WriteRegister(instr.dst, reshaped_tensor);
    } else {
      LOG(FATAL) << "Internal and external output tensor shapes are mismatched";
    }
  }
}

bool VirtualMachine::FindIndex(const std::vector<Index>& indices, Index val) const {
  auto it = std::find(indices.begin(), indices.end(), val);
  return it != indices.end();
}

runtime::Module CreateVirtualMachine(Executable* exec) {
  auto vm = make_object<VirtualMachine>();
  vm->LoadExecutable(GetObjectPtr<Executable>(exec));
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._VirtualMachine").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  auto* exec = dynamic_cast<Executable*>(mod.operator->());
  *rv = CreateVirtualMachine(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
