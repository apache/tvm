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

inline ObjectRef CopyTo(ObjectRef src, const DLDevice& dev) {
  if (src->IsInstance<NDArray::ContainerType>()) {
    auto nd_array = Downcast<NDArray>(src);
    if (nd_array->device.device_type != dev.device_type) {
      return nd_array.CopyTo(dev);
    }
    return src;
  } else {
    ICHECK(src->IsInstance<ADTObj>())
        << "VM data must be NDArray or a list of NDArray, but received: " << src->_type_key;
    std::vector<ObjectRef> ret;
    ADT adt = Downcast<ADT>(src);
    for (size_t i = 0; i < adt.size(); i++) {
      ret.push_back(CopyTo(adt[i], dev));
    }
    return ADT(adt->tag, ret.begin(), ret.end());
  }
}

std::vector<int64_t> ToShape(NDArray shape_tensor) {
  std::vector<int64_t> shape;
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

  return shape;
}

void VirtualMachine::OpStartHook(Instruction instr) {}
void VirtualMachine::OpStopHook() {}

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
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
        const std::vector<ObjectRef>& func_args = it->second;
        *rv = Invoke(func, func_args);
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
          auto gvit = exec_->global_map.find(func_name);
          ICHECK(gvit != exec_->global_map.end()) << "Cannot find function " << func_name;
          auto func_index = gvit->second;
          const auto& vm_func = exec_->functions[func_index];
          const auto& param_names = vm_func.params;
          for (uint64_t i = 0; i < param_names.size(); i++) {
            if (input_name == param_names[i]) {
              return static_cast<int64_t>(i);
            }
          }
          return static_cast<int64_t>(-1);
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
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VirtualMachine::SetInput(std::string func_name, TVMArgs args, int offset) {
  ICHECK(exec_) << "The executable is not created yet.";
  auto gvit = exec_->global_map.find(func_name);
  ICHECK(gvit != exec_->global_map.end()) << "Cannot find function " << func_name;
  auto func_index = gvit->second;
  const auto& vm_func = exec_->functions[func_index];
  const auto& param_names = vm_func.params;
  ICHECK_EQ(args.size() - offset, param_names.size())
      << "The number of provided parameters doesn't match the number of arguments";
  ICHECK_EQ(param_names.size(), vm_func.params_device_type.size())
      << "The number of provided parameters doesn't match the number of assigned devices";
  std::vector<ObjectRef> func_args(param_names.size());
  for (int i = offset; i < args.size(); ++i) {
    Index device_type = vm_func.params_device_type[i - offset];
    Device dev = GetDevice(device_type);

    if (args[i].type_code() == kTVMDLTensorHandle) {
      // Automatically convert input DLTensors to NDArray
      DLTensor* tensor = args[i];
      std::vector<int64_t> shape;
      for (int64_t i = 0; i < tensor->ndim; i++) {
        shape.push_back(tensor->shape[i]);
      }
      NDArray ary = NDArray::Empty(shape, tensor->dtype, dev);
      ary.CopyFrom(tensor);
      func_args[i - offset] = ary;
    } else {
      ObjectRef obj = CopyTo(args[i], dev);
      func_args[i - offset] = obj;
    }
  }
  inputs_.erase(func_name);
  inputs_.emplace(func_name, func_args);
}

inline Device VirtualMachine::GetDevice(Index device_type) const {
  ICHECK_GE(devices_.size(), device_type) << "devices_ doesn't contain device:" << device_type;

  auto dev = devices_[device_type];
  ICHECK_EQ(static_cast<Index>(dev.device_type), device_type)
      << "device type " << device_type << " has not been initialized in the device list.";
  return dev;
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
  VLOG(2) << "Invoking global " << func.name << " " << args.size();

  PushFrame(func.params.size(), this->pc_ + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  VLOG(2) << "func.params= " << func.params.size();

  code_ = func.instructions.data();
  pc_ = 0;
}

ObjectRef VirtualMachine::Invoke(const VMFunction& func, const std::vector<ObjectRef>& args) {
  VLOG(2) << "Executing Function: " << std::endl << func;

  InvokeGlobal(func, args);
  RunLoop();
  return return_register_;
}

ObjectRef VirtualMachine::Invoke(const std::string& name, const std::vector<ObjectRef>& args) {
  ICHECK(exec_) << "The executable has not been created yet.";
  auto it = exec_->global_map.find(name);
  ICHECK(it != exec_->global_map.end()) << "Cannot find function " << name << " in the executable";
  auto func_index_ = it->second;
  VLOG(2) << "Invoke Global " << name << " at index " << func_index_;
  return Invoke(exec_->functions[func_index_], args);
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

void VirtualMachine::LoadExecutable(const Executable* exec) {
  ICHECK(exec) << "The executable is not created yet.";
  exec_ = exec;

  runtime::Module lib = exec_->GetLib();

  ICHECK(exec->primitive_map.empty() || lib.operator->())
      << "If the executable has declared primitive functions, the"
      << "generated kernel library must non-be null.";

  for (const auto& it : exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (packed_funcs_.size() <= packed_index) {
      packed_funcs_.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, true);
    ICHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    packed_funcs_[packed_index] = pf;
  }
  for (size_t i = 0; i < packed_funcs_.size(); ++i) {
    ICHECK(packed_funcs_[i] != nullptr) << "Packed function " << i << " is not initialized";
  }
}

void VirtualMachine::Init(const std::vector<Device>& devs,
                          const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(devs.size(), alloc_types.size());
  // Cache the device
  for (size_t i = 0; i < devs.size(); i++) {
    auto dev_type = static_cast<size_t>(devs[i].device_type);
    auto alloc = MemoryManager::GetOrCreateAllocator(devs[i], alloc_types[i]);
    if (devices_.size() <= dev_type) {
      devices_.resize(dev_type + 1);
      allocators_.resize(dev_type + 1);
    }
    devices_[dev_type] = devs[i];
    allocators_[dev_type] = alloc;
  }
}

inline void VirtualMachine::WriteRegister(Index r, const ObjectRef& val) {
  frames_.back().register_file[r] = val;
}

ObjectRef VirtualMachine::ReadRegister(Index r) const { return frames_.back().register_file[r]; }

int64_t VirtualMachine::LoadScalarInt(Index r) const {
  int64_t result = 0;
  const auto& obj = ReadRegister(r);
  NDArray array = Downcast<NDArray>(CopyTo(obj, {kDLCPU, 0}));

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

void VirtualMachine::RunLoop() {
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
          Device dev = GetDevice(exec_->const_device_type[instr.const_index]);
          const_pool_[instr.const_index] = CopyTo(constant_obj, dev);
        }
        WriteRegister(instr.dst, const_pool_[instr.const_index]);
        if (is_not_cached) {
          OpStopHook();
        }
        pc_++;
        goto main_loop;
      }
      case Opcode::LoadConsti: {
        auto tensor = NDArray::Empty({1}, {kDLInt, 64, 1}, {kDLCPU, 0});
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
        VLOG(2) << "InvokedPacked " << instr.packed_index << " arity=" << instr.arity;
        ICHECK_LE(instr.packed_index, packed_funcs_.size());
        const auto& func = packed_funcs_[instr.packed_index];
        const auto& arity = instr.arity;
        std::vector<ObjectRef> args;
        for (Index i = 0; i < arity; ++i) {
          VLOG(2) << "arg" << i << " $" << instr.packed_args[i];
          auto arg = ReadRegister(instr.packed_args[i]);
          args.push_back(arg);
        }

        // We no longer need to write the registers back, we write directly
        // through the registers mutably.
        InvokePacked(instr.packed_index, func, arity, instr.output_size, args);
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
        auto tag_tensor = NDArray::Empty({1}, {kDLInt, 32, 1}, {kDLCPU, 0});
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
        auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

        for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
          shape[i] = instr.alloc_tensor.shape[i];
        }

        auto storage_obj = ReadRegister(instr.alloc_tensor.storage);
        auto offset = LoadScalarInt(instr.alloc_tensor.offset);
        auto storage = Downcast<Storage>(storage_obj);
#if TVM_LOG_DEBUG
        std::ostringstream os;
        os << "AllocTensor: ";
        os << "offset=" << offset;
        os << ", shape=[";
        for (auto i : shape) {
          os << i << ",";
        }
        os << "]";
        os << ", dtype=" << DLDataType2String(instr.alloc_tensor.dtype);
        VLOG(2) << os.str();
#endif
        auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor.dtype);

        WriteRegister(instr.dst, obj);
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocTensorReg: {
        OpStartHook(instr);
        Device cpu_dev = GetDevice(static_cast<Index>(kDLCPU));
        auto shape_obj = ReadRegister(instr.alloc_tensor_reg.shape_register);
        NDArray shape_tensor = Downcast<NDArray>(CopyTo(shape_obj, cpu_dev));
        auto shape = ToShape(shape_tensor);
        auto storage_obj = ReadRegister(instr.alloc_tensor_reg.storage);
        auto storage = Downcast<Storage>(storage_obj);
        auto offset = LoadScalarInt(instr.alloc_tensor.offset);
        auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor_reg.dtype);

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
        auto size = LoadScalarInt(instr.alloc_storage.allocation_size);
        auto alignment = instr.alloc_storage.alignment;
        auto storage_obj = SimpleObjAllocator().make_object<StorageObj>();
        auto dev_type = instr.alloc_storage.device_type;
        ICHECK_LT(static_cast<size_t>(dev_type), allocators_.size())
            << "Memory allocator for device " << dev_type << " has not been initialized";
        auto* alloc = allocators_[dev_type];
        ICHECK(alloc) << "Did you forget to init the VirtualMachine with devices?";
        VLOG(2) << "AllocStorage: allocation_size=" << size << ", alignment=" << alignment
                << ", dtype_hint=" << DLDataType2String(instr.alloc_storage.dtype_hint)
                << ", device_type=" << instr.alloc_storage.device_type;
        storage_obj->buffer = alloc->Alloc(size, alignment, instr.alloc_storage.dtype_hint);
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
        auto out_tensor = NDArray::Empty({ndim}, {kDLInt, 64, 1}, {kDLCPU, 0});
        for (int i = 0; i < ndim; ++i) {
          reinterpret_cast<int64_t*>(out_tensor->data)[i] = input_array->shape[i];
        }
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
        Device cpu_dev = GetDevice(static_cast<Index>(kDLCPU));
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
        WriteRegister(instr.dst, out_tensor);
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      case Opcode::DeviceCopy: {
        OpStartHook(instr);
        auto tensor_src = ReadRegister(instr.src);
        NDArray src_data = Downcast<NDArray>(tensor_src);
        Device src_dev = src_data->device;
        ICHECK_EQ(static_cast<Index>(src_dev.device_type), instr.src_device_type);

        Device dst_dev;
        dst_dev.device_type = static_cast<DLDeviceType>(instr.dst_device_type);
        dst_dev.device_id = 0;

        NDArray dst_data = src_data.CopyTo(dst_dev);
        WriteRegister(instr.dst, dst_data);
        OpStopHook();
        pc_++;
        goto main_loop;
      }
      default:
        LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
    }
  }
}

runtime::Module CreateVirtualMachine(const Executable* exec) {
  auto vm = make_object<VirtualMachine>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._VirtualMachine").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVirtualMachine(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
