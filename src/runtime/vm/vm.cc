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
 * \file src/runtime/vm/vm.cc
 * \brief The Relay virtual machine.
 */

#include <dmlc/memory_io.h>
#include <tvm/logging.h>
#include <tvm/runtime/vm.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "memory_manager.h"
#include "naive_allocator.h"

using namespace tvm::runtime;

namespace tvm {
namespace runtime {
namespace vm {

Instruction::Instruction() {}

template <typename T>
static T* Duplicate(T* src, Index size) {
  auto dst = new T[size];
  std::copy(src, src + size, dst);
  return dst;
}

Instruction::Instruction(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Move:
      this->from = instr.from;
      return;
    case Opcode::Fatal:
      return;
    case Opcode::Ret:
      this->result = instr.result;
      return;
    case Opcode::AllocTensor:
      this->alloc_tensor.ndim = instr.alloc_tensor.ndim;
      this->alloc_tensor.shape = Duplicate<int64_t>(instr.alloc_tensor.shape,
                                                    instr.alloc_tensor.ndim);
      this->alloc_tensor.dtype = instr.alloc_tensor.dtype;
      return;
    case Opcode::AllocTensorReg:
      this->alloc_tensor_reg.shape_register = instr.alloc_tensor_reg.shape_register;
      this->alloc_tensor_reg.dtype = instr.alloc_tensor_reg.dtype;
      return;
    case Opcode::AllocDatatype:
      this->constructor_tag = instr.constructor_tag;
      this->num_fields = instr.num_fields;
      this->datatype_fields = Duplicate<RegName>(instr.datatype_fields, instr.num_fields);
      return;
    case Opcode::AllocClosure:
      this->clo_index = instr.clo_index;
      this->num_freevar = instr.num_freevar;
      this->free_vars = Duplicate<RegName>(instr.free_vars, instr.num_freevar);
      return;
    case Opcode::InvokePacked:
      this->packed_index = instr.packed_index;
      this->arity = instr.arity;
      this->output_size = instr.output_size;
      this->packed_args = Duplicate<RegName>(instr.packed_args, instr.arity);
      return;
    case Opcode::InvokeClosure:
      this->closure = instr.closure;
      this->num_closure_args = instr.num_closure_args;
      this->closure_args = Duplicate<RegName>(instr.closure_args, instr.num_closure_args);
      return;
    case Opcode::Invoke:
      this->func_index = instr.func_index;
      this->num_args = instr.num_args;
      this->invoke_args_registers = Duplicate<RegName>(instr.invoke_args_registers, instr.num_args);
      return;
    case Opcode::If:
      this->if_op = instr.if_op;
      return;
    case Opcode::LoadConst:
      this->const_index = instr.const_index;
      return;
    case Opcode::LoadConsti:
      this->load_consti = instr.load_consti;
      return;
    case Opcode::GetField:
      this->object = instr.object;
      this->field_index = instr.field_index;
      return;
    case Opcode::GetTag:
      this->get_tag = instr.get_tag;
      return;
    case Opcode::Goto:
      this->pc_offset = instr.pc_offset;
      return;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

template<typename T>
static inline void FreeIf(T* t) {
  if (t != nullptr) {
    delete t;
  }
}

Instruction& Instruction::operator=(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Move:
      this->from = instr.from;
      return *this;
    case Opcode::Fatal:
      return *this;
    case Opcode::LoadConsti:
      this->load_consti = instr.load_consti;
      return *this;
    case Opcode::Ret:
      this->result = instr.result;
      return *this;
    case Opcode::AllocTensor:
      this->alloc_tensor.ndim = instr.alloc_tensor.ndim;
      this->alloc_tensor.shape = Duplicate<int64_t>(instr.alloc_tensor.shape,
                                                    instr.alloc_tensor.ndim);
      this->alloc_tensor.dtype = instr.alloc_tensor.dtype;
      return *this;
    case Opcode::AllocTensorReg:
      this->alloc_tensor_reg.shape_register = instr.alloc_tensor_reg.shape_register;
      this->alloc_tensor_reg.dtype = instr.alloc_tensor_reg.dtype;
      return *this;
    case Opcode::AllocDatatype:
      this->constructor_tag = instr.constructor_tag;
      this->num_fields = instr.num_fields;
      FreeIf(this->datatype_fields);
      this->datatype_fields = Duplicate<RegName>(instr.datatype_fields, instr.num_fields);
      return *this;
    case Opcode::AllocClosure:
      this->clo_index = instr.clo_index;
      this->num_freevar = instr.num_freevar;
      FreeIf(this->free_vars);
      this->free_vars = Duplicate<RegName>(instr.free_vars, instr.num_freevar);
      return *this;
    case Opcode::InvokePacked:
      this->packed_index = instr.packed_index;
      this->arity = instr.arity;
      this->output_size = instr.output_size;
      FreeIf(this->packed_args);
      this->packed_args = Duplicate<RegName>(instr.packed_args, instr.arity);
      return *this;
    case Opcode::InvokeClosure:
      this->closure = instr.closure;
      this->num_closure_args = instr.num_closure_args;
      FreeIf(this->closure_args);
      this->closure_args = Duplicate<RegName>(instr.closure_args, instr.num_closure_args);
      return *this;
    case Opcode::Invoke:
      this->func_index = instr.func_index;
      this->num_args = instr.num_args;
      FreeIf(this->invoke_args_registers);
      this->invoke_args_registers = Duplicate<RegName>(instr.invoke_args_registers, instr.num_args);
      return *this;
    case Opcode::If:
      this->if_op = instr.if_op;
      return *this;
    case Opcode::LoadConst:
      this->const_index = instr.const_index;
      return *this;
    case Opcode::GetField:
      this->object = instr.object;
      this->field_index = instr.field_index;
      return *this;
    case Opcode::GetTag:
      this->get_tag = instr.get_tag;
      return *this;
    case Opcode::Goto:
      this->pc_offset = instr.pc_offset;
      return *this;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

Instruction::~Instruction() {
  switch (this->op) {
    case Opcode::Move:
    case Opcode::Ret:
    case Opcode::AllocTensorReg:
    case Opcode::If:
    case Opcode::LoadConst:
    case Opcode::GetField:
    case Opcode::GetTag:
    case Opcode::Goto:
    case Opcode::LoadConsti:
    case Opcode::Fatal:
      return;
    case Opcode::AllocTensor:
      delete this->alloc_tensor.shape;
      return;
    case Opcode::AllocDatatype:
      delete this->datatype_fields;
      return;
    case Opcode::AllocClosure:
      delete this->free_vars;
      return;
    case Opcode::InvokePacked:
      delete this->packed_args;
      return;
    case Opcode::InvokeClosure:
      delete this->closure_args;
      return;
    case Opcode::Invoke:
      delete this->invoke_args_registers;
      return;
    default:
      std::ostringstream out;
      LOG(FATAL) << "Invalid instruction " << static_cast<int>(this->op);
  }
}

Instruction Instruction::Ret(RegName result) {
  Instruction instr;
  instr.op = Opcode::Ret;
  instr.result = result;
  return instr;
}

Instruction Instruction::Fatal() {
  Instruction instr;
  instr.op = Opcode::Fatal;
  return instr;
}

Instruction Instruction::InvokePacked(Index packed_index,
                                      Index arity,
                                      Index output_size,
                                      const std::vector<RegName>& args) {
  Instruction instr;
  instr.op = Opcode::InvokePacked;
  instr.packed_index = packed_index;
  instr.arity = arity;
  instr.output_size = output_size;
  instr.packed_args = new RegName[arity];
  for (Index i = 0; i < arity; ++i) {
    instr.packed_args[i] = args[i];
  }
  return instr;
}

Instruction Instruction::AllocTensor(std::vector<int64_t> shape, DLDataType dtype, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocTensor;
  instr.dst = dst;
  instr.alloc_tensor.ndim = shape.size();
  instr.alloc_tensor.shape = new int64_t[shape.size()];
  for (size_t i = 0; i < shape.size(); ++i) {
    instr.alloc_tensor.shape[i] = shape[i];
  }
  instr.alloc_tensor.dtype = dtype;
  return instr;
}

Instruction Instruction::AllocTensorReg(RegName shape_register, DLDataType dtype, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocTensorReg;
  instr.dst = dst;
  instr.alloc_tensor_reg.shape_register = shape_register;
  instr.alloc_tensor_reg.dtype = dtype;
  return instr;
}

Instruction Instruction::AllocDatatype(Index tag, Index num_fields,
                                       const std::vector<RegName>& datatype_fields, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocDatatype;
  instr.dst = dst;
  instr.constructor_tag = tag;
  instr.num_fields = num_fields;
  instr.datatype_fields = new RegName[num_fields];
  for (Index i = 0; i < num_fields; ++i) {
    instr.datatype_fields[i] = datatype_fields[i];
  }
  return instr;
}

Instruction Instruction::AllocClosure(Index func_index, Index free_vars,
                                      const std::vector<RegName>& free_var_register, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocClosure;
  instr.dst = dst;
  instr.clo_index = func_index;
  instr.num_freevar = free_vars;
  instr.free_vars = new RegName[instr.num_freevar];
  for (Index i = 0; i < instr.num_freevar; ++i) {
    instr.free_vars[i] = free_var_register[i];
  }
  return instr;
}

Instruction Instruction::GetField(RegName object, Index field_index, RegName dst) {
  Instruction instr;
  instr.op = Opcode::GetField;
  instr.dst = dst;
  instr.object = object;
  instr.field_index = field_index;
  return instr;
}

Instruction Instruction::GetTag(RegName object, RegName dst) {
  Instruction instr;
  instr.op = Opcode::GetTag;
  instr.dst = dst;
  instr.get_tag.object = object;
  return instr;
}

Instruction Instruction::If(RegName test, RegName target, Index true_branch, Index false_branch) {
  Instruction instr;
  instr.op = Opcode::If;
  instr.if_op.test = test;
  instr.if_op.target = target;
  instr.if_op.true_offset = true_branch;
  instr.if_op.false_offset = false_branch;
  return instr;
}

Instruction Instruction::Goto(Index pc_offset) {
  Instruction instr;
  instr.op = Opcode::Goto;
  instr.pc_offset = pc_offset;
  return instr;
}

Instruction Instruction::Invoke(Index func_index, const std::vector<RegName>& args_registers,
                                RegName dst) {
  Instruction instr;
  instr.op = Opcode::Invoke;
  instr.dst = dst;
  instr.func_index = func_index;
  instr.num_args = args_registers.size();
  instr.invoke_args_registers = new RegName[instr.num_args];
  for (Index i = 0; i < instr.num_args; ++i) {
    instr.invoke_args_registers[i] = args_registers[i];
  }
  return instr;
}

Instruction Instruction::InvokeClosure(RegName closure, const std::vector<RegName>& args,
                                       RegName dst) {
  Instruction instr;
  instr.op = Opcode::InvokeClosure;
  instr.dst = dst;
  instr.closure = closure;
  instr.num_closure_args = args.size();
  instr.closure_args = new RegName[args.size()];
  for (size_t i = 0; i < args.size(); ++i) {
    instr.closure_args[i] = args[i];
  }
  return instr;
}

Instruction Instruction::LoadConst(Index const_index, RegName dst) {
  Instruction instr;
  instr.op = Opcode::LoadConst;
  instr.dst = dst;
  instr.const_index = const_index;
  return instr;
}

Instruction Instruction::LoadConsti(Index val, RegName dst) {
  Instruction instr;
  instr.op = Opcode::LoadConsti;
  instr.dst = dst;
  instr.load_consti.val = val;
  return instr;
}

Instruction Instruction::Move(RegName src, RegName dst) {
  Instruction instr;
  instr.op = Opcode::Move;
  instr.dst = dst;
  instr.from = src;
  return instr;
}

void DLDatatypePrint(std::ostream& os, const DLDataType& dtype) {
  switch (dtype.code) {
    case kDLInt:
      os << "int";
      break;
    case kDLUInt:
      os << "uint";
      break;
    case kDLFloat:
      os << "float";
      break;
  }

  os << int(dtype.bits);
  if (dtype.lanes != 1) {
    os << "x" << dtype.lanes;
  }
}

template<typename T>
std::string StrJoin(T* items, int offset, int cnt, std::string delim = ", ") {
  if (cnt == 0) {
    return "";
  }
  std::ostringstream oss;
  oss << items[offset];
  for (int i = 1; i < cnt; ++i) {
    oss << delim << items[offset + i];
  }
  return oss.str();
}

void InstructionPrint(std::ostream& os, const Instruction& instr) {
  switch (instr.op) {
    case Opcode::Move: {
      os << "move $" << instr.dst << " $" << instr.from;
      break;
    }
    case Opcode::Ret: {
      os << "ret $" << instr.result;
      break;
    }
    case Opcode::Fatal: {
      os << "fatal";
      break;
    }
    case Opcode::InvokePacked: {
      os << "invoke_packed PackedFunc[" << instr.packed_index << "] (in: $"
         << StrJoin<RegName>(instr.packed_args, 0,
                             instr.arity - instr.output_size, ", $")
         << ", out: $"
         << StrJoin<RegName>(instr.packed_args, instr.arity - instr.output_size,
                             instr.output_size, ", $")
         << ")";
      break;
    }
    case Opcode::AllocTensor: {
      os << "alloc_tensor $" << instr.dst << " ["
         << StrJoin<int64_t>(instr.alloc_tensor.shape, 0,
                             instr.alloc_tensor.ndim)
         << "] ";
      DLDatatypePrint(os, instr.alloc_tensor.dtype);
      break;
    }
    case Opcode::AllocTensorReg: {
      os << "alloc_tensor_reg $" << instr.dst << " $"
         << instr.alloc_tensor_reg.shape_register << " ";
      DLDatatypePrint(os, instr.alloc_tensor_reg.dtype);
      break;
    }
    case Opcode::AllocDatatype: {
      os << "alloc_data $" << instr.dst << " tag(" << instr.constructor_tag << ") [$"
         << StrJoin<RegName>(instr.datatype_fields, 0, instr.num_fields, ",$") << "]";
      break;
    }
    case Opcode::AllocClosure: {
      os << "alloc_closure $" << instr.dst << " VMFunc[" << instr.clo_index
         << "]($" << StrJoin<RegName>(instr.free_vars, 0, instr.num_freevar, ",$")
         << ")";
      break;
    }
    case Opcode::If: {
      os << "if " << "$" << instr.if_op.test << " " << instr.if_op.target << " "
         << instr.if_op.true_offset << " " << instr.if_op.false_offset;
      break;
    }
    case Opcode::Invoke: {
      os << "invoke $" << instr.dst << " VMFunc[" << instr.func_index << "]($"
         << StrJoin<RegName>(instr.invoke_args_registers, 0, instr.num_args, ",$")
         << ")";
      break;
    }
    case Opcode::InvokeClosure: {
      os << "invoke_closure $" << instr.dst << " $" << instr.closure << "($"
         << StrJoin<RegName>(instr.closure_args, 0, instr.num_closure_args, ",$")
         << ")";
      break;
    }
    case Opcode::LoadConst: {
      os << "load_const $" << instr.dst << " Const[" << instr.const_index << "]";
      break;
    }
    case Opcode::LoadConsti: {
      os << "load_consti $" << instr.dst << " Const[" << instr.load_consti.val << "]";
      break;
    }
    case Opcode::GetField: {
      os << "get_field $" << instr.dst << " $" << instr.object << "["
         << instr.field_index << "]";
      break;
    }
    case Opcode::GetTag: {
      os << "get_tag $" << instr.dst << " $" << instr.get_tag.object;
      break;
    }
    case Opcode::Goto: {
      os << "goto " << instr.pc_offset;
      break;
    }
    default:
      LOG(FATAL) << "should never hit this case" << static_cast<int>(instr.op);
      break;
  }
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
  InstructionPrint(os, instr);
  return os;
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

Object CopyTo(Object src, const DLContext& ctx) {
  if (src->tag == ObjectTag::kTensor) {
    auto tensor = ToNDArray(src);
    if (tensor->ctx.device_type != ctx.device_type) {
      auto copy = tensor.CopyTo(ctx);
      return Object::Tensor(copy);
    } else {
      return src;
    }
  } else {
    return src;
  }
}

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      auto ctx = this->GetParamsContext();
      std::vector<Object> func_args;
      for (int i = 1; i < args.size(); ++i) {
        Object obj = CopyTo(args[i], ctx);
        func_args.push_back(obj);
      }
      auto it = std::find_if(functions.begin(), functions.end(),
                             [func_name](const VMFunction& func) {
                               return func.name == func_name;
                             });

      CHECK(it != functions.end()) << "Cannot find function " << func_name << "\n";
      CHECK_EQ(func_args.size() + params_.size(), it->params.size())
          << "The number of provided parameters doesn't match the number of arguments"
          << "\n";
      if (!params_.empty()) {
        for (const auto& p : it->params) {
          const auto& pit = params_.find(p);
          if (pit != params_.end()) {
            func_args.push_back(pit->second);
          }
        }
        CHECK_EQ(func_args.size(), it->params.size());
      }
      *rv = this->Invoke(func_name, func_args);
    });
  } else if (name == "init") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size() % 2, 0);
      std::vector<TVMContext> contexts;
      for (int i = 0; i < args.size() / 2; ++i) {
        TVMContext ctx;
        int device_type = args[i * 2];
        ctx.device_type = DLDeviceType(device_type);
        ctx.device_id = args[i * 2 + 1];
        contexts.push_back(ctx);
      }
      this->Init(contexts);
    });
  } else if (name == "load_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->LoadParams(args[0].operator std::string());
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

TVMContext VirtualMachine::GetParamsContext() const {
  // Use the fallback device if no device index is available.
  int fallback_device_type = static_cast<int>(ctxs[0].device_type);
  // TODO(wweic): For heterogeneous execution, get device information from byte

  const auto& cit =
    std::find_if(ctxs.begin(), ctxs.end(), [&fallback_device_type](const TVMContext& c) {
      return fallback_device_type == static_cast<int>(c.device_type);
    });
  return (cit == ctxs.end() ? ctxs[0] : *cit);
}

void VirtualMachine::LoadParams(const std::string& params) {
  dmlc::MemoryStringStream mss(const_cast<std::string*>(&params));
  dmlc::Stream* strm = &mss;
  uint64_t header, reserved;
  CHECK(strm->Read(&header)) << "Invalid parameter file";
  CHECK(header == kTVMNDArrayListMagic) << "Invalid parameter file";
  CHECK(strm->Read(&reserved)) << "Invalid parameter file";

  std::vector<std::string> names;
  CHECK(strm->Read(&names)) << "Invalid parameter file";

  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size()) << "Invalid parameter file";

  auto ctx = GetParamsContext();
  for (size_t i = 0; i < size; i++) {
    NDArray arr;
    CHECK(arr.Load(strm)) << "Invalid parameter file";
    runtime::Object obj = runtime::Object::Tensor(arr);
    auto copy = CopyTo(obj, ctx);
    params_.emplace(std::make_pair(names[i], copy));
  }
}

void VirtualMachine::PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, func_index, arg_count, code, vm_func.register_file_size);
  frames.push_back(frame);
}

Index VirtualMachine::PopFrame() {
  CHECK_GT(frames.size(), 0);
  const VMFrame& fr = frames.back();
  func_index = fr.func_index;
  code = fr.code;
  pc = fr.pc;
  auto call_stack_size = frames.size();
  frames.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<Object>& args) {
  DLOG(INFO) << "Invoking global " << func.name << " " << args.size();

  PushFrame(func.params.size(), this->pc + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  DLOG(INFO) << "func.params= " << func.params.size();

  code = func.instructions.data();
  pc = 0;
}

Object VirtualMachine::Invoke(const VMFunction& func, const std::vector<Object>& args) {
  DLOG(INFO) << "Executing Function: " << std::endl << func;

  InvokeGlobal(func, args);
  RunLoop();
  auto alloc = MemoryManager::Global()->GetAllocator(ctxs[0]);
  DLOG(INFO) << "Memory used: " << alloc->UsedMemory() << " B";
  return return_register;
}

Object VirtualMachine::Invoke(const std::string& name, const std::vector<Object>& args) {
  auto func_index = this->global_map[name];
  DLOG(INFO) << "Invoke Global " << name << " at index " << func_index;
  return Invoke(this->functions[func_index], args);
}

void VirtualMachine::InvokePacked(Index packed_index, const PackedFunc& func,
                                  Index arg_count, Index output_size,
                                  const std::vector<Object>& args) {
  size_t arity = 0;
  for (Index i = 0; i < arg_count; i++) {
    if (args[i].ptr_->tag == ObjectTag::kDatatype) {
      arity += args[i].AsDatatype()->fields.size();
    } else {
      ++arity;
    }
  }

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  int idx = 0;
  for (Index i = 0; i < arg_count; i++) {
    if (args[i].ptr_->tag == ObjectTag::kDatatype) {
      auto dt_cell = args[i].AsDatatype();
      for (auto obj : dt_cell->fields) {
        NDArray data = ToNDArray(obj);
        setter(idx++, data);
      }
    } else {
      NDArray data = ToNDArray(args[i]);
      setter(idx++, data);
    }
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
}

void VirtualMachine::Init(const std::vector<TVMContext>& ctxs) {
  this->ctxs = ctxs;

  // Get the list of packed functions.
  CHECK(primitive_map.empty() || lib.operator->())
      << "runtime module should have been built for primitive functions"
      << "\n";
  for (const auto& it : primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (packed_funcs.size() <= packed_index) {
      packed_funcs.resize(packed_index + 1);
    }
    packed_funcs[packed_index] = lib.GetFunction(packed_name);
  }
}

inline void VirtualMachine::WriteRegister(Index r, const Object& val) {
  frames.back().register_file[r] = val;
}

inline Object VirtualMachine::ReadRegister(Index r) const {
  return frames.back().register_file[r];
}

inline int32_t VirtualMachine::LoadScalarInt(Index r) const {
  int32_t result;
  const auto& obj = ReadRegister(r);
  NDArray array = ToNDArray(obj).CopyTo({kDLCPU, 0});

  if (array->dtype.bits <= 8) {
    result = reinterpret_cast<int8_t*>(array->data)[0];
  } else if (array->dtype.bits <= 16) {
    result = reinterpret_cast<int16_t*>(array->data)[0];
  } else {
    result = reinterpret_cast<int32_t*>(array->data)[0];
  }
  return result;
}

void VirtualMachine::RunLoop() {
  CHECK(this->code);
  this->pc = 0;
  Index frame_start = frames.size();
  while (true) {
  main_loop:
    auto const& instr = this->code[this->pc];
    DLOG(INFO) << "Executing(" << pc << "): " << instr;
#if USE_RELAY_DEBUG
    InstructionPrint(std::cout, instr);
#endif  // USE_RELAY_DEBUG

    switch (instr.op) {
      case Opcode::Move: {
        Object from_obj;
        from_obj = ReadRegister(instr.from);
        WriteRegister(instr.dst, from_obj);
        pc++;
        goto main_loop;
      }
      case Opcode::Fatal: {
        throw std::runtime_error("VM encountered fatal error");
      }
      case Opcode::LoadConst: {
        auto constant_obj = this->constants[instr.const_index];
        auto device_obj = CopyTo(constant_obj, ctxs[0]);
        WriteRegister(instr.dst, device_obj);
        pc++;
        goto main_loop;
      }
      case Opcode::LoadConsti: {
        auto tensor = NDArray::Empty({1}, {kDLInt, 64, 1}, {kDLCPU, 0});
        reinterpret_cast<int64_t*>(tensor->data)[0] = instr.load_consti.val;
        WriteRegister(instr.dst, Object::Tensor(tensor));
        pc++;
        goto main_loop;
      }
      case Opcode::Invoke: {
        std::vector<Object> args;
        for (Index i = 0; i < instr.num_args; ++i) {
          args.push_back(ReadRegister(instr.invoke_args_registers[i]));
        }
        InvokeGlobal(this->functions[instr.func_index], args);
        frames.back().caller_return_register = instr.dst;
        goto main_loop;
      }
      case Opcode::InvokePacked: {
        const auto& func = packed_funcs[instr.packed_index];
        const auto& arity = instr.arity;
        std::vector<Object> args;
        for (Index i = 0; i < arity; ++i) {
          args.push_back(ReadRegister(instr.packed_args[i]));
        }
        InvokePacked(instr.packed_index, func, arity, instr.output_size, args);
        for (Index i = 0; i < instr.output_size; ++i) {
          WriteRegister(instr.packed_args[instr.arity - instr.output_size + i],
                        args[instr.arity - instr.output_size + i]);
        }
        pc++;
        goto main_loop;
      }
      case Opcode::InvokeClosure: {
        auto object = ReadRegister(instr.closure);
        const auto& closure = object.AsClosure();
        std::vector<Object> args;
        for (auto free_var : closure->free_vars) {
          args.push_back(free_var);
        }
        for (Index i = 0; i < instr.num_closure_args; ++i) {
          args.push_back(ReadRegister(instr.closure_args[i]));
        }
        InvokeGlobal(this->functions[closure->func_index], args);
        frames.back().caller_return_register = instr.dst;
        goto main_loop;
      }
      case Opcode::GetField: {
        auto object = ReadRegister(instr.object);
        CHECK(object->tag == ObjectTag::kDatatype)
            << "Object is not data type object, register " << instr.object << ", Object tag "
            << static_cast<int>(object->tag);
        const auto& tuple = object.AsDatatype();
        auto field = tuple->fields[instr.field_index];
        WriteRegister(instr.dst, field);
        pc++;
        goto main_loop;
      }
      case Opcode::GetTag: {
        auto object = ReadRegister(instr.get_tag.object);
        CHECK(object->tag == ObjectTag::kDatatype)
            << "Object is not data type object, register "
            << instr.get_tag.object << ", Object tag "
            << static_cast<int>(object->tag);
        const auto& data = object.AsDatatype();
        auto tag = data->tag;
        auto tag_tensor = NDArray::Empty({1}, {kDLInt, 32, 1}, {kDLCPU, 0});
        reinterpret_cast<int32_t*>(tag_tensor->data)[0] = tag;
        WriteRegister(instr.dst, Object::Tensor(tag_tensor));
        pc++;
        goto main_loop;
      }
      case Opcode::Goto: {
        pc += instr.pc_offset;
        goto main_loop;
      }
      case Opcode::If: {
        int32_t test_val = LoadScalarInt(instr.if_op.test);
        int32_t target_val = LoadScalarInt(instr.if_op.target);

        if (test_val == target_val) {
          CHECK_NE(instr.if_op.true_offset, 0);
          pc += instr.if_op.true_offset;
        } else {
          CHECK_NE(instr.if_op.false_offset, 0);
          pc += instr.if_op.false_offset;
        }

        goto main_loop;
      }
      case Opcode::AllocTensor: {
        auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);
        for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
          shape[i] = instr.alloc_tensor.shape[i];
        }
        auto allocator = MemoryManager::Global()->GetAllocator(ctxs[0]);
        auto data = allocator->Empty(shape, instr.alloc_tensor.dtype, ctxs[0]);
        auto obj = Object::Tensor(data);
        WriteRegister(instr.dst, obj);
        pc++;
        goto main_loop;
      }
      case Opcode::AllocTensorReg: {
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;

        auto shape_tensor_obj = ReadRegister(instr.alloc_tensor_reg.shape_register);
        NDArray shape_tensor = ToNDArray(shape_tensor_obj).CopyTo(cpu_ctx);

        int64_t* dims = static_cast<int64_t*>(shape_tensor->data);
        auto num_dims = shape_tensor->shape[0];
        auto shape = std::vector<int64_t>(shape_tensor->shape[0]);
        shape.assign(dims, dims + num_dims);
        auto allocator = MemoryManager::Global()->GetAllocator(ctxs[0]);
        auto data = allocator->Empty(shape, instr.alloc_tensor_reg.dtype, ctxs[0]);
        auto obj = Object::Tensor(data);
        WriteRegister(instr.dst, obj);
        pc++;
        goto main_loop;
      }
      case Opcode::AllocDatatype: {
        std::vector<Object> fields;
        for (Index i = 0; i < instr.num_fields; ++i) {
          fields.push_back(ReadRegister(instr.datatype_fields[i]));
        }
        Object obj = Object::Datatype(instr.constructor_tag, fields);
        WriteRegister(instr.dst, obj);
        pc++;
        goto main_loop;
      }
      case Opcode::AllocClosure: {
        std::vector<Object> free_vars;
        for (Index i = 0; i < instr.num_freevar; i++) {
          free_vars.push_back(ReadRegister(instr.free_vars[i]));
        }
        WriteRegister(instr.dst, Object::Closure(instr.func_index, free_vars));
        pc++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_register = ReadRegister(instr.result);
        auto caller_return_register = frames.back().caller_return_register;

        if (PopFrame() == frame_start) {
          return;
          // Otherwise we are just returning from a local call.
        } else {
          WriteRegister(caller_return_register, return_register);
          goto main_loop;
        }
      }
    }
  }
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
