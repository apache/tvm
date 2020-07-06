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
 * \brief The Relay virtual machine.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm.h>
#include <tvm/support/logging.h>

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

VMClosure::VMClosure(size_t func_index, std::vector<ObjectRef> free_vars) {
  auto ptr = make_object<VMClosureObj>();
  ptr->func_index = func_index;
  ptr->free_vars = std::move(free_vars);
  data_ = std::move(ptr);
}

inline Storage make_storage(size_t size, size_t alignment, DLDataType dtype_hint, TVMContext ctx) {
  // We could put cache in here, from ctx to storage allocator.
  auto storage_obj = SimpleObjAllocator().make_object<StorageObj>();
  auto alloc = MemoryManager::Global()->GetAllocator(ctx);
  DCHECK(alloc != nullptr) << "allocator must not null";
  storage_obj->buffer = alloc->Alloc(size, alignment, dtype_hint);
  return Storage(storage_obj);
}

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
      this->alloc_tensor.storage = instr.alloc_tensor.storage;
      this->alloc_tensor.offset = instr.alloc_tensor.offset;
      this->alloc_tensor.ndim = instr.alloc_tensor.ndim;
      this->alloc_tensor.shape =
          Duplicate<int64_t>(instr.alloc_tensor.shape, instr.alloc_tensor.ndim);
      this->alloc_tensor.dtype = instr.alloc_tensor.dtype;
      return;
    case Opcode::AllocTensorReg:
      this->alloc_tensor_reg.storage = instr.alloc_tensor_reg.storage;
      this->alloc_tensor_reg.offset = instr.alloc_tensor_reg.offset;
      this->alloc_tensor_reg.shape_register = instr.alloc_tensor_reg.shape_register;
      this->alloc_tensor_reg.dtype = instr.alloc_tensor_reg.dtype;
      return;
    case Opcode::AllocADT:
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
    case Opcode::AllocStorage:
      this->alloc_storage = instr.alloc_storage;
      return;
    case Opcode::ShapeOf:
      this->shape_of.tensor = instr.shape_of.tensor;
      return;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

template <typename T>
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
      this->alloc_tensor.storage = this->alloc_tensor.storage;
      this->alloc_tensor.offset = instr.alloc_tensor.offset;
      this->alloc_tensor.ndim = instr.alloc_tensor.ndim;
      this->alloc_tensor.shape =
          Duplicate<int64_t>(instr.alloc_tensor.shape, instr.alloc_tensor.ndim);
      this->alloc_tensor.dtype = instr.alloc_tensor.dtype;
      return *this;
    case Opcode::AllocTensorReg:
      this->alloc_tensor_reg.storage = instr.alloc_tensor_reg.storage;
      this->alloc_tensor_reg.offset = instr.alloc_tensor_reg.offset;
      this->alloc_tensor_reg.shape_register = instr.alloc_tensor_reg.shape_register;
      this->alloc_tensor_reg.dtype = instr.alloc_tensor_reg.dtype;
      return *this;
    case Opcode::AllocADT:
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
    case Opcode::AllocStorage:
      this->alloc_storage = instr.alloc_storage;
      return *this;
    case Opcode::ShapeOf:
      this->shape_of.tensor = instr.shape_of.tensor;
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
    case Opcode::AllocStorage:
    case Opcode::ShapeOf:
    case Opcode::Fatal:
      return;
    case Opcode::AllocTensor:
      delete[] this->alloc_tensor.shape;
      return;
    case Opcode::AllocADT:
      delete[] this->datatype_fields;
      return;
    case Opcode::AllocClosure:
      delete[] this->free_vars;
      return;
    case Opcode::InvokePacked:
      delete[] this->packed_args;
      return;
    case Opcode::InvokeClosure:
      delete[] this->closure_args;
      return;
    case Opcode::Invoke:
      delete[] this->invoke_args_registers;
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

Instruction Instruction::InvokePacked(Index packed_index, Index arity, Index output_size,
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

Instruction Instruction::AllocTensor(RegName storage, RegName offset,
                                     const std::vector<int64_t>& shape, DLDataType dtype,
                                     Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocTensor;
  instr.dst = dst;
  instr.alloc_tensor.storage = storage;
  instr.alloc_tensor.offset = offset;
  instr.alloc_tensor.ndim = shape.size();
  instr.alloc_tensor.shape = new int64_t[shape.size()];
  for (size_t i = 0; i < shape.size(); ++i) {
    instr.alloc_tensor.shape[i] = shape[i];
  }
  instr.alloc_tensor.dtype = dtype;
  return instr;
}

Instruction Instruction::AllocTensorReg(RegName storage, RegName offset, RegName shape_register,
                                        DLDataType dtype, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocTensorReg;
  instr.dst = dst;
  instr.alloc_tensor_reg.storage = storage;
  instr.alloc_tensor_reg.offset = offset;
  instr.alloc_tensor_reg.shape_register = shape_register;
  instr.alloc_tensor_reg.dtype = dtype;
  return instr;
}

Instruction Instruction::AllocStorage(RegName size, Index alignment, DLDataType dtype_hint,
                                      Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocStorage;
  instr.dst = dst;
  instr.alloc_storage.allocation_size = size;
  instr.alloc_storage.alignment = alignment;
  instr.alloc_storage.dtype_hint = dtype_hint;
  return instr;
}

Instruction Instruction::ShapeOf(RegName tensor, Index dst) {
  Instruction instr;
  instr.op = Opcode::ShapeOf;
  instr.dst = dst;
  instr.shape_of.tensor = tensor;
  return instr;
}

Instruction Instruction::AllocADT(Index tag, Index num_fields,
                                  const std::vector<RegName>& datatype_fields, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocADT;
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

template <typename T>
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
         << StrJoin<RegName>(instr.packed_args, 0, instr.arity - instr.output_size, ", $")
         << ", out: $"
         << StrJoin<RegName>(instr.packed_args, instr.arity - instr.output_size, instr.output_size,
                             ", $")
         << ")";
      break;
    }
    case Opcode::AllocTensor: {
      os << "alloc_tensor $" << instr.dst << " $" << instr.alloc_tensor.storage << " $"
         << instr.alloc_tensor.offset << " ["
         << StrJoin<int64_t>(instr.alloc_tensor.shape, 0, instr.alloc_tensor.ndim) << "] ";
      DLDatatypePrint(os, instr.alloc_tensor.dtype);
      break;
    }
    case Opcode::AllocTensorReg: {
      os << "alloc_tensor_reg $" << instr.dst << " $" << instr.alloc_tensor_reg.storage << " $"
         << instr.alloc_tensor_reg.offset << " $" << instr.alloc_tensor_reg.shape_register << " ";
      DLDatatypePrint(os, instr.alloc_tensor_reg.dtype);
      break;
    }
    case Opcode::AllocADT: {
      os << "alloc_data $" << instr.dst << " tag(" << instr.constructor_tag << ") [$"
         << StrJoin<RegName>(instr.datatype_fields, 0, instr.num_fields, ",$") << "]";
      break;
    }
    case Opcode::AllocClosure: {
      os << "alloc_closure $" << instr.dst << " VMFunc[" << instr.clo_index << "]($"
         << StrJoin<RegName>(instr.free_vars, 0, instr.num_freevar, ",$") << ")";
      break;
    }
    case Opcode::If: {
      os << "if "
         << "$" << instr.if_op.test << " $" << instr.if_op.target << " " << instr.if_op.true_offset
         << " " << instr.if_op.false_offset;
      break;
    }
    case Opcode::Invoke: {
      os << "invoke $" << instr.dst << " VMFunc[" << instr.func_index << "]($"
         << StrJoin<RegName>(instr.invoke_args_registers, 0, instr.num_args, ",$") << ")";
      break;
    }
    case Opcode::InvokeClosure: {
      os << "invoke_closure $" << instr.dst << " $" << instr.closure << "($"
         << StrJoin<RegName>(instr.closure_args, 0, instr.num_closure_args, ",$") << ")";
      break;
    }
    case Opcode::LoadConst: {
      os << "load_const $" << instr.dst << " Const[" << instr.const_index << "]";
      break;
    }
    case Opcode::LoadConsti: {
      os << "load_consti $" << instr.dst << " " << instr.load_consti.val;
      break;
    }
    case Opcode::GetField: {
      os << "get_field $" << instr.dst << " $" << instr.object << "[" << instr.field_index << "]";
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
    case Opcode::AllocStorage: {
      os << "alloc_storage $" << instr.dst << " $" << instr.alloc_storage.allocation_size << " "
         << instr.alloc_storage.alignment << " "
         << DLDataType2String(instr.alloc_storage.dtype_hint);
      break;
    }
    case Opcode::ShapeOf: {
      os << "shape_of $" << instr.dst << " $" << instr.shape_of.tensor;
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

inline ObjectRef CopyTo(ObjectRef src, const DLContext& ctx) {
  if (src->IsInstance<NDArray::ContainerType>()) {
    auto nd_array = Downcast<NDArray>(src);
    if (nd_array->ctx.device_type != ctx.device_type) {
      return nd_array.CopyTo(ctx);
    }
  }
  return src;
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
  CHECK_EQ(rank, 1U) << "shape tensor should be a k-length vector, found " << rank;
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

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK(exec_) << "The executable is not created yet.";
      std::string func_name = args[0];
      auto git = exec_->global_map.find(func_name);
      CHECK(git != exec_->global_map.end())
          << "Cannot find function " << func_name << " in the executable";
      auto func = exec_->functions[git->second];
      if (func.params.empty()) {
        *rv = Invoke(func, {});
      } else {
        auto it = inputs_.find(func_name);
        CHECK(it != inputs_.end()) << "Input has not been set for function " << func_name;
        const std::vector<ObjectRef>& func_args = it->second;
        *rv = Invoke(func, func_args);
      }
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
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK(exec_) << "The executable is not created yet.";
      std::string func_name = args[0];
      auto gvit = exec_->global_map.find(func_name);
      CHECK(gvit != exec_->global_map.end()) << "Cannot find function " << func_name;
      auto func_index = gvit->second;
      const auto& vm_func = exec_->functions[func_index];
      const auto& param_names = vm_func.params;
      // TODO(icemelon9): For heterogeneous execution, get input device information
      TVMContext ctx = ctxs_[0];
      CHECK_EQ(args.size() - 1, param_names.size())
          << "The number of provided parameters doesn't match the number of arguments";
      std::vector<ObjectRef> func_args(param_names.size());
      for (int i = 1; i < args.size(); ++i) {
        ObjectRef obj = CopyTo(args[i], ctx);
        func_args[i - 1] = obj;
      }
      inputs_.erase(func_name);
      inputs_.emplace(func_name, func_args);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

TVMContext VirtualMachine::GetParamsContext() const {
  CHECK(!ctxs_.empty()) << "Context has not been initialized yet.";

  // Use the fallback device if no device index is available.
  int fallback_device_type = static_cast<int>(ctxs_[0].device_type);
  // TODO(wweic): For heterogeneous execution, get device information from byte

  const auto& cit =
      std::find_if(ctxs_.begin(), ctxs_.end(), [&fallback_device_type](const TVMContext& c) {
        return fallback_device_type == static_cast<int>(c.device_type);
      });
  return (cit == ctxs_.end() ? ctxs_[0] : *cit);
}

void VirtualMachine::PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, func_index_, arg_count, code_, vm_func.register_file_size);
  frames_.push_back(frame);
}

Index VirtualMachine::PopFrame() {
  CHECK_GT(frames_.size(), 0);
  const VMFrame& fr = frames_.back();
  func_index_ = fr.func_index;
  code_ = fr.code;
  pc_ = fr.pc;
  auto call_stack_size = frames_.size();
  frames_.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<ObjectRef>& args) {
  DLOG(INFO) << "Invoking global " << func.name << " " << args.size();

  PushFrame(func.params.size(), this->pc_ + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  DLOG(INFO) << "func.params= " << func.params.size();

  code_ = func.instructions.data();
  pc_ = 0;
}

ObjectRef VirtualMachine::Invoke(const VMFunction& func, const std::vector<ObjectRef>& args) {
  DLOG(INFO) << "Executing Function: " << std::endl << func;

  InvokeGlobal(func, args);
  RunLoop();
  // TODO(wweic) ctx could be obtained from the ctxs list.
  auto alloc = MemoryManager::Global()->GetAllocator(ctxs_[0]);
  DLOG(INFO) << "Memory used: " << alloc->UsedMemory() << " B";
  return return_register_;
}

ObjectRef VirtualMachine::Invoke(const std::string& name, const std::vector<ObjectRef>& args) {
  CHECK(exec_) << "The executable has not been created yet.";
  auto it = exec_->global_map.find(name);
  CHECK(it != exec_->global_map.end()) << "Cannot find function " << name << " in the executable";
  auto func_index_ = it->second;
  DLOG(INFO) << "Invoke Global " << name << " at index " << func_index_;
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
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* dt_cell = args[i].as<ADTObj>()) {
      for (size_t fi = 0; fi < dt_cell->size; ++fi) {
        auto obj = (*dt_cell)[fi];
        auto nd_array = Downcast<NDArray>(obj);
        setter(idx++, nd_array);
      }
    } else {
      auto nd_array = Downcast<NDArray>(args[i]);
      setter(idx++, nd_array);
    }
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
}

void VirtualMachine::LoadExecutable(const Executable* exec) {
  CHECK(exec) << "The executable is not created yet.";
  exec_ = exec;

  runtime::Module lib = exec_->lib;
  // Get the list of packed functions.
  CHECK(exec->primitive_map.empty() || lib.operator->())
      << "runtime module should have been built for primitive functions"
      << "\n";
  for (const auto& it : exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (packed_funcs_.size() <= packed_index) {
      packed_funcs_.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, true);
    CHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    packed_funcs_[packed_index] = pf;
  }
  for (size_t i = 0; i < packed_funcs_.size(); ++i) {
    CHECK(packed_funcs_[i] != nullptr) << "Packed function " << i << " is not initialized";
  }
}

void VirtualMachine::Init(const std::vector<TVMContext>& ctxs) { ctxs_ = ctxs; }

inline void VirtualMachine::WriteRegister(Index r, const ObjectRef& val) {
  frames_.back().register_file[r] = val;
}

inline ObjectRef VirtualMachine::ReadRegister(Index r) const {
  return frames_.back().register_file[r];
}

inline int64_t VirtualMachine::LoadScalarInt(Index r) const {
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
  CHECK(this->exec_);
  CHECK(this->code_);
  pc_ = 0;
  Index frame_start = frames_.size();
  while (true) {
  main_loop:
    auto const& instr = code_[this->pc_];
    DLOG(INFO) << "Executing(" << pc_ << "): " << instr;
#if USE_RELAY_DEBUG
    InstructionPrint(std::cout, instr);
#endif  // USE_RELAY_DEBUG

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
        auto constant_obj = exec_->constants[instr.const_index];
        // We cache the allocated object in the constant pool. To measure, the
        // first iteration will set the pool up. The other iterations will
        // directly reuse the allocated objects.
        if (const_pool_.size() <= static_cast<size_t>(instr.const_index)) {
          const_pool_.resize(instr.const_index + 1);
        }

        if (!const_pool_[instr.const_index].defined()) {
          // TODO(wweic) ctx could be obtained from the ctxs list.
          const_pool_[instr.const_index] = CopyTo(constant_obj, ctxs_[0]);
        }
        WriteRegister(instr.dst, const_pool_[instr.const_index]);
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
        DLOG(INFO) << "InvokedPacked " << instr.packed_index << " arity=" << instr.arity;
        CHECK_LE(instr.packed_index, packed_funcs_.size());
        const auto& func = packed_funcs_[instr.packed_index];
        const auto& arity = instr.arity;
        std::vector<ObjectRef> args;
        for (Index i = 0; i < arity; ++i) {
          DLOG(INFO) << "arg" << i << " $" << instr.packed_args[i];
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
          CHECK_NE(instr.if_op.true_offset, 0);
          pc_ += instr.if_op.true_offset;
        } else {
          CHECK_NE(instr.if_op.false_offset, 0);
          pc_ += instr.if_op.false_offset;
        }

        goto main_loop;
      }
      case Opcode::AllocTensor: {
        auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

        for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
          shape[i] = instr.alloc_tensor.shape[i];
        }

        auto storage_obj = ReadRegister(instr.alloc_tensor.storage);
        auto offset = LoadScalarInt(instr.alloc_tensor.offset);
        auto storage = Downcast<Storage>(storage_obj);
        auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor.dtype);

        WriteRegister(instr.dst, obj);
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocTensorReg: {
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;
        auto shape_obj = ReadRegister(instr.alloc_tensor_reg.shape_register);
        NDArray shape_tensor = Downcast<NDArray>(CopyTo(shape_obj, cpu_ctx));
        auto shape = ToShape(shape_tensor);
        auto storage_obj = ReadRegister(instr.alloc_tensor_reg.storage);
        auto storage = Downcast<Storage>(storage_obj);
        auto offset = LoadScalarInt(instr.alloc_tensor.offset);
        auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor_reg.dtype);

        WriteRegister(instr.dst, obj);
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
        auto size = LoadScalarInt(instr.alloc_storage.allocation_size);
        auto alignment = instr.alloc_storage.alignment;

        DLOG(INFO) << "AllocStorage: allocation_size=" << size << "alignment=" << alignment
                   << "dtype_hint=" << DLDataType2String(instr.alloc_storage.dtype_hint);

        auto storage = make_storage(size, alignment, instr.alloc_storage.dtype_hint, ctxs_[0]);
        WriteRegister(instr.dst, storage);
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
  CHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVirtualMachine(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
