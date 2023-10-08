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
 * \file src/runtime/vm/bytecode.cc
 * \brief The bytecode for Relay virtual machine.
 */

#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/vm/bytecode.h>

#include <sstream>

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
      this->device_index = instr.device_index;
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
      this->alloc_storage.allocation_size = instr.alloc_storage.allocation_size;
      this->alloc_storage.alignment = instr.alloc_storage.alignment;
      this->alloc_storage.dtype_hint = instr.alloc_storage.dtype_hint;
      this->alloc_storage.device_index = instr.alloc_storage.device_index;
      this->alloc_storage.ndim = instr.alloc_storage.ndim;
      if (this->alloc_storage.ndim > 0) {
        this->alloc_storage.shape =
            Duplicate<int64_t>(instr.alloc_storage.shape, instr.alloc_storage.ndim);
      }
      return;
    case Opcode::ShapeOf:
      this->shape_of.tensor = instr.shape_of.tensor;
      return;
    case Opcode::ReshapeTensor:
      this->reshape_tensor = instr.reshape_tensor;
      return;
    case Opcode::DeviceCopy:
      this->device_copy = instr.device_copy;
      return;
    case Opcode::KillRegister:
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
      this->device_index = instr.device_index;
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
      this->alloc_storage.allocation_size = instr.alloc_storage.allocation_size;
      this->alloc_storage.alignment = instr.alloc_storage.alignment;
      this->alloc_storage.dtype_hint = instr.alloc_storage.dtype_hint;
      this->alloc_storage.device_index = instr.alloc_storage.device_index;
      this->alloc_storage.ndim = instr.alloc_storage.ndim;
      if (this->alloc_storage.ndim > 0) {
        this->alloc_storage.shape =
            Duplicate<int64_t>(instr.alloc_storage.shape, instr.alloc_storage.ndim);
      }
      return *this;
    case Opcode::ShapeOf:
      this->shape_of.tensor = instr.shape_of.tensor;
      return *this;
    case Opcode::ReshapeTensor:
      this->reshape_tensor = instr.reshape_tensor;
      return *this;
    case Opcode::DeviceCopy:
      this->device_copy = instr.device_copy;
      return *this;
    case Opcode::KillRegister:
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
    case Opcode::ShapeOf:
    case Opcode::ReshapeTensor:
    case Opcode::DeviceCopy:
    case Opcode::Fatal:
    case Opcode::KillRegister:
      return;
    case Opcode::AllocStorage:
      if (this->alloc_storage.ndim > 0) {
        delete[] this->alloc_storage.shape;
      }
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
                                     RegName dst) {
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
                                        DLDataType dtype, RegName dst) {
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
                                      Index device_index, const std::vector<int64_t>& shape,
                                      RegName dst) {
  Instruction instr;
  instr.op = Opcode::AllocStorage;
  instr.dst = dst;
  instr.alloc_storage.allocation_size = size;
  instr.alloc_storage.alignment = alignment;
  instr.alloc_storage.dtype_hint = dtype_hint;
  instr.alloc_storage.device_index = device_index;
  instr.alloc_storage.ndim = static_cast<uint32_t>(shape.size());
  if (instr.alloc_storage.ndim > 0) {
    instr.alloc_storage.shape = new int64_t[shape.size()];
    for (size_t i = 0; i < shape.size(); ++i) {
      instr.alloc_storage.shape[i] = shape[i];
    }
  }
  return instr;
}

Instruction Instruction::ShapeOf(RegName tensor, RegName dst) {
  Instruction instr;
  instr.op = Opcode::ShapeOf;
  instr.dst = dst;
  instr.shape_of.tensor = tensor;
  return instr;
}

Instruction Instruction::ReshapeTensor(RegName tensor, RegName newshape, RegName dst) {
  Instruction instr;
  instr.op = Opcode::ReshapeTensor;
  instr.dst = dst;
  instr.reshape_tensor.tensor = tensor;
  instr.reshape_tensor.newshape = newshape;
  return instr;
}

Instruction Instruction::DeviceCopy(RegName src, Index src_device_index, Index dst_device_index,
                                    RegName dst) {
  Instruction instr;
  instr.op = Opcode::DeviceCopy;
  instr.dst = dst;
  instr.device_copy.src = src;
  instr.device_copy.src_device_index = src_device_index;
  instr.device_copy.dst_device_index = dst_device_index;
  return instr;
}

Instruction Instruction::KillRegister(RegName dst) {
  Instruction instr;
  instr.op = Opcode::KillRegister;
  instr.dst = dst;
  return instr;
}

Instruction Instruction::AllocADT(Index tag, Index num_fields,
                                  const std::vector<RegName>& datatype_fields, RegName dst) {
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
                                      const std::vector<RegName>& free_var_register, RegName dst) {
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

Instruction Instruction::LoadConst(Index const_index, Index device_index, RegName dst) {
  Instruction instr;
  instr.op = Opcode::LoadConst;
  instr.dst = dst;
  instr.const_index = const_index;
  instr.device_index = device_index;
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
    case kDLBfloat:
      os << "bfloat";
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
      os << "load_const $" << instr.dst << " Const[" << instr.const_index << "] "
         << instr.device_index;
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
      os << "alloc_storage $" << instr.dst << " ";
      if (instr.alloc_storage.ndim > 0) {
        os << "[" << StrJoin<int64_t>(instr.alloc_storage.shape, 0, instr.alloc_storage.ndim)
           << "] ";
      } else {
        os << "$" << instr.alloc_storage.allocation_size << " " << instr.alloc_storage.alignment
           << " ";
      }
      os << DLDataType2String(instr.alloc_storage.dtype_hint) << " "
         << instr.alloc_storage.device_index;
      break;
    }
    case Opcode::ShapeOf: {
      os << "shape_of $" << instr.dst << " $" << instr.shape_of.tensor;
      break;
    }
    case Opcode::ReshapeTensor: {
      os << "reshape_tensor $" << instr.dst << " $" << instr.reshape_tensor.tensor << " $"
         << instr.reshape_tensor.newshape;
      break;
    }
    case Opcode::DeviceCopy: {
      os << "device_copy $" << instr.dst << " $" << instr.device_copy.src << " "
         << instr.device_copy.dst_device_index << " " << instr.device_copy.src_device_index;
      break;
    }
    case Opcode::KillRegister: {
      os << "kill_register $" << instr.dst;
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

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
