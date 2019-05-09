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

#include <tvm/logging.h>
#include <tvm/runtime/vm.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../../runtime/vm/memory_manager.h"
#include "../../runtime/vm/naive_allocator.h"

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
    case Opcode::Select:
      this->select_cond = instr.select_cond;
      this->select_op1 = instr.select_op1;
      this->select_op2 = instr.select_op2;
      return;
    case Opcode::Ret:
      this->result = instr.result;
      return;
    case Opcode::AllocTensor:
      this->shape_register = instr.shape_register;
      this->dtype = instr.dtype;
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
      this->closure_args_num = instr.closure_args_num;
      this->closure_args = Duplicate<RegName>(instr.closure_args, instr.closure_args_num);
      return;
    case Opcode::Invoke:
      this->func_index = instr.func_index;
      this->num_args = instr.num_args;
      this->invoke_args_registers = Duplicate<RegName>(instr.invoke_args_registers, instr.num_args);
      return;
    case Opcode::If:
      this->if_cond = instr.if_cond;
      this->true_offset = instr.true_offset;
      this->false_offset = instr.false_offset;
      return;
    case Opcode::LoadConst:
      this->const_index = instr.const_index;
      return;
    case Opcode::GetField:
      this->object = instr.object;
      this->field_index = instr.field_index;
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

Instruction::~Instruction() {
  switch (this->op) {
    case Opcode::Move:
    case Opcode::Select:
    case Opcode::Ret:
    case Opcode::AllocTensor:
    case Opcode::If:
    case Opcode::LoadConst:
    case Opcode::GetField:
    case Opcode::Goto:
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
      out << "Invalid instruction " << static_cast<int>(this->op);
      throw std::runtime_error(out.str());
  }
}

Instruction Instruction::Ret(RegName result) {
  Instruction instr;
  instr.op = Opcode::Ret;
  instr.result = result;
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

Instruction Instruction::AllocTensor(RegName shape_register, DLDataType dtype, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocTensor;
  instr.dst = dst;
  instr.shape_register = shape_register;
  instr.dtype = dtype;
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

Instruction Instruction::If(RegName cond, Index true_branch, Index false_branch) {
  Instruction instr;
  instr.op = Opcode::If;
  instr.if_cond = cond;
  instr.true_offset = true_branch;
  instr.false_offset = false_branch;
  return instr;
}

Instruction Instruction::Select(RegName cond, RegName op1, RegName op2, RegName dst) {
  Instruction instr;
  instr.op = Opcode::Select;
  instr.dst = dst;
  instr.select_cond = cond;
  instr.select_op1 = op1;
  instr.select_op2 = op2;
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
  instr.closure_args_num = args.size();
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

  os << dtype.bits;
  if (dtype.lanes != 0) {
    os << "[" << dtype.lanes << "]";
  }
}

void InstructionPrint(std::ostream& os, const Instruction& instr) {
  switch (instr.op) {
    case Opcode::Move: {
      os << "move " << instr.from << " " << instr.dst;
      break;
    }
    case Opcode::Ret: {
      os << "ret " << instr.result;
      break;
    }
    case Opcode::InvokePacked: {
      os << "invoke_packed ";
      os << instr.packed_index;
      os << " " << instr.arity;
      os << "(";
      for (Index i = 0; i < instr.arity; ++i) {
        os << instr.packed_args[i] << ",";
      }
      os << ")";
      os << " " << instr.output_size;
      break;
    }
    case Opcode::AllocTensor: {
      os << "alloc_tensor ";
      os << instr.dst << " ";
      os << instr.shape_register << " ";
      DLDatatypePrint(os, instr.dtype);
      break;
    }
    case Opcode::AllocDatatype: {
      os << "alloc_data ";
      os << instr.dst << " ";
      os << instr.constructor_tag << " ";
      os << instr.num_fields;
      break;
    }
    case Opcode::AllocClosure: {
      os << "alloc_closure ";
      os << instr.dst << " ";
      os << instr.clo_index << " ";
      os << instr.num_freevar << "(";
      for (Index i = 0; i < instr.num_freevar; ++i) {
        os << instr.free_vars[i] << ",";
      }
      os << ")";
      break;
    }
    case Opcode::If: {
      os << "if "
         << "$" << instr.if_cond << " " << instr.true_offset << " " << instr.false_offset;
      break;
    }
    case Opcode::Invoke: {
      os << "invoke "
         << "$" << instr.dst << " " << instr.func_index << " " << instr.num_args << "(";
      for (Index i = 0; i < instr.num_args; ++i) {
        os << instr.invoke_args_registers[i] << ",";
      }
      os << ")";
      break;
    }
    case Opcode::InvokeClosure: {
      os << "invoke_closure "
         << "$" << instr.dst << " " << instr.closure << " " << instr.closure_args_num << "()";
      break;
    }
    case Opcode::LoadConst: {
      os << "load_const "
         << "$" << instr.dst << " " << instr.const_index;
      break;
    }
    case Opcode::GetField: {
      os << "get_field " << instr.dst << " " << instr.object << " " << instr.field_index;
      break;
    }
    case Opcode::Goto: {
      os << "goto " << instr.pc_offset;
      break;
    }
    case Opcode::Select: {
      os << "select " << instr.dst << " " << instr.select_cond << " " << instr.select_op1 << " "
         << instr.select_op2;
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
    os << i << ": ";
    InstructionPrint(os, vm_func.instructions[i]);
    os << ";" << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, const VMFunction& vm_func) {
  VMFunctionPrint(os, vm_func);
  return os;
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
  DLOG(INFO) << "===================\nInvoking global " << func.name << " " << args.size()
                  << std::endl;

  PushFrame(func.params, this->pc + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  DLOG(INFO) << "func.params= " << func.params << std::endl;

  code = func.instructions.data();
  pc = 0;
}

Object VirtualMachine::Invoke(const VMFunction& func, const std::vector<Object>& args) {
  DLOG(INFO) << "Executing Function: " << std::endl << func << std::endl;

  InvokeGlobal(func, args);
  Run();
  auto alloc = MemoryManager::Global()->GetAllocator(ctxs[0]);
  DLOG(INFO) << "Memory used: " << alloc->UsedMemory() << " B\n";
  return return_register;
}

Object VirtualMachine::Invoke(const std::string& name, const std::vector<Object>& args) {
  auto func_index = this->global_map_[name];
  DLOG(INFO) << "Invoke Global " << name << " at index " << func_index << std::endl;
  return Invoke(this->functions[func_index], args);
}

void InvokePacked(const PackedFunc& func, Index arg_count, Index output_size,
                  const std::vector<Object>& args) {
  std::vector<TVMValue> values(arg_count);
  std::vector<int> codes(arg_count);
  runtime::TVMArgsSetter setter(values.data(), codes.data());

  for (Index i = 0; i < arg_count; i++) {
    NDArray data = ToNDArray(args[i]);
    setter(i, data);
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arg_count), &rv);
}

void VirtualMachine::Init(const std::vector<TVMContext>& ctxs) { this->ctxs = ctxs; }

inline void VirtualMachine::WriteRegister(Index r, const Object& val) {
  frames.back().register_file[r] = val;
}

inline Object VirtualMachine::ReadRegister(Index r) const {
  return frames.back().register_file[r];
}

void VirtualMachine::Run() {
  CHECK(this->code);
  this->pc = 0;
  Index frame_start = frames.size();
  while (true) {
  main_loop:
    auto const& instr = this->code[this->pc];
    DLOG(INFO) << "\nExecuting(" << pc << "): ";
#if USE_RELAY_DEBUG
    InstructionPrint(std::cout, instr);
#endif  // USE_RELAY_DEBUG

    switch (instr.op) {
      case Opcode::Move: {
        Object from_obj;
        if (instr.from == 0) {
          from_obj = return_register;
        } else {
          from_obj = ReadRegister(instr.from);
        }
        WriteRegister(instr.dst, from_obj);
        pc++;
        goto main_loop;
      }
      case Opcode::LoadConst: {
        WriteRegister(instr.dst, this->constants[instr.const_index]);
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
        InvokePacked(func, arity, instr.output_size, args);
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
        for (Index i = 0; i < instr.closure_args_num; ++i) {
          args.push_back(ReadRegister(instr.closure_args[i]));
        }
        for (auto free_var : closure->free_vars) {
          args.push_back(free_var);
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
      case Opcode::Goto: {
        pc += instr.pc_offset;
        goto main_loop;
      }
      case Opcode::If: {
        // How do we do this efficiently?
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;

        const auto& cond = ReadRegister(instr.if_cond);
        NDArray cpu_array = ToNDArray(cond).CopyTo(cpu_ctx);
        // CHECK_EQ(cpu_array->dtype, Bool());
        bool branch = reinterpret_cast<uint8_t*>(cpu_array->data)[0];

        if (branch) {
          pc += instr.true_offset;
        } else {
          pc += instr.false_offset;
        }

        goto main_loop;
      }
      case Opcode::AllocTensor: {
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;

        auto shape_tensor_obj = ReadRegister(instr.shape_register);
        NDArray shape_tensor = ToNDArray(shape_tensor_obj).CopyTo(cpu_ctx);

        int64_t* dims = static_cast<int64_t*>(shape_tensor->data);
        auto num_dims = shape_tensor->shape[0];
        auto shape = std::vector<int64_t>(shape_tensor->shape[0]);
        shape.assign(dims, dims + num_dims);
        auto allocator = MemoryManager::Global()->GetAllocator(ctxs[0]);
        auto data = allocator->Empty(shape, instr.dtype, ctxs[0]);
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
      case Opcode::Select: {
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;

        auto cond = ReadRegister(instr.select_cond);
        NDArray cpu_array = ToNDArray(cond).CopyTo(cpu_ctx);
        // CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
        bool branch = reinterpret_cast<uint8_t*>(cpu_array->data)[0];

        if (branch) {
          auto op1 = ReadRegister(instr.select_op1);
          WriteRegister(instr.dst, op1);
        } else {
          auto op2 = ReadRegister(instr.select_op2);
          WriteRegister(instr.dst, op2);
        }
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
