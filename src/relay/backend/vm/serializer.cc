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
 * \file src/relay/backend/vm/serializer.cc
 * \brief Implementation of serializing APIs for the Relay VM.
 */
#include "serializer.h"

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "serialize_util.h"

namespace tvm {
namespace relay {
namespace vm {

void Serializer::Init(const VirtualMachine* vm) {
  vm_ = vm;
  // Initialize the stream object.
  strm_ = new dmlc::MemoryStringStream(&code_);
}

runtime::PackedFunc Serializer::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "get_lib") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetLib();
    });
  } else if (name == "get_primitive_ops") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetPrimitiveOps();
    });
  } else if (name == "get_bytecode") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetBytecode();
    });
  } else if (name == "get_globals") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetGlobals();
    });
  } else if (name == "get_stats") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->Stats();
    });
  } else if (name == "serialize") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->Serialize();
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

tvm::Array<tvm::Expr> Serializer::GetPrimitiveOps() const {
  std::vector<tvm::Expr> ret;
  for (const auto& it : vm_->primitive_map) {
    auto packed_name = tvm::ir::StringImm::make(it.first);
    auto packed_index = static_cast<size_t>(it.second);
    if (ret.size() <= packed_index) {
      ret.resize(packed_index + 1);
    }
    ret[packed_index] = packed_name;
  }
  return ret;
}

std::string Serializer::Stats() const {
  std::ostringstream oss;
  oss << "Relay VM statistics:" << std::endl;

  // Get the number of constants and the shape of each of them.
  oss << "  Constant shapes (# " << vm_->constants.size() << "): [";
  for (const auto& it : vm_->constants) {
    auto cell = it.AsTensor();
    CHECK(cell.operator->());
    runtime::NDArray data = cell->data;
    const auto& shape = data.Shape();

    // Scalar
    if (shape.empty()) {
      oss << "scalar, ";
      continue;
    }

    oss << "[";
    for (auto s : shape) {
      oss << s << ", ";
    }
    oss.seekp(-2, oss.cur);
    oss << "], " << std::endl;
  }
  if (!vm_->constants.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of globals and the name of each of them.
  oss << "  Globals (#" << vm_->global_map.size() << "): [";
  for (const auto& it : vm_->global_map) {
    oss << "(\"" << it.first << "\", " << it.second << ")" << ", ";
  }
  if (!vm_->global_map.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of primitive ops and the name of each of them.
  oss << "  Primitive ops (#" << vm_->primitive_map.size() << "): [";
  const auto& prim_ops = GetPrimitiveOps();
  for (const auto& it : prim_ops) {
    oss << it << ", ";
  }
  if (!prim_ops.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  return oss.str();
}

TVMByteArray Serializer::Serialize() {
  uint64_t header = kTVMVMBytecodeMagic;
  strm_->Write(header);
  std::string version = TVM_VERSION;
  strm_->Write(version);

  // Global section.
  SerializeGlobalSection();

  // Constant section.
  SerializeConstantSection();

  // Primitive names.
  SerializePrimitiveOpNames();

  // Code section.
  SerializeCodeSection();

  TVMByteArray arr;
  arr.data = code_.c_str();
  arr.size = code_.length();
  return arr;
}

void Serializer::SerializeGlobalSection() {
  auto globals = GetGlobals();
  std::vector<std::string> glbs;
  for (const auto& it : globals) {
    glbs.push_back(it.as<tvm::ir::StringImm>()->value);
  }
  strm_->Write(glbs);
}

void Serializer::SerializeConstantSection() {
  std::vector<DLTensor*> arrays;
  for (const auto& obj : vm_->constants) {
    auto cell = obj.AsTensor();
    runtime::NDArray data = cell->data;
    arrays.push_back(const_cast<DLTensor*>(data.operator->()));
  }
  strm_->Write(static_cast<uint64_t>(vm_->constants.size()));
  for (const auto& it : arrays) {
    runtime::SaveDLTensor(strm_, it);
  }
}

void Serializer::SerializePrimitiveOpNames() {
  auto names = GetPrimitiveOps();
  std::vector<std::string> primitive_names;
  for (const auto& it : names) {
    primitive_names.push_back(it.as<tvm::ir::StringImm>()->value);
  }
  strm_->Write(primitive_names);
}

// Serialize a virtual machine instruction. It creates a list that contains the
// hash, opcode, and all fields of an instruction.
//
// For example, the function signature used to create an `AllocTensor`
// instruction is:
//   Instruction AllocTensor(std::vector<Index> shape, DLDataType dtype, RegName dst)
//
// The serialized form will be:
//   `hash 5 dtype.code dtype.bits dtype.lanes ndim dst_register val1 val2 ... valn`
//
// where hash is the hash of serialized instruction that is computed internally
// by the `VMInstructionSerializer`. It is used for sanity check before decoding.
// 5 shows opcode of `AllocTensor`, `(dtype.code dtype.bits dtype.lanes)`
// represents a `DLDataType`, `ndim` is the number of dimensions, `dst_register`
// is the destination register, and the rest of it together indicates the shape
// of the tensor to be allocated.
VMInstructionSerializer SerializeInstruction(const Instruction& instr) {
  std::vector<Index> fields;
  // Save the opcode.
  DLOG(INFO) << "Serializing: " << instr << std::endl;
  switch (instr.op) {
    case Opcode::Move: {
      // Number of fields = 2
      fields.assign({instr.from, instr.dst});
      break;
    }
    case Opcode::Ret: {
      // Number of fields = 1
      fields.push_back(instr.result);
      break;
    }
    case Opcode::Fatal: {
      // Number of fields = 0
      break;
    }
    case Opcode::InvokePacked: {
      // Number of fields = 3 + instr.arity
      // Note that arity includes both input arguments and outputs. We will
      // put all the `arity` number of fields in the end for serialization.
      fields.assign({instr.packed_index, instr.arity, instr.output_size});
      // Save the args.
      fields.insert(fields.end(), instr.packed_args, instr.packed_args + instr.arity);
      break;
    }
    case Opcode::AllocTensor: {
      // Number of fields = 5 + instr.alloc_tensor.ndim
      // Save `DLDataType` and the dst register.
      const auto& dtype = instr.alloc_tensor.dtype;
      fields.assign({dtype.code, dtype.bits, dtype.lanes});

      // The number of dimensions is not needed for constructing an
      // `AllocTensor` instruction as it equals to the length of the `shape`
      // vector. However, we save it to conveniently deserialize the instruction
      // because we will know how many fields are needed by the `shape` argument.
      fields.push_back(instr.alloc_tensor.ndim);
      fields.push_back(instr.dst);

      // Save the shape of the tensor.
      // Note that this field is rotated to the end of the list.
      fields.insert(fields.end(), instr.alloc_tensor.shape,
                    instr.alloc_tensor.shape + instr.alloc_tensor.ndim);
      break;
    }
    case Opcode::AllocTensorReg: {
      // Number of fields = 5
      fields.push_back(instr.alloc_tensor_reg.shape_register);
      // Save `DLDataType` and the dst register.
      const auto& dtype = instr.alloc_tensor.dtype;
      fields.assign({dtype.code, dtype.bits, dtype.lanes});
      fields.push_back(instr.dst);
      break;
    }
    case Opcode::AllocDatatype: {
      // Number of fields = 3 + instr.num_fields
      fields.assign({instr.constructor_tag, instr.num_fields, instr.dst});

      // Save the fields.
      fields.insert(fields.end(), instr.datatype_fields,
                    instr.datatype_fields + instr.num_fields);
      break;
    }
    case Opcode::AllocClosure: {
      // Number of fields = 3 + instr.num_freevar
      fields.assign({instr.clo_index, instr.num_freevar, instr.dst});

      // Save the free vars.
      fields.insert(fields.end(), instr.free_vars,
                    instr.free_vars + instr.num_freevar);
      break;
    }
    case Opcode::If: {
      // Number of fields = 4
      fields.assign({instr.if_op.test,
                     instr.if_op.target,
                     instr.if_op.true_offset,
                     instr.if_op.false_offset});
      break;
    }
    case Opcode::Invoke: {
      // Number of fields = 3 + instr.num_args
      fields.assign({instr.func_index, instr.num_args, instr.dst});

      // Save the args.
      fields.insert(fields.end(), instr.invoke_args_registers,
                    instr.invoke_args_registers + instr.num_args);
      break;
    }
    case Opcode::InvokeClosure: {
      // Number of fields = 3 + instr.num_closure_args
      fields.assign({instr.closure, instr.num_closure_args, instr.dst});

      // Save the args.
      fields.insert(fields.end(), instr.closure_args,
                    instr.closure_args + instr.num_closure_args);
      break;
    }
    case Opcode::LoadConst: {
      // Number of fields = 2
      fields.assign({instr.const_index, instr.dst});
      break;
    }
    case Opcode::LoadConsti: {
      // Number of fields = 2
      fields.assign({instr.load_consti.val, instr.dst});
      break;
    }
    case Opcode::GetField: {
      // Number of fields = 3
      fields.assign({instr.object, instr.field_index, instr.dst});
      break;
    }
    case Opcode::GetTag: {
      // Number of fields = 2
      fields.assign({instr.get_tag.object, instr.dst});
      break;
    }
    case Opcode::Goto: {
      // Number of fields = 1
      fields.push_back(instr.pc_offset);
      break;
    }
    default:
      LOG(FATAL) << "Invalid opcode" << static_cast<int>(instr.op);
      break;
  }

  return VMInstructionSerializer(static_cast<Index>(instr.op), fields);
}

void Serializer::SerializeCodeSection() {
  // Save the number of functions.
  strm_->Write(static_cast<uint64_t>(vm_->functions.size()));
  for (const auto& func : vm_->functions) {
    // Serialize the function info.
    VMFunctionSerializer func_format(func.name,
                                     func.register_file_size,
                                     func.instructions.size(),
                                     func.params);
    func_format.Save(strm_);

    // Serialize each instruction.
    for (const auto& instr : func.instructions) {
      const auto& serialized_instr = SerializeInstruction(instr);
      serialized_instr.Save(strm_);
    }
  }
}

tvm::Array<tvm::Expr> Serializer::GetGlobals() const {
  tvm::Array<tvm::Expr> ret;
  std::vector<std::pair<std::string, Index> > globals(vm_->global_map.begin(),
                                                      vm_->global_map.end());
  auto comp = [](const std::pair<std::string, Index>& a,
                 const std::pair<std::string, Index>& b) {
    return a.second < b.second;
  };
  std::sort(globals.begin(), globals.end(), comp);
  for (const auto& it : globals) {
    ret.push_back(tvm::ir::StringImm::make(it.first));
  }
  return ret;
}

std::string Serializer::GetBytecode() const {
  std::ostringstream oss;

  for (const auto& func : vm_->functions) {
    // Print the header of the function format.
    oss << "# func name, reg file size, param count, inst count:"
        << std::endl;
    oss << func.name << " "
        << func.register_file_size << " "
        << func.params.size() << " "
        << func.instructions.size() << std::endl;

    // Print pramams of a `VMFunction`.
    oss << "# Parameters:"<< std::endl;
    for (const auto& param : func.params) {
      oss << param << " ";
    }
    oss << std::endl;

    // Print the instructions of a `VMFunction`.
    // The part after ";" is the instruction in text format.
    oss << "hash, opcode, fields # inst(text):"<< std::endl;
    for (const auto& instr : func.instructions) {
      const auto& serialized_instr = SerializeInstruction(instr);
      oss << std::hex << "0x" << serialized_instr.Hash() << " "
          << std::dec << serialized_instr.opcode << " ";
      for (auto it : serialized_instr.fields) {
        oss << it << " ";
      }
      oss << "  # " << instr;
      if (oss.str().back() != '\n') oss << std::endl;
    }
  }

  return oss.str();
}

runtime::Module Serializer::GetLib() const {
  return vm_->lib;
}

runtime::Module CreateSerializer(const VirtualMachine* vm) {
  std::shared_ptr<Serializer> exec = std::make_shared<Serializer>();
  exec->Init(vm);
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay._vm._Serializer")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* vm = dynamic_cast<VirtualMachine*>(mod.operator->());
  CHECK(vm) << "Virtual machine has not been defined yet."
            << "\n";
  *rv = CreateSerializer(vm);
});

}  // namespace vm
}  // namespace relay
}  // namespace tvm
