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
 * \file src/relay/backend/vm/deserializer.cc
 * \brief Implementation of APIs to deserialize the serialized VM bytecode.
 */

#include "deserializer.h"

#include <tvm/runtime/registry.h>
#include <memory>
#include <sstream>

#include "serialize_util.h"

namespace tvm {
namespace relay {
namespace vm {

#define STREAM_CHECK(val, section)                                         \
  CHECK(val) << "Invalid VM file format in the " << section << " section." \
             << "\n";

void Deserializer::Init(const std::string& code, const runtime::Module& lib) {
  code_ = code;
  vm_ = std::make_shared<VirtualMachine>();
  vm_->lib = lib;
  strm_ = new dmlc::MemoryStringStream(&code_);
}

runtime::PackedFunc Deserializer::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "deserialize") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->Deserialize();
      *rv = runtime::Module(vm_);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void Deserializer::Deserialize() {
  // Check header.
  uint64_t header;
  STREAM_CHECK(strm_->Read(&header), "header");
  STREAM_CHECK(header == kTVMVMBytecodeMagic, "header");

  // Check version.
  std::string version;
  STREAM_CHECK(strm_->Read(&version), "version");
  STREAM_CHECK(version == TVM_VERSION, "version");

  // Global section.
  DeserializeGlobalSection();

  // Constant section.
  DeserializeConstantSection();

  // Primitive names that will be invoked by `InvokePacked` instructions.
  DeserializePrimitiveOpNames();

  // Code section.
  DeserializeCodeSection();
}

void Deserializer::DeserializeGlobalSection() {
  std::vector<std::string> globals;
  STREAM_CHECK(strm_->Read(&globals), "global");
  for (size_t i = 0; i < globals.size(); i++) {
    vm_->global_map.insert({globals[i], i});
  }
}

void Deserializer::DeserializeConstantSection() {
  uint64_t sz;
  // Load the number of constants.
  STREAM_CHECK(strm_->Read(&sz, sizeof(sz)), "constant");

  size_t size = static_cast<size_t>(sz);
  // Load each of the constants.
  for (size_t i = 0; i < size; i++) {
    runtime::NDArray constant;
    STREAM_CHECK(constant.Load(strm_), "constant");
    runtime::Object obj = runtime::Object::Tensor(constant);
    vm_->constants.push_back(obj);
  }
}

void Deserializer::DeserializePrimitiveOpNames() {
  std::vector<std::string> primitive_names;
  STREAM_CHECK(strm_->Read(&primitive_names), "primitive name");
  for (size_t i = 0; i < primitive_names.size(); i++) {
    vm_->primitive_map.insert({primitive_names[i], i});
  }
}

// Extract the `cnt` number of fields started at `start` from the list
// `instr_fields`.
inline std::vector<Index> ExtractFields(const std::vector<Index>& instr_fields,
                                        Index start,
                                        Index cnt) {
  CHECK_LE(static_cast<size_t>(start + cnt), instr_fields.size());
  std::vector<Index> ret;
  for (auto i = start; i < start + cnt; i++) {
    ret.push_back(instr_fields[i]);
  }
  return ret;
}

Instruction DeserializeInstruction(const VMInstructionSerializer& instr) {
  Opcode opcode = static_cast<Opcode>(instr.opcode);
  switch (opcode) {
    case Opcode::Move: {
      // Number of fields = 2
      DCHECK_EQ(instr.fields.size(), 2U);
      return Instruction::Move(instr.fields[0], instr.fields[1]);
    }
    case Opcode::Ret: {
      // Number of fields = 1
      DCHECK_EQ(instr.fields.size(), 1U);
      return Instruction::Ret(instr.fields[0]);
    }
    case Opcode::Fatal: {
      // Number of fields = 0
      DCHECK(instr.fields.empty());
      return Instruction::Fatal();
    }
    case Opcode::InvokePacked: {
      // Number of fields = 3 + instr.arity
      DCHECK_GE(instr.fields.size(), 3U);
      DCHECK_EQ(instr.fields.size(), 3U + static_cast<size_t>(instr.fields[1]));

      Index packed_index = instr.fields[0];
      Index arity = instr.fields[1];
      Index output_size = instr.fields[2];
      std::vector<RegName> args = ExtractFields(instr.fields, 3, arity);
      return Instruction::InvokePacked(packed_index, arity, output_size, args);
    }
    case Opcode::AllocTensor: {
      // Number of fields = 5 + instr.alloc_tensor.ndim
      DCHECK_GE(instr.fields.size(), 5U);
      DCHECK_EQ(instr.fields.size(), 5U + static_cast<size_t>(instr.fields[3]));

      DLDataType dtype;
      dtype.code = instr.fields[0];
      dtype.bits = instr.fields[1];
      dtype.lanes = instr.fields[2];

      Index ndim = instr.fields[3];
      RegName dst = instr.fields[4];

      std::vector<Index> shape = ExtractFields(instr.fields, 5, ndim);

      return Instruction::AllocTensor(shape, dtype, dst);
    }
    case Opcode::AllocTensorReg: {
      // Number of fields = 5
      DCHECK_EQ(instr.fields.size(), 5U);
      Index shape_register = instr.fields[0];

      DLDataType dtype;
      dtype.code = instr.fields[1];
      dtype.bits = instr.fields[2];
      dtype.lanes = instr.fields[3];

      RegName dst = instr.fields[4];

      return Instruction::AllocTensorReg(shape_register, dtype, dst);
    }
    case Opcode::AllocDatatype: {
      // Number of fields = 3 + instr.num_fields
      DCHECK_GE(instr.fields.size(), 3U);
      DCHECK_EQ(instr.fields.size(), 3U + static_cast<size_t>(instr.fields[1]));

      Index constructor_tag = instr.fields[0];
      Index num_fields = instr.fields[1];
      RegName dst = instr.fields[2];
      std::vector<Index> fields = ExtractFields(instr.fields, 3, num_fields);

      return Instruction::AllocDatatype(constructor_tag, num_fields, fields, dst);
    }
    case Opcode::AllocClosure: {
      // Number of fields = 3 + instr.num_freevar
      DCHECK_GE(instr.fields.size(), 3U);
      DCHECK_EQ(instr.fields.size(), 3U + static_cast<size_t>(instr.fields[1]));

      Index clo_index = instr.fields[0];
      Index num_freevar = instr.fields[1];
      RegName dst = instr.fields[2];
      std::vector<Index> free_vars = ExtractFields(instr.fields, 3, num_freevar);

      return Instruction::AllocClosure(clo_index, num_freevar, free_vars, dst);
    }
    case Opcode::If: {
      // Number of fields = 4
      DCHECK_EQ(instr.fields.size(), 4U);
      Index test = instr.fields[0];
      Index target = instr.fields[1];
      Index true_offset = instr.fields[2];
      Index false_offset = instr.fields[3];

      return Instruction::If(test, target, true_offset, false_offset);
    }
    case Opcode::Invoke: {
      // Number of fields = 3 + instr.num_args
      DCHECK_GE(instr.fields.size(), 3U);
      DCHECK_EQ(instr.fields.size(), 3U + static_cast<size_t>(instr.fields[1]));

      Index func_index = instr.fields[0];
      Index num_args = instr.fields[1];
      RegName dst = instr.fields[2];
      std::vector<Index> args = ExtractFields(instr.fields, 3, num_args);

      return Instruction::Invoke(func_index, args, dst);
    }
    case Opcode::InvokeClosure: {
      // Number of fields = 3 + instr.num_closure_args
      DCHECK_GE(instr.fields.size(), 3U);
      DCHECK_EQ(instr.fields.size(), 3U + static_cast<size_t>(instr.fields[1]));

      Index closure = instr.fields[0];
      Index num_closure_args = instr.fields[1];
      RegName dst = instr.fields[2];
      std::vector<Index> args = ExtractFields(instr.fields, 3, num_closure_args);

      return Instruction::InvokeClosure(closure, args, dst);
    }
    case Opcode::LoadConst: {
      // Number of fields = 2
      DCHECK_EQ(instr.fields.size(), 2U);
      return Instruction::LoadConst(instr.fields[0], instr.fields[1]);
    }
    case Opcode::LoadConsti: {
      // Number of fields = 2
      DCHECK_EQ(instr.fields.size(), 2U);
      return Instruction::LoadConsti(instr.fields[0], instr.fields[1]);
    }
    case Opcode::GetField: {
      // Number of fields = 3
      DCHECK_EQ(instr.fields.size(), 3U);
      return Instruction::GetField(instr.fields[0], instr.fields[1], instr.fields[2]);
    }
    case Opcode::GetTag: {
      // Number of fields = 2
      DCHECK_EQ(instr.fields.size(), 2U);
      return Instruction::GetTag(instr.fields[0], instr.fields[1]);
    }
    case Opcode::Goto: {
      // Number of fields = 1
      DCHECK_EQ(instr.fields.size(), 1U);
      return Instruction::Goto(instr.fields[0]);
    }
    default:
      LOG(FATAL) << "Invalid opcode" << instr.opcode;
      return Instruction();
  }
}

void Deserializer::DeserializeCodeSection() {
  // Load the number of functions.
  uint64_t sz;
  STREAM_CHECK(strm_->Read(&sz, sizeof(sz)), "code");

  size_t num_funcs = static_cast<size_t>(sz);
  vm_->functions.resize(num_funcs);
  for (size_t i = 0; i < num_funcs; i++) {
    // Load the function info.
    VMFunctionSerializer loaded_func;
    STREAM_CHECK(loaded_func.Load(strm_), "code/function");

    // Load the instructions.
    std::vector<Instruction> instructions;
    for (size_t j = 0; j < loaded_func.num_instructions; j++) {
      VMInstructionSerializer instr;
      std::vector<Index> instr_fields;
      STREAM_CHECK(instr.Load(strm_), "code/instruction");
      instructions.push_back(DeserializeInstruction(instr));
    }

    // Create the VM function.
    VMFunction vm_func = VMFunction(loaded_func.name,
                                    loaded_func.params,
                                    instructions,
                                    loaded_func.register_file_size);
    auto it = vm_->global_map.find(loaded_func.name);
    CHECK(it != vm_->global_map.end());
    CHECK_LE(it->second, vm_->global_map.size());
    vm_->functions[it->second] = vm_func;
  }
}

runtime::Module CreateDeserializer(const std::string& code, const runtime::Module lib) {
  std::shared_ptr<Deserializer> exec = std::make_shared<Deserializer>();
  exec->Init(code, lib);
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay._vm._Deserializer")
.set_body_typed(CreateDeserializer);

}  // namespace vm
}  // namespace relay
}  // namespace tvm
