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
 * \file tvm/runtime/vm/executable.cc
 * \brief The implementation of a virtual machine executable APIs.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "../file_utils.h"
#include "../library_module.h"
#include "serialize_utils.h"

namespace tvm {
namespace runtime {
namespace vm {

#define STREAM_CHECK(val, section)                                          \
  ICHECK(val) << "Invalid VM file format in the " << section << " section." \
              << "\n";

// Helper to serialize a vm instruction.
VMInstructionSerializer SerializeInstruction(const Instruction& instr);
// Helper to deserialize a serialized vm instruction.
Instruction DeserializeInstruction(const VMInstructionSerializer& instr);

PackedFunc Executable::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_lib") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetLib(); });
  } else if (name == "get_bytecode") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetBytecode(); });
  } else if (name == "get_stats") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->Stats(); });
  } else if (name == "save") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->Save(); });
  } else if (name == "get_function_arity") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      *rv = this->GetFunctionArity(func_name);
    });
  } else if (name == "get_function_param_name") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::string func_name = args[0];
      int index = args[1];
      *rv = this->GetFunctionParameterName(func_name, index);
    });
  } else if (name == "vm_load_executable") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      auto vm = make_object<VirtualMachine>();
      vm->LoadExecutable(this);
      *rv = Module(vm);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc(nullptr);
  }
}

int Executable::GetFunctionArity(std::string func_name) const {
  auto it = global_map.find(func_name);
  if (it == global_map.end()) {
    LOG(ERROR) << "Cannot find function " << func_name << " in executable";
    return -1;
  }
  const auto& func = functions[it->second];
  return func.params.size();
}

std::string Executable::GetFunctionParameterName(std::string func_name, uint32_t index) const {
  auto it = global_map.find(func_name);
  if (it == global_map.end()) {
    LOG(ERROR) << "Cannot find function " << func_name << " in executable";
    return "";
  }
  const auto& func = functions[it->second];
  if (index > func.params.size()) {
    LOG(ERROR) << "Invalid parameter index";
    return "";
  }
  return func.params[index];
}

std::string Executable::GetBytecode() const {
  std::ostringstream oss;

  for (size_t i = 0; i < functions.size(); ++i) {
    const auto& func = functions[i];
    // Print the header of the function format.
    oss << "VM Function[" << i << "]: " << func.name << "(";
    for (const auto& param : func.params) {
      oss << param << ", ";
    }
    oss.seekp(-2, std::ios_base::end);
    oss << ")" << std::endl;
    oss << "# reg file size = " << func.register_file_size << std::endl;
    oss << "# instruction count = " << func.instructions.size() << std::endl;

    // Print the instructions of a `VMFunction`.
    // The part after ";" is the instruction in text format.
    oss << "opcode, fields # inst(text):" << std::endl;
    for (size_t idx = 0; idx < func.instructions.size(); ++idx) {
      const auto& instr = func.instructions[idx];
      const auto& serialized_instr = SerializeInstruction(instr);
      oss << std::setw(2) << idx << ": " << serialized_instr.opcode << " ";
      for (auto it : serialized_instr.fields) {
        oss << it << " ";
      }
      oss << "  # " << instr;
      if (oss.str().back() != '\n') oss << std::endl;
    }
    oss << std::endl;
  }

  return oss.str();
}

std::string Executable::Stats() const {
  std::ostringstream oss;
  oss << "Relay VM executable statistics:" << std::endl;

  // Get the number of constants and the shape of each of them.
  oss << "  Constant shapes (# " << constants.size() << "): [";
  for (const auto& it : constants) {
    const auto constant = Downcast<NDArray>(it);
    const auto& shape = constant.Shape();

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
  if (!constants.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of globals and the name of each of them.
  oss << "  Globals (#" << global_map.size() << "): [";
  for (const auto& it : global_map) {
    oss << "(\"" << it.first << "\", " << it.second << ")"
        << ", ";
  }
  if (!global_map.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  // Get the number of primitive ops and the name of each of them.
  oss << "  Primitive ops (#" << primitive_map.size() << "): [";
  std::vector<std::string> prim_ops;
  for (const auto& it : primitive_map) {
    auto packed_index = static_cast<size_t>(it.second);
    if (prim_ops.size() <= packed_index) {
      prim_ops.resize(packed_index + 1);
    }
    prim_ops[packed_index] = it.first;
  }
  for (const auto& it : prim_ops) {
    oss << it << ", ";
  }
  if (!prim_ops.empty()) oss.seekp(-2, oss.cur);
  oss << "]" << std::endl;

  return oss.str();
}

void SaveHeader(dmlc::Stream* strm) {
  uint64_t header = kTVMVMBytecodeMagic;
  strm->Write(header);
  std::string version = TVM_VERSION;
  strm->Write(version);
}

TVMByteArray Executable::Save() {
  // Initialize the stream object.
  code_.clear();
  dmlc::MemoryStringStream strm(&code_);

  // Save header
  SaveHeader(&strm);

  // Global section.
  SaveGlobalSection(&strm);

  // Constant section.
  SaveConstantSection(&strm);

  // Primitive names.
  SavePrimitiveOpNames(&strm);

  // Code section.
  SaveCodeSection(&strm);

  TVMByteArray arr;
  arr.data = code_.c_str();
  arr.size = code_.length();
  return arr;
}

void Executable::SaveGlobalSection(dmlc::Stream* strm) {
  std::vector<std::pair<std::string, Index> > globals(this->global_map.begin(),
                                                      this->global_map.end());
  auto comp = [](const std::pair<std::string, Index>& a, const std::pair<std::string, Index>& b) {
    return a.second < b.second;
  };
  std::sort(globals.begin(), globals.end(), comp);

  std::vector<std::string> glbs;
  for (const auto& it : globals) {
    glbs.push_back(it.first);
  }
  strm->Write(glbs);
}

void Executable::SaveConstantSection(dmlc::Stream* strm) {
  std::vector<DLTensor*> arrays;
  for (const auto& obj : this->constants) {
    const auto cell = Downcast<runtime::NDArray>(obj);
    arrays.push_back(const_cast<DLTensor*>(cell.operator->()));
  }
  strm->Write(static_cast<uint64_t>(this->constants.size()));
  for (const auto& it : arrays) {
    runtime::SaveDLTensor(strm, it);
  }

  // Save the const to device mapping.
  strm->Write(this->const_device_type);
}

void Executable::SavePrimitiveOpNames(dmlc::Stream* strm) {
  std::vector<std::string> primitive_names;
  for (const auto& it : this->primitive_map) {
    auto packed_index = static_cast<size_t>(it.second);
    if (primitive_names.size() <= packed_index) {
      primitive_names.resize(packed_index + 1);
    }
    primitive_names[packed_index] = it.first;
  }
  strm->Write(primitive_names);
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
// by the `VMInstructionExecutable`. It is used for sanity check before decoding.
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
      // Number of fields = 7 + instr.alloc_tensor.ndim
      fields.push_back(instr.alloc_tensor.storage);
      fields.push_back(instr.alloc_tensor.offset);
      // Save `DLDataType` and the dst register.
      const auto& dtype = instr.alloc_tensor.dtype;
      fields.push_back(dtype.code);
      fields.push_back(dtype.bits);
      fields.push_back(dtype.lanes);

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
      // Number of fields = 7
      fields.push_back(instr.alloc_tensor_reg.storage);
      fields.push_back(instr.alloc_tensor_reg.offset);
      fields.push_back(instr.alloc_tensor_reg.shape_register);
      // Save `DLDataType` and the dst register.
      const auto& dtype = instr.alloc_tensor_reg.dtype;
      fields.push_back(dtype.code);
      fields.push_back(dtype.bits);
      fields.push_back(dtype.lanes);
      fields.push_back(instr.dst);
      break;
    }
    case Opcode::AllocStorage: {
      fields.push_back(instr.alloc_storage.allocation_size);
      fields.push_back(instr.alloc_storage.alignment);
      // Save `DLDataType` and the dst register.
      const auto& dtype = instr.alloc_storage.dtype_hint;
      fields.push_back(dtype.code);
      fields.push_back(dtype.bits);
      fields.push_back(dtype.lanes);
      fields.push_back(instr.alloc_storage.device_type);
      fields.push_back(instr.dst);
      break;
    }
    case Opcode::AllocADT: {
      // Number of fields = 3 + instr.num_fields
      fields.assign({instr.constructor_tag, instr.num_fields, instr.dst});

      // Save the fields.
      fields.insert(fields.end(), instr.datatype_fields, instr.datatype_fields + instr.num_fields);
      break;
    }
    case Opcode::AllocClosure: {
      // Number of fields = 3 + instr.num_freevar
      fields.assign({instr.clo_index, instr.num_freevar, instr.dst});

      // Save the free vars.
      fields.insert(fields.end(), instr.free_vars, instr.free_vars + instr.num_freevar);
      break;
    }
    case Opcode::If: {
      // Number of fields = 4
      fields.assign({instr.if_op.test, instr.if_op.target, instr.if_op.true_offset,
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
      fields.insert(fields.end(), instr.closure_args, instr.closure_args + instr.num_closure_args);
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
    case Opcode::ShapeOf: {
      // Number of fields = 2
      fields.assign({instr.shape_of.tensor, instr.dst});
      break;
    }
    case Opcode::ReshapeTensor: {
      // Number of fields = 3
      fields.assign({instr.reshape_tensor.tensor, instr.reshape_tensor.newshape, instr.dst});
      break;
    }
    case Opcode::DeviceCopy: {
      // Number of fields = 4
      fields.assign({instr.src, instr.src_device_type, instr.dst_device_type, instr.dst});
      break;
    }
    default:
      LOG(FATAL) << "Invalid opcode" << static_cast<int>(instr.op);
      break;
  }

  return VMInstructionSerializer(static_cast<Index>(instr.op), fields);
}

void Executable::SaveCodeSection(dmlc::Stream* strm) {
  // Save the number of functions.
  strm->Write(static_cast<uint64_t>(this->functions.size()));
  for (const auto& func : this->functions) {
    // Save the function info.
    VMFunctionSerializer func_format(func.name, func.register_file_size, func.instructions.size(),
                                     func.params, func.params_device_type);
    func_format.Save(strm);

    // Serialize each instruction.
    for (const auto& instr : func.instructions) {
      const auto& serialized_instr = SerializeInstruction(instr);
      serialized_instr.Save(strm);
    }
  }
}

void LoadHeader(dmlc::Stream* strm) {
  // Check header.
  uint64_t header;
  STREAM_CHECK(strm->Read(&header), "header");
  STREAM_CHECK(header == kTVMVMBytecodeMagic, "header");

  // Check version.
  std::string version;
  STREAM_CHECK(strm->Read(&version), "version");
  STREAM_CHECK(version == TVM_VERSION, "version");
}

runtime::Module Executable::GetLib() const {
  ICHECK_LE(this->imports_.size(), 1)
      << "The kernel library must be imported as the only module in an Executable";

  if (this->imports().size() == 0) {
    return Module(nullptr);
  } else {
    return this->imports_[0];
  }
}

void Executable::SetLib(const runtime::Module& lib) {
  ICHECK(lib.defined()) << "the provided library can not be null";

  ICHECK_EQ(this->imports_.size(), 0)
      << "A VMExecutable should never have more than one import inside an the executable, \n"
      << "the first import should *always* be the library containing"
      << "the platform specific kernel code";

  this->Import(lib);
}

runtime::Module Executable::Load(const std::string& code, const runtime::Module lib) {
  auto exec = make_object<Executable>();

  // Support null-initialization of lib, to enable initialization during
  // deserialization before we have we have deserialized the imports.
  if (lib.defined()) {
    exec->SetLib(lib);
  }

  exec->code_ = code;
  dmlc::MemoryStringStream strm(&exec->code_);

  // Load header.
  LoadHeader(&strm);

  // Global section.
  exec->LoadGlobalSection(&strm);

  // Constant section.
  exec->LoadConstantSection(&strm);

  // Primitive names that will be invoked by `InvokePacked` instructions.
  exec->LoadPrimitiveOpNames(&strm);

  // Code section.
  exec->LoadCodeSection(&strm);

  return runtime::Module(exec);
}

void Executable::LoadGlobalSection(dmlc::Stream* strm) {
  std::vector<std::string> globals;
  STREAM_CHECK(strm->Read(&globals), "global");
  for (size_t i = 0; i < globals.size(); i++) {
    this->global_map.insert({globals[i], i});
  }
}

void Executable::LoadConstantSection(dmlc::Stream* strm) {
  uint64_t sz;
  // Load the number of constants.
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "constant");

  size_t size = static_cast<size_t>(sz);
  // Load each of the constants.
  for (size_t i = 0; i < size; i++) {
    runtime::NDArray constant;
    STREAM_CHECK(constant.Load(strm), "constant");
    this->constants.push_back(constant);
  }

  // Load the const to device mapping.
  std::vector<Index> const_device_type;
  STREAM_CHECK(strm->Read(&const_device_type), "constant");
  ICHECK_EQ(size, const_device_type.size());
  this->const_device_type = const_device_type;
}

void Executable::LoadPrimitiveOpNames(dmlc::Stream* strm) {
  std::vector<std::string> primitive_names;
  STREAM_CHECK(strm->Read(&primitive_names), "primitive name");
  for (size_t i = 0; i < primitive_names.size(); i++) {
    this->primitive_map.insert({primitive_names[i], i});
  }
}

// Extract the `cnt` number of fields started at `start` from the list
// `instr_fields`.
inline std::vector<Index> ExtractFields(const std::vector<Index>& instr_fields, Index start,
                                        Index cnt) {
  ICHECK_LE(static_cast<size_t>(start + cnt), instr_fields.size());
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
      // Number of fields = 7 + instr.alloc_tensor.ndim
      DCHECK_GE(instr.fields.size(), 7U);
      DCHECK_EQ(instr.fields.size(), 7U + static_cast<size_t>(instr.fields[5]));

      RegName storage_reg = instr.fields[0];
      RegName offset = instr.fields[1];

      DLDataType dtype;
      dtype.code = instr.fields[2];
      dtype.bits = instr.fields[3];
      dtype.lanes = instr.fields[4];

      Index ndim = instr.fields[5];
      RegName dst = instr.fields[6];

      std::vector<Index> shape = ExtractFields(instr.fields, 7, ndim);

      return Instruction::AllocTensor(storage_reg, offset, shape, dtype, dst);
    }
    case Opcode::AllocTensorReg: {
      // Number of fields = 7
      DCHECK_EQ(instr.fields.size(), 7U);

      RegName storage_reg = instr.fields[0];
      RegName offset = instr.fields[1];
      Index shape_register = instr.fields[2];

      DLDataType dtype;
      dtype.code = instr.fields[3];
      dtype.bits = instr.fields[4];
      dtype.lanes = instr.fields[5];

      RegName dst = instr.fields[6];

      return Instruction::AllocTensorReg(storage_reg, offset, shape_register, dtype, dst);
    }
    case Opcode::AllocADT: {
      // Number of fields = 3 + instr.num_fields
      DCHECK_GE(instr.fields.size(), 3U);
      DCHECK_EQ(instr.fields.size(), 3U + static_cast<size_t>(instr.fields[1]));

      Index constructor_tag = instr.fields[0];
      Index num_fields = instr.fields[1];
      RegName dst = instr.fields[2];
      std::vector<Index> fields = ExtractFields(instr.fields, 3, num_fields);

      return Instruction::AllocADT(constructor_tag, num_fields, fields, dst);
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
    case Opcode::AllocStorage: {
      // Number of fields = 7
      DCHECK_GE(instr.fields.size(), 7U);
      Index allocation_size = instr.fields[0];
      Index alignment = instr.fields[1];

      DLDataType dtype;
      dtype.code = instr.fields[2];
      dtype.bits = instr.fields[3];
      dtype.lanes = instr.fields[4];

      Index device_type = instr.fields[5];
      RegName dst = instr.fields[6];

      return Instruction::AllocStorage(allocation_size, alignment, dtype, device_type, dst);
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
    case Opcode::ShapeOf: {
      // Number of fields = 2
      DCHECK_EQ(instr.fields.size(), 2U);
      return Instruction::ShapeOf(instr.fields[0], instr.fields[1]);
    }
    case Opcode::ReshapeTensor: {
      // Number of fields = 3
      DCHECK_EQ(instr.fields.size(), 3U);
      return Instruction::ReshapeTensor(instr.fields[0], instr.fields[1], instr.fields[2]);
    }
    case Opcode::DeviceCopy: {
      // Number of fields = 4
      DCHECK_EQ(instr.fields.size(), 4U);
      return Instruction::DeviceCopy(instr.fields[0], instr.fields[1], instr.fields[2],
                                     instr.fields[3]);
    }
    default:
      LOG(FATAL) << "Invalid opcode" << instr.opcode;
      return Instruction();
  }
}

void Executable::LoadCodeSection(dmlc::Stream* strm) {
  // Load the number of functions.
  uint64_t sz;
  STREAM_CHECK(strm->Read(&sz, sizeof(sz)), "code");

  size_t num_funcs = static_cast<size_t>(sz);
  this->functions.resize(num_funcs);
  for (size_t i = 0; i < num_funcs; i++) {
    // Load the function info.
    VMFunctionSerializer loaded_func;
    STREAM_CHECK(loaded_func.Load(strm), "code/function");

    // Load the instructions.
    std::vector<Instruction> instructions;
    for (size_t j = 0; j < loaded_func.num_instructions; j++) {
      VMInstructionSerializer instr;
      std::vector<Index> instr_fields;
      STREAM_CHECK(instr.Load(strm), "code/instruction");
      instructions.push_back(DeserializeInstruction(instr));
    }

    // Create the VM function.
    VMFunction vm_func = VMFunction(loaded_func.name, loaded_func.params, instructions,
                                    loaded_func.register_file_size, loaded_func.params_device_type);
    auto it = this->global_map.find(loaded_func.name);
    ICHECK(it != this->global_map.end());
    ICHECK_LE(it->second, this->global_map.size());
    this->functions[it->second] = vm_func;
  }
}

void Executable::SaveToBinary(dmlc::Stream* stream) {
  auto code_bytes = this->Save();
  std::string code(code_bytes.data, code_bytes.size);
  stream->Write(code);

  ICHECK(this->imports()[0].defined()) << "the library must be imported before serialization";
}

Module ExecutableLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string code;
  stream->Read(&code);
  auto exec = Executable::Load(code, Module());
  return exec;
}

void Executable::SaveToFile(const std::string& path, const std::string& format) {
  std::string data;
  dmlc::MemoryStringStream writer(&data);
  dmlc::SeekStream* strm = &writer;
  SaveToBinary(strm);
  SaveBinaryToFile(path, data);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_VMExecutable").set_body_typed(ExecutableLoadBinary);

// Load module from module.
Module ExecutableLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  LoadBinaryFromFile(file_name, &data);
  dmlc::MemoryStringStream reader(&data);
  dmlc::Stream* strm = &reader;
  auto exec = ExecutableLoadBinary(reinterpret_cast<void*>(strm));
  return exec;
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_VMExecutable").set_body_typed(ExecutableLoadFile);

TVM_REGISTER_GLOBAL("runtime.GetNumOfGlobals").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  *rv = static_cast<int>(exec->global_map.size());
});

TVM_REGISTER_GLOBAL("runtime.GetGlobalFields").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  int idx = args[1];
  std::vector<std::pair<std::string, Index> > globals(exec->global_map.begin(),
                                                      exec->global_map.end());
  auto comp = [](const std::pair<std::string, Index>& a, const std::pair<std::string, Index>& b) {
    return a.second < b.second;
  };
  std::sort(globals.begin(), globals.end(), comp);
  ICHECK_LT(idx, globals.size());
  *rv = globals[idx].first;
});

TVM_REGISTER_GLOBAL("runtime.GetNumOfPrimitives").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  *rv = static_cast<int>(exec->primitive_map.size());
});

TVM_REGISTER_GLOBAL("runtime.GetPrimitiveFields").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec);
  int idx = args[1];
  ICHECK_GE(idx, 0);
  ICHECK_LT(idx, exec->primitive_map.size());

  for (const auto& it : exec->primitive_map) {
    if (idx == static_cast<int>(it.second)) {
      *rv = it.first;
      break;
    }
  }
});

TVM_REGISTER_GLOBAL("runtime.Load_Executable")
    .set_body_typed([](std::string code, runtime::Module lib) {
      return Executable::Load(code, lib);
    });

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
