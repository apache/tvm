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
 * \file tvm/runtime/vm/executable.h
 * \brief The Relay virtual machine executable.
 */
#ifndef TVM_RUNTIME_VM_EXECUTABLE_H_
#define TVM_RUNTIME_VM_EXECUTABLE_H_

#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/vm/bytecode.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

struct VMFunction;

/*!
 * \brief The executable emitted by the VM compiler.
 *
 * The executable contains information (e.g. data in different memory regions)
 * to run in a virtual machine.
 *
 *  - Global section, containing all globals.
 *  - Constant section, storing the constant pool.
 *  - Primitive name section, containing the function name of the primitive ops
 *  used by the virtual machine.
 *  - Code section, handling the VM functions and bytecode.
 */
class Executable : public ModuleNode {
 public:
  /*!
   * \brief Get a PackedFunc from an executable module.
   *
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc or nullptr when it is not available.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  /*!
   * \brief Serialize the executable into global section, constant section, and
   * code section.
   *
   * \return The binary representation of the VM.
   */
  TVMByteArray Save();

  /*!
   * \brief Load the saved VM executable.
   *
   * \param code The bytecode in string.
   * \param lib The compiled runtime library.
   *
   * \return exe The constructed executable.
   */
  static runtime::Module Load(const std::string& code, const runtime::Module lib);

  /*!
   * \brief Get the serialized form of the `functions`. This is
   * essentially bytecode serialization.
   *
   * \return The serialized vm bytecode.
   *
   * \note The bytecode is in the following format:
   *   func_name reg_file_size num_instructions
   *   param1 param2 ... paramM
   *   instruction1
   *   instruction2
   *   ...
   *   instructionN
   *
   * Each instruction is printed in the following format:
   *   opcode num_fields field1 ... fieldX # The text format.
   *
   * Serializing an `Instruction` requires us to deal with the bytecode. Each line
   * of the instructions could be serialized as the following format:
   *   hash, opcode, f1, f2, ..., fX, field with variable length
   *   1. hash: the hash of the instruction. This number will be used to help us
   * validate if an instruction is well-formed during deserialization.
   *   2. opcode: the opcode code of the instruction.
   *   3. f1, f2, ..., fX. These fields together represent the fixed fields in
   * an instruction, e.g., `from` and `dst` fields of a `Move` instruction. For
   * example, `DLDataType` will be unpacked into three fields (code, bits, lanes).
   *   4. The rest of the line indicates the field with variable length, e.g.,
   * the shape of a tensor, the args used by an `InvokPacked` instruction, etc.

   * The field starting from # is only used for debugging. The serialized code
   * doesn't contain it, therefore the deserializer doens't need to handle it.
   */
  std::string GetBytecode() const;

  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globls and constants, etc.
   */
  std::string Stats() const;

  /*!
   * \brief Get the `lib` module in an executable. Users have the flexibility to call
   * `export_library` from the frontend to save the library to disk.
   *
   * \return The runtime module that contains the hardwre dependent code.
   */
  runtime::Module GetLib() const { return lib; }

  /*!
   * \brief Get the arity of the VM Fucntion.
   * \param func Function name.
   * \return The number of parameters.
   */
  int GetFunctionArity(std::string func) const;

  /*!
   * \brief Get the parameter name given the function name and parameter index.
   * \param func Function name.
   * \param index Parameter index.
   * \return The parameter name.
   */
  std::string GetFunctionParameterName(std::string func, uint32_t index) const;

  virtual ~Executable() {}

  const char* type_key() const final { return "VMExecutable"; }

  /*! \brief The runtime module/library that contains both the host and also the device
   * code when executing on non-CPU devices. */
  runtime::Module lib;
  /*! \brief The global constant pool. */
  std::vector<ObjectRef> constants;
  /*! \brief A map from globals (as strings) to their index in the function map. */
  std::unordered_map<std::string, Index> global_map;
  /*! \brief A mapping from the packed function (as string) to the index that
   * corresponds to the position of the `packed_funcs` list in a `VirtualMachine` object.
   */
  std::unordered_map<std::string, Index> primitive_map;
  /*! \brief The virtual machine's function table. */
  std::vector<VMFunction> functions;
  /*! \brief The device type for each constant. */
  std::vector<Index> const_device_type;

 private:
  /*!
   * \brief Save the globals.
   *
   * \param strm The input stream.
   */
  void SaveGlobalSection(dmlc::Stream* strm);

  /*!
   * \brief Save the constant pool.
   *
   * \param strm The input stream.
   */
  void SaveConstantSection(dmlc::Stream* strm);

  /*!
   * \brief Save primitive op names.
   *
   *  \param strm The input stream.
   */
  void SavePrimitiveOpNames(dmlc::Stream* strm);

  /*!
   * \brief Save the vm functions.
   *
   * \param strm The input stream.
   */
  void SaveCodeSection(dmlc::Stream* strm);

  /*!
   * \brief Load the globals.
   *
   * \param strm The input stream.
   */
  void LoadGlobalSection(dmlc::Stream* strm);

  /*!
   * \brief Load the constant pool.
   *
   * \param strm The input stream.
   */
  void LoadConstantSection(dmlc::Stream* strm);

  /*!
   * \brief Load primitive op names.
   *
   * \param strm The input stream.
   */
  void LoadPrimitiveOpNames(dmlc::Stream* strm);

  /*!
   * \brief Load the vm functions.
   *
   * \param strm The input stream.
   */
  void LoadCodeSection(dmlc::Stream* strm);

  /*! \brief The serialized bytecode. */
  std::string code_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_EXECUTABLE_H_
