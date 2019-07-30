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
 * \file src/relay/backend/vm/serializer.h
 * \brief Define a serializer for the Relay VM.
 *
 * The following components of a Relay VM will be serialized:
 *  - The `constants`, e.g., the constant pool, that contains the
 *  constants used in a Relay program.
 *  - The `packed_funcs` that essentially contains the generated code for
 *  a specific target. We return it as a runtime module that can be exported as
 *  a library file (e.g., .so, .o, or .tar).
 *  - The `global_map` that contains the globals.
 *  - The `primitive_map` that contains the name of individual primitive operators.
 *  - The `functions`, e.g., the `VMFunction`. Each `VMFunction` is composed of
 *  a list of instructions/bytecode.
 *
 * Note that only the library is returned as a separate module. All othere parts
 * are stored in a single serialized code that is organized with the following
 * sections in order.
 *  - Global section, containing all globals.
 *  - Constant section, storing the constant pool.
 *  - Primitive name section, containing the function name of the primitive ops
 *  used by the virtual machine.
 *  - Code section, handling the VM functions and bytecode.
 *
 * The code section is again organized as follows for each VM function:
 *   func_name, register_file_size, num_instructions (N)
 *   param1, param2, ..., paramM
 *   instruction1
 *   instruction2
 *   ...
 *   instructionN
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
 */

#ifndef TVM_RELAY_BACKEND_VM_SERIALIZER_H_
#define TVM_RELAY_BACKEND_VM_SERIALIZER_H_

#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <tvm/ir.h>
#include <tvm/node/container.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/vm.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;

/*!
 * \brief The Relay VM serializer.
 */
class Serializer : public runtime::ModuleNode {
 public:
  /*!
   * \brief Initialize the serializer for a virtual machine.
   *
   *  \param vm The Relay virtual machine.
   */
  inline void Init(const VirtualMachine* vm);

  /*!
   * \brief Return the member function to the frontend.
   *
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   *
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  const char* type_key() const final { return "Serializer"; }

  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globls and constants, etc.
   */
  std::string Stats() const;

  /*!
   * \brief Serialize the `vm_` into global section, constant section, and code
   * section.
   *
   * \return The binary representation of the VM.
   */
  TVMByteArray Serialize();

  /*!
   * \brief Get a list of the globals used by the `_vm`.
   *
   * \return The global map in the form a list.
   */
  tvm::Array<tvm::Expr> GetGlobals() const;

  /*!
   * \brief Get the primitive operators that are contained in the Relay VM.
   *
   * \return The list of primitve operators.
   */
  tvm::Array<tvm::Expr> GetPrimitiveOps() const;

  /*!
   * \brief Get the serialized form of the `functions` in `vm_`. This is
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
   * The field starting from # is only used for debugging. The serialized code
   * doesn't contain it, therefore the deserializer doens't need to handle it.
   */
  std::string GetBytecode() const;

  /*! \brief Get the `lib` module in vm_. Serialization of `runtime::module`
   * has already been supported by TVM. Therefore, we only return the runtime
   * module and let users have the flexibility to call `export_library` from
   * the frontend to save the library to disk.
   *
   * \return The runtime module that contains the hardwre dependent code.
   */
  inline runtime::Module GetLib() const;

  virtual ~Serializer() { delete strm_; }

 private:
  /*! \brief Serialize the globals in vm_. */
  void SerializeGlobalSection();

  /*! \brief Serialize the constant pool in vm_. */
  void SerializeConstantSection();

  /*! \brief Serialize primitive op names in vm_. */
  void SerializePrimitiveOpNames();

  /*! \brief Serialize the vm functions in vm_. */
  void SerializeCodeSection();

  /*! \brief The Relay virtual machine for to be serialized. */
  const VirtualMachine* vm_;

  /*! \brief The stream used for serialization. */
  dmlc::Stream* strm_;

  /*! \brief The serialized code. */
  std::string code_;
};

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_VM_SERIALIZER_H_
