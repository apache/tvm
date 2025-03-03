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
 * \file tvm/runtime/relax_vm/executable.h
 */
#ifndef TVM_RUNTIME_RELAX_VM_EXECUTABLE_H_
#define TVM_RUNTIME_RELAX_VM_EXECUTABLE_H_

#include <tvm/runtime/container/closure.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "./bytecode.h"

// Convention: this version should set to minimum TVM version it support
// NOTE: this file only changes if we change relax vm format
// for example if relax vm format do not change in 0.15, this should remain as 0.14
// if it changes in 0.16, we will change it to 0.16
#define RELAX_VM_VERSION "0.14"

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief Information entry in executable function table.
 *
 * Contains metadata about the compiled function, as
 * well as the compiled VM instructions.
 */
struct VMFuncInfo {
  /*! \brief kind of the function. */
  enum class FuncKind : int {
    /*! \brief system level packed function */
    kPackedFunc = 0,
    /*! \brief VM function. */
    kVMFunc = 1,
    /*! \brief VMTIR function. */
    kVMTIRFunc = 2,
  };
  /*! \brief The kind of function. */
  FuncKind kind;
  /*! \brief The function's name, global symbol */
  std::string name;
  /*! \brief The start instruction index of the function. */
  Index start_instr = 0;
  /*! \brief The end instruction index of the function. */
  Index end_instr = 0;
  /*! \brief The number of arguments of the function. */
  Index num_args = 0;
  /*! \brief The register file size of the function. */
  Index register_file_size = 0;
  /*! \brief The function parameter names.*/
  std::vector<std::string> param_names;

  // defined customized loader save
  void Save(dmlc::Stream* writer) const;
  bool Load(dmlc::Stream* reader);
};

/*!
 * \brief The executable emitted by the VM compiler.
 *
 * The executable contains information (e.g. data in different memory regions)
 * to run in a virtual machine.
 */
class Executable : public runtime::ModuleNode {
 public:
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kBinarySerializable; };

  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globals and constants, etc.
   * \return The statistics represented by a string.
   */
  std::string Stats() const;
  /*!
   * \brief Get the i-th instruction from the executable.
   * \param i The index of the instruction to be fetched.
   * \return The instruction.
   */
  Instruction GetInstruction(Index i) const;
  /*!
   * \brief Set j-th byte data of i-th instruction to val.
   * \param i The index of the instruction to be updated.
   * \param j The index of the byte data of the instruction to be updated.
   * \param val The value to be set
   */
  void SetInstructionData(Index i, Index j, ExecWord val);
  /*!
   * \brief Print the instructions as text format.
   * \return The text format of the instructions.
   */
  String AsText() const;
  /*!
   * \brief Print the instructions as python program.
   * \return The python program of the instructions, represented by a string.
   */
  String AsPython() const;
  /*!
   * \brief Write the Executable to the binary stream in serialized form.
   * \param stream The binary stream to save the executable to.
   */
  void SaveToBinary(dmlc::Stream* stream) final;
  /*!
   * \brief Load Executable from the binary stream in serialized form.
   * \param stream The binary stream that load the executable from.
   * \return The loaded executable, in the form of a `runtime::Module`.
   */
  static Module LoadFromBinary(void* stream);
  /*!
   * \brief Write the Executable to the provided path as a file containing its serialized content.
   * \param file_name The name of the file to write the serialized data to.
   * \param format The target format of the saved file.
   */
  void SaveToFile(const String& file_name, const String& format) final;
  /*! \brief Create a Relax virtual machine and load `this` as the executable. */
  Module VMLoadExecutable() const;
  /*! \brief Create a Relax virtual machine with profiler and load `this` as the executable. */
  Module VMProfilerLoadExecutable() const;
  /*! \brief Check if the Executable contains a specific function. */
  bool HasFunction(const String& name) const;
  /*!
   * \brief Load Executable from the file.
   * \param file_name The path of the file that load the executable from.
   * \return The loaded executable, in the form of a `runtime::Module`.
   */
  static Module LoadFromFile(const String& file_name);

  /*! \brief The virtual machine's function table. */
  std::vector<VMFuncInfo> func_table;
  /*! \brief A map from globals (as strings) to their index in the function map. */
  std::unordered_map<std::string, Index> func_map;
  /*! \brief The global constant pool. */
  std::vector<TVMRetValue> constants;
  /*! \brief The offset of instruction. */
  std::vector<Index> instr_offset;
  /*! \brief The byte data of instruction. */
  std::vector<ExecWord> instr_data;

  virtual ~Executable() {}

  TVM_MODULE_VTABLE_BEGIN("relax.Executable");
  TVM_MODULE_VTABLE_ENTRY("stats", &Executable::Stats);
  TVM_MODULE_VTABLE_ENTRY("as_text", &Executable::AsText);
  TVM_MODULE_VTABLE_ENTRY("as_python", &Executable::AsPython);
  TVM_MODULE_VTABLE_ENTRY("vm_load_executable", &Executable::VMLoadExecutable);
  TVM_MODULE_VTABLE_ENTRY("vm_profiler_load_executable", &Executable::VMProfilerLoadExecutable);
  TVM_MODULE_VTABLE_ENTRY("has_function", &Executable::HasFunction);
  TVM_MODULE_VTABLE_END();

 private:
  /*!
   * \brief Save the globals.
   * \param strm The input stream.
   */
  void SaveGlobalSection(dmlc::Stream* strm);
  /*!
   * \brief Save the constant pool.
   * \param strm The input stream.
   */
  void SaveConstantSection(dmlc::Stream* strm);
  /*!
   * \brief Save the instructions.
   * \param strm The input stream.
   */
  void SaveCodeSection(dmlc::Stream* strm);
  /*!
   * \brief Save the packed functions.
   * \param strm The input stream.
   */
  void SavePackedFuncNames(dmlc::Stream* strm);
  /*!
   * \brief Load the globals.
   * \param strm The input stream.
   */
  void LoadGlobalSection(dmlc::Stream* strm);
  /*!
   * \brief Load the constant pool.
   * \param strm The input stream.
   */
  void LoadConstantSection(dmlc::Stream* strm);
  /*!
   * \brief Load the instructions.
   * \param strm The input stream.
   */
  void LoadCodeSection(dmlc::Stream* strm);
  /*!
   * \brief Save the packed functions.
   * \param strm The input stream.
   */
  void LoadPackedFuncNames(dmlc::Stream* strm);
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::relax_vm::VMFuncInfo, true);
}  // namespace dmlc
#endif  // TVM_RUNTIME_RELAX_VM_EXECUTABLE_H_
