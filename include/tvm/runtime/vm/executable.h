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

#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/vm/bytecode.h>

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
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
class TVM_DLL Executable : public ModuleNode {
 public:
  TVM_MODULE_VTABLE_BEGIN("VMExecutable");
  TVM_MODULE_VTABLE_ENTRY("get_lib", &Executable::GetLib);
  TVM_MODULE_VTABLE_ENTRY("get_bytecode", &Executable::GetBytecode);
  TVM_MODULE_VTABLE_ENTRY("get_constants", &Executable::GetConstants);
  TVM_MODULE_VTABLE_ENTRY("get_virtual_devices", &Executable::GetVirtualDevices);
  TVM_MODULE_VTABLE_ENTRY("get_primitives", &Executable::GetPrimitives);
  TVM_MODULE_VTABLE_ENTRY("get_stats", &Executable::Stats);
  TVM_MODULE_VTABLE_ENTRY("save", &Executable::Save);
  TVM_MODULE_VTABLE_ENTRY("get_function_arity", &Executable::GetFunctionArity);
  TVM_MODULE_VTABLE_ENTRY("get_function_param_name", &Executable::GetFunctionParameterName);
  TVM_MODULE_VTABLE_ENTRY("vm_load_executable", &Executable::VMLoadExecutable);
  TVM_MODULE_VTABLE_ENTRY("move_late_bound_consts", &Executable::MoveLateBoundConstantsToFile);
  TVM_MODULE_VTABLE_ENTRY("get_late_bound_consts", &Executable::GetLateBoundConstants);
  TVM_MODULE_VTABLE_ENTRY("load_late_bound_consts", &Executable::LoadLateBoundConstantsFromFile);
  TVM_MODULE_VTABLE_ENTRY("load_late_bound_consts_from_map",
                          &Executable::LoadLateBoundConstantsFromMap);
  TVM_MODULE_VTABLE_END();

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kBinarySerializable; };
  /*! \brief Creates a VM that loads `this` as the executable. */
  Module VMLoadExecutable();
  /*!
   * \brief Write the Executable to the binary stream in serialized form.
   *
   * Late-bound constants (if any) must have already been saved by \p
   * MoveLateBoundConstantsToBinary.
   *
   * \param stream The binary stream to save the executable to.
   */
  void SaveToBinary(dmlc::Stream* stream) final;

  /*!
   * \brief Write the Executable to the provided path as a file containing its serialized content.
   *
   * Late-bound constants (if any) must have already been saved by \p
   * MoveLateBoundConstantsToBinary.
   *
   * \param path The path to write the serialized data to.
   * \param format The format of the serialized blob.
   */
  void SaveToFile(const String& path, const String& format) final;

  /*!
   * \brief Serialize the executable into global section, constant section, and
   * code section. This object must outlive the returned byte array.
   *
   * Late-bound constants (if any) must have already been saved by \p
   * MoveLateBoundConstantsToBinary.
   *
   * \return The binary representation of the VM.
   */
  TVMByteArray Save();

  /*!
   * \brief Load the saved VM executable.
   *
   * Late-bound constants (if any) must then be loaded by \p LoadLateBoundConstantsFromBinary.
   *
   * \param code The bytecode in string.
   * \param lib The compiled runtime library.
   *
   * \return exe The constructed executable.
   */
  static runtime::Module Load(const std::string& code, const runtime::Module lib);

  /*!
   * \brief Returns the late-bound constants for the executable (if any) as a byte-stream.
   * Leaves the executable's late-bound constants map empty. Only constants who's byte
   * tensor size is greater than or equal to \p byte_limit are marked as late-bound. \p byte_limit
   * may be zero.
   *
   * Must be called before \p SaveToBinary and friends if late-bound constants are
   * desired. Otherwise can be ignore.
   */
  void MoveLateBoundConstantsToStream(dmlc::Stream* stream, int64_t byte_limit);

  /*!
   * \brief As for \p MoveLateBoundConstantsToStream, but save to file at \p path.
   */
  void MoveLateBoundConstantsToFile(const std::string& path, int64_t byte_limit);

  /*!
   * \brief Get a map of all constants with larger that byte_limit in size.
   */
  Map<String, NDArray> GetLateBoundConstants(int64_t byte_limit);

  /*!
   * \brief Restores the late-bound constants for the executable (if any) from given byte-stream.
   *
   * Must be called after \p Load but before any other methods if \p MoveLateBoundConstantsToBinary
   * was used when saving. Otherwise can be ignored.
   */
  void LoadLateBoundConstantsFromStream(dmlc::Stream* stream);

  /*!
   * \brief Restores the late-bound constants for the executable (if any) from given map.
   *
   * Must be called after \p Load but before any other methods if \p MoveLateBoundConstantsToBinary
   * was used when saving. Otherwise can be ignored.
   */
  void LoadLateBoundConstantsFromMap(Map<String, NDArray> map);

  /*!
   * \brief As for \p LoadLateBoundConstantsFromStream, but load from file at \p path.
   */
  void LoadLateBoundConstantsFromFile(const std::string& path);

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
   *
   * The field starting from # is only used for debugging. The serialized code
   * doesn't contain it, therefore the deserializer doens't need to handle it.
   */
  std::string GetBytecode() const;

  /*!
   * \brief Returns a description of all the constants in the executable in human-readable
   * format. Intended for debugging and diff-testing.
   */
  std::string GetConstants() const;

  /*!
   * \brief Returns a description of all the (virtual) devices in the executable in human-readable
   * format. Intended for debugging and diff-testing.
   */
  std::string GetVirtualDevices() const;

  /*!
   * \brief Returns a description of all the 'primitive' (ie PackedFuncs) in the executable in
   * human-readable format. These correspond either to PrimFuncs we've compiled locally, or
   * functions compiled by a BYOC external codegen. Intended for debugging and diff-testing.
   */
  std::string GetPrimitives() const;

  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globls and constants, etc.
   */
  std::string Stats() const;

  /*!
   * \brief Get the `lib` module in an executable. Users have the flexibility to call
   * `export_library` from the frontend to save the library to disk.
   *
   * \return The runtime module that contains the hardware dependent code.
   */
  runtime::Module GetLib() const;

  /*!
   * \brief Set the `lib` module in an executable.
   *
   * This allows us to do partial initialization in the case of (de|ser)ialization cases.
   * This method also ensures correct initialization of library ensuring we only Import a
   * single library.
   *
   * NB: This also provides some abstraction over how libraries are stored as there are plans
   * to iterate on the way runtime::Module works in the backend of the compiler.
   */
  void SetLib(const runtime::Module& lib);

  /*!
   * \brief Get VMFunction.
   * \param func_name The function's name.
   * \return VMFunction.
   */
  const VMFunction& GetVMFunctionWithName(const std::string& func_name) const;

  /*!
   * \brief Get the arity of the VMFunction.
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
  std::string GetFunctionParameterName(std::string func, int index) const;

  virtual ~Executable() {}

  /*!
   * \brief The (compile-time, virtual) devices corresponding to each device index.
   * This vector contains a pair Device and its memory_scope.
   */
  std::vector<std::pair<Device, std::string>> virtual_devices;
  /*!
   * \brief The device index corresponding to the 'host' device. That will hold and evaluate
   * shape-related data and code.
   */
  int host_device_index = -1;
  /*!
   * \brief The global constant array.
   *
   * LoadConst instructions indexes are w.r.t. this vector. Late-bound constants are removed
   * from this table after saving late-bound constants.
   */
  std::vector<ObjectRef> constants;
  /*!
   * \brief For each constant index the name of the late-bound constant, or null if constant is
   * immediate. Only populated after loading executable but before loading late-bound constants.
   */
  std::vector<String> late_bound_constant_names;

  /*! \brief A map from globals (as strings) to their index in the Relay function map. */
  std::unordered_map<std::string, Index> global_map;
  /*! \brief A mapping from the packed function's global name (as string) to the index that
   * corresponds to the position of the `packed_funcs` list in a `VirtualMachine` object.
   */
  std::unordered_map<std::string, Index> primitive_map;
  /*! \brief The structural hashes of the operators in this function. */
  std::map<Index, Map<String, ObjectRef>> op_attrs;
  /*! \brief The virtual machine's function table. */
  std::vector<VMFunction> functions;
  /*! \brief The index of the device holding each constant. */
  std::vector<Index> const_device_indexes;

 private:
  /*!
   * \brief Save the virtual devices
   *
   * /param strm The output stream.
   */
  void SaveVirtualDevicesSection(dmlc::Stream* strm);

  /*!
   * \brief Save the globals.
   *
   * \param strm The output stream.
   */
  void SaveGlobalSection(dmlc::Stream* strm);

  /*!
   * \brief Save the constant pool.
   *
   * \param stream The output stream.
   */
  void SaveConstantSection(dmlc::Stream* stream);

  /*!
   * \brief Load the constant pool.
   *
   * \param stream The input stream.
   */
  void LoadConstantSection(dmlc::Stream* stream);

  /*!
   * \brief Save primitive op names.
   *
   *  \param strm The output stream.
   */
  void SavePrimitiveOpNames(dmlc::Stream* strm);

  /*!
   * \brief Save the vm functions.
   *
   * \param strm The output stream.
   */
  void SaveCodeSection(dmlc::Stream* strm);

  /*!
   * \brief Load the virtual devices
   *
   * /param strm The input stream.
   */
  void LoadVirtualDevicesSection(dmlc::Stream* strm);

  /*!
   * \brief Load the globals.
   *
   * \param strm The input stream.
   */
  void LoadGlobalSection(dmlc::Stream* strm);

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
