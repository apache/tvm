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
 * \file src/runtime/vm/serialize_utils.h
 * \brief Definitions of helpers for serializing and deserializing a Relay VM.
 */
#ifndef TVM_RUNTIME_VM_SERIALIZE_UTILS_H_
#define TVM_RUNTIME_VM_SERIALIZE_UTILS_H_

#include <dmlc/memory_io.h>
#include <tvm/runtime/vm/executable.h>

#include <functional>
#include <string>
#include <vector>

#include "../../support/utils.h"

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief The magic number for the serialized VM bytecode file  */
constexpr uint64_t kTVMVMBytecodeMagic = 0xD225DE2F4214151D;

template <typename T>
static inline uint64_t VectorHash(uint64_t key, const std::vector<T>& values) {
  for (const auto& it : values) {
    key = support::HashCombine(key, it);
  }
  return key;
}

// A struct to hold the funciton info in the code section.
struct VMFunctionSerializer {
  /*! \brief The name of the VMFunction. */
  std::string name;
  /*! \brief The number of registers used by the VMFunction. */
  Index register_file_size;
  /*! \brief The number of instructions in the VMFunction. */
  size_t num_instructions;
  /*! \brief The parameters of the VMFunction. */
  std::vector<std::string> params;
  /*! \brief The index for the devices holding each parameter of the VMFunction. */
  std::vector<Index> param_device_indexes;

  VMFunctionSerializer() = default;

  VMFunctionSerializer(const std::string& name, Index register_file_size, size_t num_instructions,
                       const std::vector<std::string>& params,
                       const std::vector<Index>& param_device_indexes)
      : name(name),
        register_file_size(register_file_size),
        num_instructions(num_instructions),
        params(params),
        param_device_indexes(param_device_indexes) {}

  /*!
   * \brief Load the serialized function header.
   * \param strm The stream used to load data.
   * \return True if successful. Otherwise, false.
   */
  bool Load(dmlc::Stream* strm) {
    std::vector<std::string> func_info;
    if (!strm->Read(&func_info)) return false;
    ICHECK_EQ(func_info.size(), 3U) << "Failed to decode the vm function."
                                    << "\n";
    name = func_info[0];
    register_file_size = std::stoll(func_info[1]);
    // Get the number of instructions.
    num_instructions = static_cast<size_t>(std::stoll(func_info[2]));
    if (!strm->Read(&params)) return false;
    if (!strm->Read(&param_device_indexes)) return false;
    return true;
  }

  /*!
   * \brief Save the VM function header into the serialized form.
   * \param strm The stream used to save data.
   */
  void Save(dmlc::Stream* strm) const {
    std::vector<std::string> func_info;
    func_info.push_back(name);
    func_info.push_back(std::to_string(register_file_size));
    func_info.push_back(std::to_string(num_instructions));
    strm->Write(func_info);
    strm->Write(params);
    strm->Write(param_device_indexes);
  }
};

struct VMInstructionSerializer {
  /*! \brief The opcode of the instruction. */
  Index opcode;
  /*! \brief The fields of the instruction. */
  std::vector<Index> fields;

  VMInstructionSerializer() = default;

  VMInstructionSerializer(Index opcode, const std::vector<Index>& fields)
      : opcode(opcode), fields(fields) {}

  /*!
   * \brief Compute the hash of the serialized instruction.
   * \return The hash that combines the opcode and all fields of the VM
   * instruction.
   */
  Index Hash() const {
    uint64_t key = static_cast<uint64_t>(opcode);
    key = VectorHash(key, fields);
    return key;
  }

  /*!
   * \brief Load the serialized instruction.
   * \param strm The stream used to load data.
   * \return True if successful. Otherwise, false.
   */
  bool Load(dmlc::Stream* strm) {
    std::vector<Index> instr;
    if (!strm->Read(&instr)) return false;
    ICHECK_GE(instr.size(), 2U);
    Index loaded_hash = instr[0];
    opcode = instr[1];

    for (size_t i = 2; i < instr.size(); i++) {
      fields.push_back(instr[i]);
    }

    Index hash = Hash();
    ICHECK_EQ(loaded_hash, hash) << "Found mismatch in hash for opcode: " << opcode << "\n";
    return true;
  }

  /*!
   * \brief Save the instruction into the serialized form.
   * \param strm The stream used to save data.
   */
  void Save(dmlc::Stream* strm) const {
    Index hash = Hash();
    std::vector<Index> serialized({hash, opcode});
    serialized.insert(serialized.end(), fields.begin(), fields.end());
    strm->Write(serialized);
  }
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_SERIALIZE_UTILS_H_
