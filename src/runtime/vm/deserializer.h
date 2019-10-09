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
 * \file src/runtime/vm/deserializer.h
 * \brief Define a deserializer for the serialized Relay VM executable.
 */

#ifndef TVM_RUNTIME_VM_DESERIALIZER_H_
#define TVM_RUNTIME_VM_DESERIALIZER_H_

#include <dmlc/memory_io.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/vm.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

using namespace tvm::runtime::vm;
namespace runtime = tvm::runtime;

class Deserializer : public runtime::ModuleNode {
 public:
  /*!
   * \brief Initialize the deserializer for creating a virtual machine
   * executable object.
   *
   * \param code The serialized code.
   * \param lib The serialized runtime module/library that contains the
   * hardware dependent code.
   */
  void Init(const std::string& code, const runtime::Module& lib);

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

  const char* type_key() const final { return "Deserializer"; }

  /*! \brief Deserialize the serialized VM executable. */
  void Deserialize();

  virtual ~Deserializer() { delete strm_; }

 private:
  /*! \brief Deserialize the globals in `exec_`. */
  void DeserializeGlobalSection();

  /*! \brief Deserialize the constant pool in `exec_`. */
  void DeserializeConstantSection();

  /*! \brief Deserialize primitive op names in `exec_`. */
  void DeserializePrimitiveOpNames();

  /*! \brief Deserialize the vm functions in `exec_`. */
  void DeserializeCodeSection();

  /*! \brief Deserialize the context in `exec_`. */
  void DeserializeContextSection();

  /*! \brief The code to be serialized. */
  std::string code_;

  /*! \brief The stream used for serialization. */
  dmlc::Stream* strm_;

  /*! \brief The VM executable to be created. */
  std::shared_ptr<Executable> exec_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_DESERIALIZER_H_
