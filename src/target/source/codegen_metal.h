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
 * \file codegen_metal.h
 * \brief Generate Metal device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_METAL_H_
#define TVM_TARGET_SOURCE_CODEGEN_METAL_H_

#include <tvm/target/codegen.h>

#include <string>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenMetal final : public CodeGenC {
 public:
  CodeGenMetal();
  // override print thread tag.
  void PrintArgUnionDecl();
  void AddFunction(const PrimFunc& f);  // NOLINT(*)
  void InitFuncState(const PrimFunc& f) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintStorageSync(const CallNode* op) final;                           // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;                        // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;                             // NOLINT(*)
  // print load of single element
  void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  // print store of single element.
  void PrintVecElemStore(const std::string& vec, DataType t, int i, const std::string& value) final;
  // overload visitor
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  // overload visitor
  void VisitExpr_(const CallNode* op, std::ostream& os) final;  // NOLINT(*)
  // reuse parent's function.
  using CodeGenC::PrintType;

 private:
  int thread_index_bits_{32};
};
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_METAL_H_
