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
 * \file codegen_vhls.h
 * \brief Utility to generate vhls code
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_VHLS_H_
#define TVM_TARGET_SOURCE_CODEGEN_VHLS_H_

#include <tvm/target/codegen.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <string>
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenVivadoHLS final : public CodeGenC {
 public:
  void Init(bool output_ssa);
  void PrintType(DataType t, std::ostream& os);
  void AddFunction(LoweredFunc f);
  void PreFunctionBody(LoweredFunc f);
  void VisitExpr_(const MinNode *op, std::ostream& os);
  void VisitExpr_(const MaxNode *op, std::ostream& os);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_VHLS_H_
