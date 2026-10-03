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
#ifndef SRC_TIRX_SCRIPT_PRINTER_UTILS_H_
#define SRC_TIRX_SCRIPT_PRINTER_UTILS_H_

#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/var.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

void PrintPrimFunc(DocTranslatorObj* d, const tirx::PrimFuncNode* func, ExprDoc decorator,
                   const ffi::String& dialect_attr);
bool CanTranslateExplicitResultCall(const CallNode* call);
bool IsScalarBuffer(DocTranslatorObj* d, const Expr& source);
ffi::Array<StmtDoc> Body(const tirx::Stmt& stmt, DocTranslatorObj* d);
ffi::Optional<ExprDoc> TIRCallPrefixDocTranslate(DocTranslatorObj* d, const CallNode* call);
ffi::Optional<ExprDoc> FFIKernelDocTranslate(DocTranslatorObj* d, const CallNode* call,
                                             const Type& result_type,
                                             const ffi::Array<ExprDoc>& args);
ffi::Optional<ExprDoc> TIRCallDocTranslate(DocTranslatorObj* d, const CallNode* call,
                                           const Type& result_type,
                                           const ffi::Array<ExprDoc>& args);
ExprDoc TensorRegionValue(DocTranslatorObj* d, const TensorRegionNode* region, bool require_region);
ffi::Optional<ExprDoc> VarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination);

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // SRC_TIRX_SCRIPT_PRINTER_UTILS_H_
