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
#ifndef SRC_SCRIPT_PRINTER_IR_UTILS_H_
#define SRC_SCRIPT_PRINTER_IR_UTILS_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/stmt.h>
#include <tvm/ir/type.h>
#include <tvm/script/printer/support.h>

#include <optional>
#include <utility>

namespace tvm {
namespace script {
namespace printer {

namespace details {

ffi::Array<StmtDoc> Body(const Stmt& stmt, DocTranslatorObj* d);
ffi::Array<Doc> TensorIndices(DocTranslatorObj* d, const ffi::Array<PrimExpr>& indices,
                              bool store = false);

ffi::Optional<ExprDoc> InvokeDocHook(ffi::AnyView hook, DocTranslatorObj* d, ffi::AnyView input,
                                     const ffi::Object* destination);
void EmitVarDefinition(DocTranslatorObj* d, const Var& var, ExprDoc rhs);

ExprDoc GlobalReference(DocTranslatorObj* d, const ffi::String& name);
ExprDoc NamedCallCallee(const ffi::String& canonical_name);

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // SRC_SCRIPT_PRINTER_IR_UTILS_H_
