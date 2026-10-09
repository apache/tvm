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

#include <tvm/ir/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

void PrintFunction(DocTranslatorObj* d, const tirx::FunctionNode* func, ExprDoc decorator,
                   const ffi::String& dialect_attr);
bool CanTranslateExplicitResultCall(const CallNode* call);
bool IsScalarBuffer(DocTranslatorObj* d, const Expr& source);

}  // namespace details

namespace type_attr {
// S-TIR owns the syntax for marked TIRx functions when its printer is loaded.
inline constexpr const char* kSTirFunctionDocTranslate = "__tvm_doc_translate_s_tir_function__";
}  // namespace type_attr
}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // SRC_TIRX_SCRIPT_PRINTER_UTILS_H_
