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
#ifndef TVM_SCRIPT_PRINTER_TIR_VAR_H_
#define TVM_SCRIPT_PRINTER_TIR_VAR_H_

#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace script {
namespace printer {

TracedObject<String> GetVarNameHint(const TracedObject<tir::Var>& var);

ExprDoc GetTypeAnnotationForVar(const TracedObject<tir::Var>& var, const IRDocsifier& p);

// a = T.var("int32")
IdDoc DefineVar(const TracedObject<tir::Var>& var, const Frame& frame, const IRDocsifier& p,
                std::function<void(AssignDoc)> add_def = nullptr);
// a: T.int32
IdDoc DeclareVar(const TracedObject<tir::Var>& var, const Frame& frame, const IRDocsifier& p,
                 std::function<void(AssignDoc)> add_decl = nullptr);

// T.iter_var(...)
ExprDoc IterVarDef(const TracedObject<tir::IterVar>& iter_var, const IRDocsifier& p);
// T.axis.S/R(...)
ExprDoc IterVarBlockVar(const TracedObject<tir::IterVar>& iter_var, const IRDocsifier& p);
// T.launch_thread(...)
ExprDoc IterVarLaunchThread(const TracedObject<tir::IterVar>& iter_var,
                            const TracedObject<PrimExpr>& value, const Frame& frame,
                            const IRDocsifier& p,
                            std::function<void(tir::Var, AssignDoc)> add_thread_binding);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_VAR_H_
