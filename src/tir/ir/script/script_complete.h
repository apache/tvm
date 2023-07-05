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
 * \file tir/ir/script/script_complete.h
 * \brief Used by TVM Script parser to expand incomplete TIR input
 */
#ifndef TVM_TIR_IR_SCRIPT_SCRIPT_COMPLETE_H_
#define TVM_TIR_IR_SCRIPT_SCRIPT_COMPLETE_H_
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

PrimFunc ScriptComplete(PrimFunc func, const Array<Buffer>& root_allocates);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_SCRIPT_SCRIPT_COMPLETE_H_
