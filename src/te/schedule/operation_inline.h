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
 * \file operation_inline.h
 */
#ifndef TVM_TE_SCHEDULE_OPERATION_INLINE_H_
#define TVM_TE_SCHEDULE_OPERATION_INLINE_H_

#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace te {

/*!
 * \brief inline all calls of f in stmt.
 *
 * \param stmt The statement to apply inline optimization.
 * \param op The op to be inlined.
 * \param args The arguments variable of the function.
 * \param body The definition body of the function.
 * \return The result stmt
 *
 * \note All the passes in this file uses SSA form and outputs SSA form.
 */
Stmt Inline(Stmt stmt, Operation op, Array<Var> args, PrimExpr body);

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_SCHEDULE_OPERATION_INLINE_H_
