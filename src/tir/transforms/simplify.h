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
 * \file simplify.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_TIR_TRANSFORMS_SIMPLIFY_H_
#define TVM_TIR_TRANSFORMS_SIMPLIFY_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tir {

/* \brief Simplifies the statement
 *
 * Applies the same behavior as the tir.transform.Simplify pass, but
 * on a single statement, usable as a subroutine in other passes.
 */
Stmt Simplify(Stmt stmt, arith::Analyzer* analyzer);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORMS_SIMPLIFY_H_
