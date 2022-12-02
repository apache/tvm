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
 * \file remove_no_op.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_TIR_TRANSFORMS_REMOVE_NO_OP_H_
#define TVM_TIR_TRANSFORMS_REMOVE_NO_OP_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt.h>

#include <optional>

#include "../analysis/control_flow_graph.h"

namespace tvm {
namespace tir {

/* \brief Remove no-ops from the statement
 *
 * Applies the same behavior as the tir.transform.RemoveNoOp pass, but
 * on a single statement, usable as a subroutine in other passes.
 *
 * \param stmt The TIR statement from which to remove no-ops
 *
 * \param analyzer The analyzer to use while proving no-ops
 *
 * \param control_flow The analyzed control-flow graph, which contains
 * the `stmt` to be analyzed.  If provided, known buffer values will
 * be used to remove no-ops.  (e.g. Removing `buf[i] = 0` in cases
 * where `buf[i]` is known to already contain zero.)  If nullptr,
 * known buffer values will not be used.
 *
 * \return The modified statement with no-ops removed
 */
Stmt RemoveNoOp(Stmt stmt, arith::Analyzer* analyzer,
                std::optional<ControlFlowGraph> touch_pattern = std::nullopt,
                const StmtNode* context = nullptr);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORMS_REMOVE_NO_OP_H_
