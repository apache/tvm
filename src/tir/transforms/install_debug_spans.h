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
 * \file install_debug_spans.h
 * \brief Interface of the InstallDebugSpans pass
 */

#ifndef TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_H_
#define TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>

#ifndef TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_OPS_H_
#define TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_OPS_H_

#define TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_EXPRS \
  X(Call)                                                      \
  X(Add)                                                       \
  X(Sub)                                                       \
  X(Mul)                                                       \
  X(Div)                                                       \
  X(Mod)                                                       \
  X(FloorDiv)                                                  \
  X(FloorMod)                                                  \
  X(Min)                                                       \
  X(Max)                                                       \
  X(EQ)                                                        \
  X(NE)                                                        \
  X(LT)                                                        \
  X(LE)                                                        \
  X(GT)                                                        \
  X(GE)                                                        \
  X(And)                                                       \
  X(Or)                                                        \
  X(Reduce)                                                    \
  X(Cast)                                                      \
  X(Not)                                                       \
  X(Select)                                                    \
  X(Ramp)                                                      \
  X(Broadcast)                                                 \
  X(Shuffle)                                                   \
  X(IntImm)                                                    \
  X(FloatImm)                                                  \
  X(StringImm)

#define TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_STMTS \
  X(AttrStmt)                                                  \
  X(IfThenElse)                                                \
  X(LetStmt)                                                   \
  X(For)                                                       \
  X(While)                                                     \
  X(Allocate)                                                  \
  X(AllocateConst)                                             \
  X(DeclBuffer)                                                \
  X(BufferStore)                                               \
  X(BufferRealize)                                             \
  X(AssertStmt)                                                \
  X(ProducerStore)                                             \
  X(ProducerRealize)                                           \
  X(Prefetch)                                                  \
  X(SeqStmt)                                                   \
  X(Evaluate)                                                  \
  X(BlockRealize)

#endif  // TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_OPS_H_

namespace tvm {
namespace tir {

/*!
 * \brief This Pass prints out the provided 'stmt' through the TIR debug printer
 while recording the statements and expressions printed on each line. Running
 this pass uses the per-line information to change the Spans attached to each
 statement and expression to the source location in the printed TIR. This pass
 also writes to a file called '<name>.tir' so the line information used is
 saved to disk.
 */
class DebugInfoInstaller : public StmtExprMutator {
 public:
  static Stmt InstallInfo(const std::string& name, const Stmt& stmt);

  PrimExpr VisitExpr(const PrimExpr& expr) override;
  Stmt VisitStmt(const Stmt& stmt) override;

 protected:
  DebugInfoInstaller(const Stmt& stmt, const std::string& filename);

#define X(TypeName) PrimExpr VisitExpr_(const TypeName##Node* op) override;
  TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_EXPRS
#undef X

#define X(TypeName) Stmt VisitStmt_(const TypeName##Node* op) override;
  TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_STMTS
#undef X

 private:
  std::unordered_map<const StmtNode*, size_t> stmt_lines_;
  std::unordered_map<const PrimExprNode*, size_t> expr_lines_;
  std::string filename_;

  Span MaybeSpan(const StmtNode* op);
  Span MaybeSpan(const PrimExprNode* op);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_H_
