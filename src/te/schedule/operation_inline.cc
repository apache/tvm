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
 * \file operation_inline.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <utility>
#include "operation_inline.h"

namespace tvm {
namespace te {

// inliner to inline a function
// the result may not be SSA,
// ConvertSSA need to be applied after this pass
class OperationInliner final : public StmtExprMutator {
 public:
  OperationInliner(Operation op, Array<Var> args, PrimExpr body)
      : operation_(op), args_(args), body_(body) {}

  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();

    if (op->func.same_as(operation_)) {
      CHECK_EQ(op->value_index, 0);
      expr = body_;
      CHECK_EQ(args_.size(), op->args.size());

      bool has_side_effect = false;
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (HasSideEffect(op->args[i])) has_side_effect = true;
      }
      if (has_side_effect) {
        for (size_t i = 0; i < args_.size(); ++i) {
          expr = LetNode::make(args_[i], op->args[i], expr);
        }
      } else {
        Map<Var, PrimExpr> vmap;
        for (size_t i = 0; i < args_.size(); ++i) {
          vmap.Set(args_[i], op->args[i]);
        }
        expr = Substitute(
            EvaluateNode::make(expr), vmap).as<EvaluateNode>()->value;
      }
      return expr;
    } else {
      return expr;
    }
  }

 private:
  Operation operation_;
  Array<Var> args_;
  PrimExpr body_;
};

Stmt Inline(Stmt stmt,
            Operation f,
            Array<Var> args,
            PrimExpr body) {
  CHECK_EQ(f->num_outputs(), 1)
      << "can only inline output single value operation";
  Stmt ret = OperationInliner(f, args, body)(std::move(stmt));
  if (ret.same_as(stmt)) return ret;
  return ConvertSSA(ret);
}
}  // namespace te
}  // namespace tvm
