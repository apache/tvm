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
 * \file canonicalize_cast.cc
 * \brief Canonicalize cast expressions to make operator fusion more efficient.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "pass_utils.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

// This pass finds upcast that is referred by multiple elemwise/broadcast operators, and creates a
// copy of it in each branch such that after fusion the previous function have output with fewer
// bits.
//
// Consider the following example:
// \code
// def @main(x: int8) {
//   %1 = cast(%x, f32)
//   %2 = exp(%1)
//   %3 = log(%1)
//   (%3, 4)
// }
// \endcode
//
// We would like to prevent sharing of the cast expression such that operator fusion can produce
// more efficient result as below.
// \code
// def @main(x: int8) {
//   %1 = fn (%p1: i8) {
//     exp(cast(%p1, f32)
//   }
//   %3 = %1(%x)
//   %2 = fn (%p1: i8) {
//     log(cast(%p1, f32)
//   }
//   %4 = %2(%x)
//   (%3, 4)
// }
// \endcode
class CastCanonicalizer : public ExprMutator {
 public:
  CastCanonicalizer() : cast_op_(Op::Get("cast")) {}

  Expr VisitExpr_(const CallNode* call) {
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");

    if (auto call_op = call->op.as<Op>()) {
      auto pattern = fpattern[call_op.value()];
      if (pattern <= kBroadcast) {
        Array<Expr> call_args = call->args;
        bool unchanged = true;
        for (size_t i = 0; i < call_args.size(); ++i) {
          Expr arg = call_args[i];
          Expr new_arg = GetNewCallArg(arg);
          if (!arg.same_as(new_arg)) {
            call_args.Set(i, new_arg);
            unchanged = false;
          }
        }
        if (unchanged) {
          return GetRef<Expr>(call);
        }
        return Call(call->op, call_args, call->attrs, call->type_args);
      }
    }

    Expr new_expr = ExprMutator::VisitExpr_(call);
    return new_expr;
  }

 private:
  std::unordered_map<const Object*, size_t> ref_counter_;
  // cast op is frequently checked for equivalence. Therefore, we cache it to
  // reduce lookup overhead.
  const Op& cast_op_;

  Expr GetNewCallArg(const Expr& e) {
    // if e is a upcast and ref count > 1, create an copy; otherwise call the default visitor
    Expr new_expr = this->VisitExpr(e);

    if (const CallNode* call = e.as<CallNode>()) {
      if (call->op == cast_op_) {
        auto attrs = call->attrs.as<CastAttrs>();
        const auto* from_type = call->args[0]->type_as<TensorTypeNode>();
        ICHECK(from_type);

        if (from_type->dtype.bits() < attrs->dtype.bits()) {
          if (++ref_counter_[call] > 1) {
            const CallNode* new_call = new_expr.as<CallNode>();
            ICHECK(new_call);
            ICHECK(new_call->op == cast_op_);
            return Call(new_call->op, new_call->args, new_call->attrs, new_call->type_args);
          }
        }
      }
    }
    return new_expr;
  }
};

Expr CanonicalizeCast(const Expr& e) { return CastCanonicalizer().Mutate(e); }

namespace transform {

Pass CanonicalizeCast() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CanonicalizeCast(f));
      };
  return CreateFunctionPass(pass_func, 3, "CanonicalizeCast", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CanonicalizeCast").set_body_typed(CanonicalizeCast);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
