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

/*! \file src/relax/transform/simplify_norm_inference.cc */

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

TensorStructInfo MatchTensorStructInfo(Expr data) {
  auto _sinfo = MatchStructInfo<TensorStructInfo>(data);
  ICHECK(_sinfo.defined()) << "Expect data to be a tensor, but get " << GetStructInfo(data);
  return _sinfo.value();
}

Expr ExpandToMatchInput(Expr data, int ndim, Array<Integer> axes) {
  axes = GetOrderedPositiveAxes(axes, ndim);
  Array<Integer> expand_axes;
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (j < static_cast<int>(axes.size()) && i == axes[j]->value) {
      ++j;
    } else {
      expand_axes.push_back(i);
    }
  }
  return expand_dims(data, expand_axes);
}

Expr SimplifyBatchNorm(const CallNode* call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  ICHECK_NOTNULL(attrs);

  Expr data = call->args[0];
  TensorStructInfo sinfo = MatchTensorStructInfo(data);
  Expr gamma = call->args[1];
  Expr beta = call->args[2];
  Expr moving_mean = ExpandToMatchInput(call->args[3], sinfo->ndim, {attrs->axis});
  Expr moving_var = ExpandToMatchInput(call->args[4], sinfo->ndim, {attrs->axis});

  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(static_cast<float>(attrs->epsilon), sinfo->dtype);
  Expr sqrt_var = sqrt(add(moving_var, epsilon));
  Expr out = divide(subtract(data, moving_mean), sqrt_var);

  if (attrs->scale) {
    out = multiply(out, ExpandToMatchInput(gamma, sinfo->ndim, {attrs->axis}));
  }
  if (attrs->center) {
    out = add(out, ExpandToMatchInput(beta, sinfo->ndim, {attrs->axis}));
  }

  return out;
}

/*! \brief A mutator to simplify the normalization inference. */
class NormInferenceSimplifier : public ExprMutator {
 public:
  static Expr Simplify(Expr expr) { return NormInferenceSimplifier()(expr); }

 private:
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr expr = ExprMutator::VisitExpr_(op);
    op = expr.as<TupleGetItemNode>();
    ICHECK_NOTNULL(op);

    auto it = batch_norm_map_.find(op->tuple);
    if (it != batch_norm_map_.end() && op->index == 0) {
      return (*it).second;
    } else {
      return expr;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) final {
    ExprMutator::VisitBinding_(binding, val);
    if (val->op == Op::Get("relax.nn.batch_norm")) {
      // NOTE: we won't directly replace the batch_norm call since
      // the following bindings may depend on the returned moving_mean and moving_var.
      // Instead, we will store the unpacked value in the batch_norm_map_, and replace it
      // at the TupleGetItemNode. And the original batch_norm call will be removed in the
      // follow-up pass `RemoveAllUnused`
      batch_norm_map_.Set(binding->var, SimplifyBatchNorm(val));
    }
  }

 private:
  /*! \brief The mapping from binding var of batch_norm to the unpacked value. */
  Map<Expr, Expr> batch_norm_map_;
};

namespace transform {
Pass SimplifyNormInference() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        f = Downcast<Function>(NormInferenceSimplifier::Simplify(f));
        // Remove original batch_norm op if it's not used.
        return RemoveAllUnused(f);
      };
  return CreateFunctionPass(/*pass_function=*/pass_func,            //
                            /*opt_level=*/0,                        //
                            /*pass_name=*/"SimplifyNormInference",  //
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.SimplifyNormInference").set_body_typed(SimplifyNormInference);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
