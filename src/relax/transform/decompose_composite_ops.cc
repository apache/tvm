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
#include <tvm/relax/struct_info.h>
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

class OpDecomposer : public ExprMutator {
 public:
  static Expr Decompose(Expr expr) { return OpDecomposer()(expr); }

 private:
  using ExprMutator::VisitExpr_;
  Expr TensorToShape(const Call& call_node) {
    ICHECK(call_node->struct_info_.defined());
    Expr expr = call_node->args[0];
    const ShapeStructInfoNode* sinfo = GetStructInfoAs<ShapeStructInfoNode>(call_node);
    ICHECK(sinfo);
    // call builtin function that converts tensor to shape tuple
    // TODO(@sunggg): Register operator for "vm.builtin.tensor_to_shape"
    Var call = builder_->Emit(Call(ExternFunc("vm.builtin.tensor_to_shape"), {expr}, {},
                                   {GetRef<ShapeStructInfo>(sinfo)}));

    // Operators like reshape take the output of `TensorToShape` as their output shape.
    // Because TOPI expects to have such output shape in symbolic shape at least (i.e.,
    // Array<PrimExpr>), we define symbolic variables and returns them as a ShapeExpr.
    Array<PrimExpr> shape_var;
    for (int i = 0; i < sinfo->ndim; i++) {
      shape_var.push_back(tir::Var("x", DataType::Int(64)));
    }
    // bind symbolic variables to the shape tuple
    relax::Var var("y", ShapeStructInfo(shape_var));
    builder_->EmitNormalized(MatchCast(var, call, ShapeStructInfo(shape_var)));
    return ShapeExpr(shape_var);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    Call call = Downcast<Call>(VisitExprPostOrder_(call_node));
    if (call->op == tensor_to_shape_op_) {
      return TensorToShape(call);
    } else {
      return call;
    }
  }

  const Op& tensor_to_shape_op_ = Op::Get("relax.tensor_to_shape");
};

namespace transform {
Pass DecomposeCompositeOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        f = Downcast<Function>(NormInferenceSimplifier::Simplify(f));
        f = Downcast<Function>(OpDecomposer::Decompose(f));
        // Remove original ops if it's not used.
        return RemoveAllUnused(f);
      };
  return CreateFunctionPass(/*pass_function=*/pass_func,            //
                            /*opt_level=*/0,                        //
                            /*pass_name=*/"DecomposeCompositeOps",  //
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.DecomposeCompositeOps").set_body_typed(DecomposeCompositeOps);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
