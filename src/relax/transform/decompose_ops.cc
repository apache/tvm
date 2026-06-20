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

/*! \file src/relax/transform/decompose_ops.cc */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/function.h>

#include <unordered_set>

#include "utils.h"

namespace tvm {
namespace relax {

TensorStructInfo MatchTensorStructInfo(Expr data) {
  auto _sinfo = MatchType<TensorStructInfo>(data);
  TVM_FFI_ICHECK(_sinfo.defined()) << "Expect data to be a tensor, but get " << GetType(data);
  return _sinfo.value();
}

Expr ExpandToMatchInput(Expr data, int ndim, ffi::Array<int64_t> axes) {
  axes = GetOrderedPositiveAxes(axes, ndim);
  ffi::Array<int64_t> expand_axes;
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (j < static_cast<int>(axes.size()) && i == axes[j]) {
      ++j;
    } else {
      expand_axes.push_back(i);
    }
  }
  return expand_dims(data, expand_axes);
}

Tuple DecomposeBatchNorm(const Call& call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  TVM_FFI_ICHECK_NOTNULL(attrs);

  Expr data = call->args[0];
  TensorStructInfo sinfo = MatchTensorStructInfo(data);
  Expr gamma = call->args[1];
  Expr beta = call->args[2];

  Expr moving_mean = ExpandToMatchInput(call->args[3], sinfo->ndim, {attrs->axis});
  Expr moving_var = ExpandToMatchInput(call->args[4], sinfo->ndim, {attrs->axis});

  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(attrs->epsilon, sinfo->dtype);
  Expr sqrt_var = sqrt(add(moving_var, epsilon));
  Expr out = divide(subtract(data, moving_mean), sqrt_var);

  if (attrs->scale) {
    out = multiply(out, ExpandToMatchInput(gamma, sinfo->ndim, {attrs->axis}));
  }
  if (attrs->center) {
    out = add(out, ExpandToMatchInput(beta, sinfo->ndim, {attrs->axis}));
  }

  return Tuple({out, call->args[3], call->args[4]});
}

Expr MutateBatchNormForTraining(Call call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  TVM_FFI_ICHECK_NOTNULL(attrs);

  TVM_FFI_ICHECK_EQ(call->args.size(), 5);
  Expr data = call->args[0];
  Expr gamma = call->args[1];
  Expr beta = call->args[2];
  Expr moving_mean = call->args[3];
  Expr moving_var = call->args[4];

  TensorStructInfo sinfo = MatchTensorStructInfo(data);

  ffi::Array<int64_t> reduce_axes;
  for (int i = 0; i < sinfo->ndim; ++i) {
    if (i != attrs->axis) {
      reduce_axes.push_back(i);
    }
  }

  Expr data_mean = mean(data, reduce_axes, false);
  Expr data_var = variance(data, reduce_axes, false);

  Expr momentum = MakeConstantScalar(attrs->momentum, sinfo->dtype);
  Expr one_minus_mom = MakeConstantScalar(1 - attrs->momentum, sinfo->dtype);

  Expr new_moving_mean = add(multiply(one_minus_mom, moving_mean), multiply(momentum, data_mean));
  Expr new_moving_var = add(multiply(one_minus_mom, moving_var), multiply(momentum, data_var));

  call.CopyOnWrite()->args = {data, gamma, beta, data_mean, data_var};
  // return call;

  return relax::Tuple({TupleGetItem(call, 0), new_moving_mean, new_moving_var});
}

Expr DecomposeLayerNorm(const Call& call) {
  auto attrs = call->attrs.as<LayerNormAttrs>();
  TVM_FFI_ICHECK_NOTNULL(attrs);

  Expr data = call->args[0];
  TensorStructInfo sinfo = MatchTensorStructInfo(data);
  Expr gamma = call->args[1];
  Expr beta = call->args[2];

  Expr data_mean = mean(data, attrs->axes, true);
  Expr data_var = variance(data, attrs->axes, true);

  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(attrs->epsilon, sinfo->dtype);
  Expr sqrt_var = sqrt(add(data_var, epsilon));
  Expr out = divide(subtract(data, data_mean), sqrt_var);

  if (attrs->scale) {
    out = multiply(out, gamma);
  }
  if (attrs->center) {
    out = add(out, beta);
  }

  return out;
}

Expr TensorToShape(const Call& call_node, const BlockBuilder& builder) {
  TVM_FFI_ICHECK(call_node->ty.defined());
  Expr expr = call_node->args[0];
  const ShapeStructInfoNode* sinfo = GetTypeAs<ShapeStructInfoNode>(call_node);
  TVM_FFI_ICHECK(sinfo);
  // call builtin function that converts tensor to shape tuple
  // TODO(@sunggg): Register operator for "vm.builtin.tensor_to_shape"
  static const Op& call_pure_packed_op = Op::Get("relax.call_pure_packed");
  Var call =
      builder->Emit(Call(call_pure_packed_op, {ExternFunc("vm.builtin.tensor_to_shape"), expr}, {},
                         {ffi::GetRef<ShapeStructInfo>(sinfo)}));

  // Operators like reshape take the output of `TensorToShape` as their output shape.
  // Because TOPI expects to have such output shape in symbolic shape at least (i.e.,
  // ffi::Array<PrimExpr>), we define symbolic variables and returns them as a ShapeExpr.
  ffi::Array<PrimExpr> shape_var;
  for (int i = 0; i < sinfo->ndim; i++) {
    shape_var.push_back(tirx::Var("x", DataType::Int(64)));
  }
  // bind symbolic variables to the shape tuple
  relax::Var var("y", ShapeStructInfo(shape_var));
  builder->EmitNormalized(MatchCast(var, call, ShapeStructInfo(shape_var)));
  return ShapeExpr(shape_var);
}

/*! \brief Update operators that have a training-specific form
 *
 * Some operators, such as relax.op.batch_norm, need additional
 * processing when being run for training.  This mutator applies any mutations required
 */
class TrainingOperatorMutator : public ExprMutator {
 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) final {
    Call call = Downcast<Call>(VisitExprPostOrder_(call_node));
    if (call->op == batch_norm_op_) {
      return MutateBatchNormForTraining(call);
    } else if (call->op == layer_norm_op_) {
      // Here we only decompose LayerNorm in training because it is more efficient as a single op.
      // In the future maybe we can also remove this decomposition during training.
      return DecomposeLayerNorm(call);
    } else {
      return call;
    }
  }

  /* composite opeartor list */
  const Op& batch_norm_op_ = Op::Get("relax.nn.batch_norm");
  const Op& layer_norm_op_ = Op::Get("relax.nn.layer_norm");
};

class OpDecomposer : public ExprMutator {
 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) final {
    Call call = Downcast<Call>(VisitExprPostOrder_(call_node));
    if (call->op == batch_norm_op_) {
      return DecomposeBatchNorm(call);
    } else if (call->op == tensor_to_shape_op_) {
      return TensorToShape(call, builder_);
    }
    return call;
  }

  /* composite opeartor list */
  const Op& batch_norm_op_ = Op::Get("relax.nn.batch_norm");
  const Op& tensor_to_shape_op_ = Op::Get("relax.tensor_to_shape");
};

namespace transform {

namespace {

/*! \brief Helper: add or remove an attribute on a BaseFunc */
BaseFunc BaseFuncWithAttr(BaseFunc func, const std::string& attr_key, Any attr_value) {
  if (auto tirx = func.as<tirx::PrimFunc>()) {
    return WithAttr(tirx.value(), attr_key, attr_value);
  } else if (auto relax_fn = func.as<relax::Function>()) {
    return WithAttr(relax_fn.value(), attr_key, attr_value);
  } else {
    return func;
  }
}

BaseFunc BaseFuncWithoutAttr(BaseFunc func, const std::string& attr_key) {
  if (auto tirx = func.as<tirx::PrimFunc>()) {
    return WithoutAttr(tirx.value(), attr_key);
  } else if (auto relax_fn = func.as<relax::Function>()) {
    return WithoutAttr(relax_fn.value(), attr_key);
  } else {
    return func;
  }
}

/*!
 * \brief Apply a pass to a single named function within an IRModule.
 *
 * Replaces all other functions with dummy ExternFunc stubs so that the
 * pass does not see them, then restores the original module.  Uses
 * exact name match (not a regex) because all in-tree callers supply a
 * literal function name.
 */
Pass ApplyDecomposeToFunction(Pass pass, ffi::String func_name) {
  auto pass_func = [pass, func_name](IRModule mod, PassContext) -> IRModule {
    std::unordered_set<ffi::String> keep_original_version;
    std::unordered_set<ffi::String> internal_functions;
    IRModule subset;

    for (auto [gvar, func] : mod->functions) {
      if (gvar->name_hint == func_name) {
        if (!func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value()) {
          // Mark internal functions as externally-exposed so that
          // call-tracing transforms inside the pass do not remove them.
          internal_functions.insert(gvar->name_hint);
          func = BaseFuncWithAttr(func, tvm::attr::kGlobalSymbol, gvar->name_hint);
        }
      } else {
        // Replace non-target functions with stubs to keep references intact.
        keep_original_version.insert(gvar->name_hint);
        func = relax::ExternFunc("dummy_" + std::string(gvar->name_hint));
        func->ty = gvar->ty;
      }
      subset->Add(gvar, func);
    }

    IRModule new_subset = pass(subset);
    if (new_subset.same_as(subset)) {
      return mod;
    }

    auto write_ptr = mod.CopyOnWrite();
    for (auto [gvar, func] : new_subset->functions) {
      if (!keep_original_version.count(gvar->name_hint)) {
        if (auto it = write_ptr->global_var_map_.find(gvar->name_hint);
            it != write_ptr->global_var_map_.end()) {
          write_ptr->Remove((*it).second);
        }
        if (internal_functions.count(gvar->name_hint)) {
          func = BaseFuncWithoutAttr(func, tvm::attr::kGlobalSymbol);
        }
        write_ptr->Add(gvar, func);
      }
    }
    return mod;
  };

  std::string pass_name = "ApplyDecomposeTo" + std::string(func_name);
  return CreateModulePass(pass_func, 0, pass_name, {});
}

}  // namespace

Pass MutateOpsForTraining() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    TrainingOperatorMutator mutator;
    return Downcast<Function>(mutator(func));
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"MutateOpsForTraining",
                            /*required=*/{});
}

Pass DecomposeOps() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    OpDecomposer mutator;
    return Downcast<Function>(mutator(func));
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"DecomposeOps",
                            /*required=*/{});
}

Pass DecomposeOpsForInference(ffi::Optional<ffi::String> func_name) {
  if (func_name) {
    return ApplyDecomposeToFunction(DecomposeOps(), func_name.value());
  } else {
    return DecomposeOps();
  }
}

Pass DecomposeOpsForTraining(ffi::Optional<ffi::String> func_name) {
  auto module_pass = tvm::transform::Sequential({MutateOpsForTraining(), DecomposeOps()},
                                                "DecomposeOpsForTraining");
  if (func_name) {
    return ApplyDecomposeToFunction(module_pass, func_name.value());
  } else {
    return module_pass;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.transform.DecomposeOpsForInference", DecomposeOpsForInference)
      .def("relax.transform.DecomposeOpsForTraining", DecomposeOpsForTraining);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
