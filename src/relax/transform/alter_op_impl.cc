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
 * \file src/relax/transform/alter_op_impl.cc
 * \brief Change the layout of PrimFunc in the graph. It uses the kOperatorName attribute to
 * identify PrimFuncs to be replaced. Marks the new PrimFuncs with kFrozenLayout attribute set to
 * true.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/serialization.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/te/operation.h>
#include <tvm/tir/transform.h>
#include <tvm/topi/tags.h>

#include "../../te/operation/create_primfunc.h"
namespace tvm {
namespace relax {

using namespace tir;
static constexpr const char* kOperatorName = "operator_name";

/*! \brief Construct ranges from shape dimensions */
static Array<Range> ConstructRangeFromShape(const Array<PrimExpr>& shape) {
  return shape.Map([](const PrimExpr& dim) { return Range(tir::make_zero(dim.dtype()), dim); });
}

static Array<PrimExpr> GetShapeFromTensorStructInfo(const TensorStructInfo& tensor_sinfo) {
  auto shape = tensor_sinfo->GetShape();
  ICHECK(shape.defined());
  return shape.value();
}

static Array<PrimExpr> GetShapeFromTensor(const Expr& expr) {
  const auto& tensor_sinfo = Downcast<TensorStructInfo>(expr->struct_info_);
  return GetShapeFromTensorStructInfo(tensor_sinfo);
}

static IndexMap DeepCopyIndexMap(const IndexMap& index_map) {
  return Downcast<IndexMap>(LoadJSON(SaveJSON(index_map)));
}

/*! \brief Checks if the \p transform is bijective on the shape of \p expr */
bool IsTransformBijective(const Expr& expr, const IndexMap& transform) {
  Array<PrimExpr> input_shape = GetShapeFromTensor(expr);
  Array<Range> initial_ranges = ConstructRangeFromShape(input_shape);
  arith::Analyzer analyzer;
  auto [inverse, padding_predicate] = transform.NonSurjectiveInverse(initial_ranges, &analyzer);
  (void)inverse;  // to avoid unused variable warning;
  if (!analyzer.CanProve(!padding_predicate)) return false;
  return true;
}

/*!
 * \brief Replace each call_tir to PrimFunc which matches the kOperatorName attribute with the
 * provided replacement PrimFunc and mark it with kFrozenLayout attribute. Insert layout
 * transformations on i/o buffers as necessary for correctness.
 */
class AlterOpImplMutator : public ExprMutator {
 public:
  AlterOpImplMutator(const IRModule& mod, const Map<String, tir::PrimFunc>& op_impl_map,
                     const Map<String, Array<IndexMap>>& op_buffer_transforms_,
                     const Map<String, Array<Array<IntImm>>>& axis_separators_)
      : ExprMutator(mod),
        mod_(mod),
        op_impl_map_(op_impl_map),
        op_buffer_transforms__(op_buffer_transforms_),
        op_buffer_axis_separators__(axis_separators_) {}

  IRModule Run() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<relax::FunctionNode>()) {
        relax::Function update_func = Downcast<Function>(VisitExpr(func));
        builder_->UpdateFunction(gv, update_func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* op) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(op));

    // TODO(@tvm-team): When we differentiate the call for tir function and packed function,
    // this logic should be changed accordingly.
    if (!call->op.same_as(call_tir_op_)) return call;

    // Do not do anything for external function
    if (call->args[0].as<ExternFuncNode>()) return call;

    // Get operator name from callee
    ICHECK(call->args[0]->IsInstance<GlobalVarNode>());
    const tir::PrimFunc& old_func =
        Downcast<tir::PrimFunc>(mod_->Lookup(Downcast<GlobalVar>(call->args[0])));
    Optional<String> maybe_op_kind = old_func->attrs.GetAttr<String>(kOperatorName);

    // If the callee does not have kOperatorName attribute or no replacement is requested for
    // it, nothing to do here.
    if (!maybe_op_kind.defined() || op_impl_map_.count(maybe_op_kind.value()) == 0) return call;
    auto op_kind = maybe_op_kind.value();

    const auto& replacement_func = op_impl_map_[op_kind];

    Array<IndexMap> buffer_transforms;
    Optional<Array<Array<IntImm>>> axis_separators;
    if (op_buffer_transforms__.count(op_kind)) buffer_transforms = op_buffer_transforms__[op_kind];
    if (op_buffer_axis_separators__.count(op_kind))
      axis_separators = op_buffer_axis_separators__[op_kind];

    ICHECK(buffer_transforms.empty() || buffer_transforms.size() == replacement_func->params.size())
        << "Either the i/o buffers do not require any transformations or transformations for each "
           "buffer is provided.";
    ICHECK_EQ(old_func->params.size(), replacement_func->params.size())
        << "Number of parameters of old and replacement PrimFunc must match";

    GlobalVar replacement_gv = GetOrCreateGlobalVarForFunc(replacement_func, op_kind);

    auto call_tir_inputs_tuple = GetRef<Tuple>(call->args[1].as<TupleNode>());
    Tuple updated_inputs = UpdateInputs(call_tir_inputs_tuple, buffer_transforms, axis_separators);

    ICHECK_EQ(call->sinfo_args.size(), 1) << "call_tir sinfo_args.size() is expected to be 1";
    StructInfo updated_ret_sinfo = UpdateStructInfo(call->sinfo_args[0], buffer_transforms);
    auto updated_call = builder_->Normalize(
        Call(call_tir_op_, {replacement_gv, updated_inputs}, call->attrs, {updated_ret_sinfo}));

    // Now transform each of the outputs to previous layout.
    return TransformOutputs(updated_call, buffer_transforms, call->sinfo_args[0], axis_separators);
  }

  Array<TensorStructInfo> GetTensorStructInfoPerOutput(const StructInfo& output_sinfo) {
    if (const auto* tensor_sinfo = output_sinfo.as<TensorStructInfoNode>())
      return {GetRef<TensorStructInfo>(tensor_sinfo)};
    const auto* tuple_sinfo = output_sinfo.as<TupleStructInfoNode>();
    ICHECK(tuple_sinfo);

    Array<TensorStructInfo> arr_tensor_sinfo;
    arr_tensor_sinfo.reserve(tuple_sinfo->fields.size());
    for (const auto& sinfo : tuple_sinfo->fields) {
      const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>();
      ICHECK(tensor_sinfo) << "Nested tuples in output of call_tir is not supported yet";
      arr_tensor_sinfo.push_back(GetRef<TensorStructInfo>(tensor_sinfo));
    }
    return arr_tensor_sinfo;
  }

  bool IsScalarConstant(const Expr& expr) {
    if (expr->IsInstance<ConstantNode>() && expr.as<ConstantNode>()->is_scalar()) {
      return true;
    }
    return false;
  }

  Expr TransformLayout(const Expr& expr, const IndexMap& index_map,
                       const Array<IntImm>& axis_separators) {
    if (IsScalarConstant(expr) || index_map.get() == nullptr) {
      return expr;
    }
    ObjectPtr<LayoutTransformAttrs> attrs = make_object<LayoutTransformAttrs>();
    // We want to avoid two layout_transform ops to share the same index map even if they are
    // identical. The scope of vars used in index map initial indices is local to the op. Not doing
    // so would confuse the structural equality check.
    attrs->index_map = std::move(DeepCopyIndexMap(index_map));
    attrs->axis_separators = std::move(axis_separators);
    return Call(layout_transform_op_, {expr}, Attrs{std::move(attrs)}, {});
  }

  /*!
   * \brief Adds the \p remove_pad op to the module if it has not already been added before.
   * \returns The global var associated with the remove_pad PrimFunc.
   */
  GlobalVar GetOrCreateRemovePadOp(const Array<PrimExpr>& old_shape, const DataType& dtype) {
    int t_shape = old_shape.size();
    if (remove_pad_map_.count(t_shape) != 0) {
      return remove_pad_map_[t_shape];
    }
    // Create dynamic shapes for input and output tensors
    Array<PrimExpr> dyn_padded_shape, dyn_old_shape;
    for (int i = 0; i < t_shape; i++) {
      tir::Var var1("p" + std::to_string(i), old_shape[i].dtype());
      tir::Var var2("i" + std::to_string(i), old_shape[i].dtype());
      dyn_padded_shape.push_back(var1);
      dyn_old_shape.push_back(var2);
    }

    // Input tensor of remove_pad op
    te::Tensor placeholder_tensor = te::placeholder(dyn_padded_shape, dtype, "input");
    // Output tensor of remove_pad op
    te::Tensor output_tensor = te::compute(
        dyn_old_shape,
        [&placeholder_tensor](const Array<tir::Var>& indices) {
          return placeholder_tensor(indices);
        },
        "output", topi::kElementWise);

    String op_name = "remove_pad";
    // Create PrimFunc and add op_name to func.attrs
    PrimFunc remove_pad_with_frozen_layout =
        WithAttr(CreatePrimFunc({placeholder_tensor, output_tensor}), kOperatorName, op_name);
    // Add PrimFunc to module
    GlobalVar gv_remove_pad = builder_->AddFunction(remove_pad_with_frozen_layout, op_name);
    // Mark the remove_pad PrimFunc as private by removing it from global scope
    builder_->UpdateFunction(gv_remove_pad,
                             WithoutAttr(remove_pad_with_frozen_layout, "global_symbol"));

    remove_pad_map_[t_shape] = gv_remove_pad;
    return gv_remove_pad;
  }

  Expr TransformLayoutInverse(const Expr& expr, const IndexMap& index_map,
                              const TensorStructInfo& old_tensor_sinfo,
                              const Array<IntImm>& axis_separator) {
    if (IsScalarConstant(expr) || index_map.get() == nullptr) {
      return expr;
    }
    Array<PrimExpr> old_shape = GetShapeFromTensorStructInfo(old_tensor_sinfo);
    Array<Range> initial_ranges = ConstructRangeFromShape(old_shape);
    arith::Analyzer analyzer;
    auto [inverse_index_map, padding_predicate] =
        index_map.NonSurjectiveInverse(initial_ranges, &analyzer);

    if (tir::is_zero(padding_predicate)) {
      return TransformLayout(expr, inverse_index_map, axis_separator);
    } else {
      auto padded_expr =
          builder_->Normalize(TransformLayout(expr, inverse_index_map, axis_separator));
      const auto& tensor_sinfo = Downcast<TensorStructInfo>(padded_expr->struct_info_);

      GlobalVar gv_remove_pad = GetOrCreateRemovePadOp(old_shape, tensor_sinfo->dtype);
      return Call(call_tir_op_, {gv_remove_pad, Tuple({padded_expr})}, {}, {old_tensor_sinfo});
    }
  }

  /*!
   * \brief Adds the \p replacement_func to the module if it has not already been added before.
   * \returns The global var associated with the PrimFunc.
   */
  GlobalVar GetOrCreateGlobalVarForFunc(const PrimFunc& replacement_func, const String& op_kind) {
    if (cache_.count(replacement_func) != 0) {
      return cache_[replacement_func];
    }
    // Retain the operator name attribute on the replacement PrimFunc. This can help any future
    // passes that use kOperatorName attribute to identify operator represented by a PrimFunc.
    PrimFunc replacement_func_with_frozen_layout =
        WithAttr(replacement_func, kOperatorName, op_kind);

    GlobalVar gv_replacement =
        builder_->AddFunction(replacement_func_with_frozen_layout, op_kind + "_replacement");
    cache_.Set(replacement_func, gv_replacement);
    return gv_replacement;
  }

  /*!
   * \brief Updates call inputs with layout transformed inputs
   */
  Tuple UpdateInputs(const Tuple& inputs, const Array<IndexMap>& transforms,
                     const Optional<Array<Array<IntImm>>>& axis_separators) {
    if (transforms.empty()) return inputs;

    Array<Expr> updated_inputs;
    int index = 0;
    for (const auto& input : inputs->fields) {
      Array<IntImm> axis_separator;
      if (axis_separators.defined()) {
        Array<Array<IntImm>> axis_separators_value = axis_separators.value();
        axis_separator = axis_separators_value[index];
      }
      auto transform = transforms[index++];
      updated_inputs.push_back(TransformLayout(input, transform, axis_separator));
    }
    return Tuple(updated_inputs);
  }

  /*! \brief Updates output struct info */
  StructInfo UpdateStructInfo(const StructInfo& out_sinfo,
                              const Array<IndexMap>& buffer_transforms) {
    if (buffer_transforms.empty()) return out_sinfo;

    if (out_sinfo->IsInstance<TensorStructInfoNode>())
      return UpdateStructInfo(Downcast<TensorStructInfo>(out_sinfo),
                              buffer_transforms[buffer_transforms.size() - 1]);

    ICHECK(out_sinfo->IsInstance<TupleStructInfoNode>())
        << "Expect output struct info of call_tir to be either TupleStructInfo or "
           "TensorStructInfo, but got "
        << out_sinfo;

    const auto& tuple_sinfo = Downcast<TupleStructInfo>(out_sinfo);
    Array<StructInfo> sinfo_fields;
    size_t first_output_index = buffer_transforms.size() - tuple_sinfo->fields.size();
    size_t i = 0;
    for (const auto& si : tuple_sinfo->fields) {
      ICHECK(si->IsInstance<TensorStructInfoNode>())
          << "Fields of TupleStructInfo must be TensorStructInfo for call_tir "
             "output structinfo, but got "
          << si;
      sinfo_fields.push_back(UpdateStructInfo(Downcast<TensorStructInfo>(si),
                                              buffer_transforms[first_output_index + i++]));
    }
    return TupleStructInfo(sinfo_fields);
  }

  /*! \brief Returns the TensorStructInfo after applying the \p transform on its shape */
  StructInfo UpdateStructInfo(const TensorStructInfo& tensor_sinfo, const IndexMap& transform) {
    if (transform.get() == nullptr) return tensor_sinfo;
    auto shape = GetShapeFromTensorStructInfo(tensor_sinfo);
    arith::Analyzer analyzer;
    auto new_shape = transform->MapShape(shape, &analyzer);
    if (tensor_sinfo->vdevice.defined()) {
      return TensorStructInfo(ShapeExpr(new_shape), tensor_sinfo->dtype,
                              tensor_sinfo->vdevice.value());
    }
    return TensorStructInfo(ShapeExpr(new_shape), tensor_sinfo->dtype);
  }

  Expr TransformOutputs(const Expr& expr, const Array<IndexMap>& buffer_transforms,
                        const StructInfo& old_struct_info,
                        const Optional<Array<Array<IntImm>>>& axis_separators) {
    if (buffer_transforms.empty()) return expr;

    Array<TensorStructInfo> old_output_sinfo = GetTensorStructInfoPerOutput(old_struct_info);

    Array<IntImm> axis_sep;
    size_t num_outputs = old_output_sinfo.size();
    if (num_outputs == 0) return expr;

    size_t first_output_index = buffer_transforms.size() - num_outputs;
    // If there is a single output, return the transformed output.
    if (num_outputs == 1) {
      IndexMap output_map = buffer_transforms[first_output_index];
      if (axis_separators.defined()) {
        Array<Array<IntImm>> axis_separators_value = axis_separators.value();
        axis_sep = axis_separators_value[first_output_index];
      }
      return TransformLayoutInverse(expr, output_map, old_output_sinfo[0], axis_sep);
    }

    // In case of more than one output, we would have to get each item of the output tuple,
    // transform it and return a tuple of all transformed outputs.
    Array<Expr> transformed_outputs;
    for (size_t i = 0; i + first_output_index < buffer_transforms.size(); ++i) {
      const auto& output_map = buffer_transforms[i + first_output_index];
      if (axis_separators.defined()) {
        Array<Array<IntImm>> axis_separators_value = axis_separators.value();
        axis_sep = axis_separators_value[i + first_output_index];
      }
      auto output = builder_->Normalize(TupleGetItem(expr, static_cast<int>(i)));
      transformed_outputs.push_back(
          TransformLayoutInverse(output, output_map, old_output_sinfo[i], axis_sep));
    }
    return Tuple(transformed_outputs);
  }

 private:
  /*! \brief Cache to keep track of the GlobalVar associated with the new PrimFunc added */
  Map<PrimFunc, GlobalVar> cache_;
  /*! \brief Input IRModule */
  const IRModule& mod_;
  /*! \brief Map from shape_dim.size to the remove_pad GlobalVar */
  std::unordered_map<int, GlobalVar> remove_pad_map_;
  /*! \brief Map from kOperatorName attribute to the replacement PrimFunc */
  const Map<String, PrimFunc>& op_impl_map_;
  /*! \brief Map from kOperatorName attribute to the layout transforms on i/o buffers */
  const Map<String, Array<IndexMap>>& op_buffer_transforms__;
  /*! \brief Map from kOperatorName attribute to the axis separatos on i/o buffers */
  const Map<String, Array<Array<IntImm>>>& op_buffer_axis_separators__;

  const Op& call_tir_op_ = Op::Get("relax.call_tir");
  const Op& layout_transform_op_ = Op::Get("relax.layout_transform");
};

namespace transform {

Pass AlterOpImpl(const Map<String, tir::PrimFunc>& op_impl_map,
                 const Map<String, Array<IndexMap>>& op_buffer_transforms_,
                 const Map<String, Array<Array<IntImm>>>& axis_separators_) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    return AlterOpImplMutator(mod, op_impl_map, op_buffer_transforms_, axis_separators_).Run();
  };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"AlterOpImpl",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.AlterOpImpl").set_body_typed(AlterOpImpl);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
