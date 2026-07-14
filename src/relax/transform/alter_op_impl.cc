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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/transform.h>
#include <tvm/topi/tags.h>

#include "../../te/operation/create_primfunc.h"
namespace tvm {
namespace relax {

using namespace tirx;
static constexpr const char* kOperatorName = "operator_name";

/*! \brief Construct ranges from shape dimensions */
static ffi::Array<Range> ConstructRangeFromShape(const ffi::Array<PrimExpr>& shape) {
  return shape.Map([](const PrimExpr& dim) { return Range(IntImm(dim.ty(), 0), dim); });
}

static ffi::Array<PrimExpr> GetShapeFromTensorType(const TensorType& tensor_ty) {
  auto shape = tensor_ty->GetShape();
  TVM_FFI_ICHECK(shape.has_value());
  return shape.value();
}

static ffi::Array<PrimExpr> GetShapeFromTensor(const Expr& expr) {
  const auto& tensor_ty = expr->ty.as_or_throw<TensorType>();
  return GetShapeFromTensorType(tensor_ty);
}

static IndexMap DeepCopyIndexMap(const IndexMap& index_map) {
  return ffi::FromJSONGraph(ffi::ToJSONGraph(index_map)).as_or_throw<IndexMap>();
}

/*! \brief Checks if the \p transform is bijective on the shape of \p expr */
bool IsTransformBijective(const Expr& expr, const IndexMap& transform) {
  ffi::Array<PrimExpr> input_shape = GetShapeFromTensor(expr);
  ffi::Array<Range> initial_ranges = ConstructRangeFromShape(input_shape);
  arith::Analyzer analyzer;
  auto [inverse, padding_predicate] = transform.NonSurjectiveInverse(initial_ranges, analyzer);
  (void)inverse;  // to avoid unused variable warning;
  if (!analyzer->CanProve(!padding_predicate)) return false;
  return true;
}

/*!
 * \brief Replace each call_tir to PrimFunc which matches the kOperatorName attribute with the
 * provided replacement PrimFunc and mark it with kFrozenLayout attribute. Insert layout
 * transformations on i/o buffers as necessary for correctness.
 */
class AlterOpImplMutator : public ExprMutator {
 public:
  AlterOpImplMutator(
      const IRModule& mod, const ffi::Map<ffi::String, tirx::PrimFunc>& op_impl_map,
      const ffi::Map<ffi::String, ffi::Array<IndexMap>>& op_buffer_transforms_,
      const ffi::Map<ffi::String, ffi::Optional<ffi::Array<ffi::Array<IntImm>>>>& axis_separators_,
      const ffi::Map<ffi::String, ffi::Optional<ffi::Array<ffi::Array<IntImm>>>>&
          input_axis_separators_)
      : ExprMutator(mod),
        mod_(mod),
        op_impl_map_(op_impl_map),
        op_buffer_transforms__(op_buffer_transforms_),
        op_buffer_axis_separators__(axis_separators_),
        op_buffer_input_axis_separators__(input_axis_separators_) {}

  IRModule Run() {
    for (const auto& gv : mod_->GetGlobalVars()) {
      const auto& func = mod_->Lookup(gv);
      if (func->IsInstance<relax::FunctionNode>()) {
        relax::Function update_func = VisitExpr(func).as_or_throw<Function>();
        builder_->UpdateFunction(gv, update_func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* op) final {
    auto call = ExprMutator::VisitExpr_(op).as_or_throw<Call>();

    // TODO(@tvm-team): When we differentiate the call for tirx function and packed function,
    // this logic should be changed accordingly.
    if (!call->op.same_as(call_tir_op_)) return call;

    // Do not do anything for external function
    if (call->args[0].as<ExternFuncNode>()) return call;

    // Get operator name from callee
    TVM_FFI_ICHECK(call->args[0]->IsInstance<GlobalVarNode>());
    const tirx::PrimFunc& old_func =
        mod_->Lookup(call->args[0].as_or_throw<GlobalVar>()).as_or_throw<tirx::PrimFunc>();
    ffi::Optional<ffi::String> maybe_op_kind = old_func->attrs.GetAttr<ffi::String>(kOperatorName);

    // If the callee does not have kOperatorName attribute or no replacement is requested for
    // it, nothing to do here.
    if (!maybe_op_kind.has_value() || op_impl_map_.count(maybe_op_kind.value()) == 0) return call;
    auto op_kind = maybe_op_kind.value();

    const auto& replacement_func = op_impl_map_[op_kind];

    ffi::Array<IndexMap> buffer_transforms;
    ffi::Optional<ffi::Array<ffi::Array<IntImm>>> axis_separators;
    ffi::Optional<ffi::Array<ffi::Array<IntImm>>> input_axis_separators;
    if (op_buffer_transforms__.count(op_kind)) buffer_transforms = op_buffer_transforms__[op_kind];
    if (op_buffer_axis_separators__.count(op_kind))
      axis_separators = op_buffer_axis_separators__[op_kind];
    if (op_buffer_input_axis_separators__.count(op_kind))
      input_axis_separators = op_buffer_input_axis_separators__[op_kind];

    TVM_FFI_ICHECK(buffer_transforms.empty() ||
                   buffer_transforms.size() == replacement_func->params.size())
        << "Either the i/o buffers do not require any transformations or transformations for each "
           "buffer is provided.";
    TVM_FFI_ICHECK_EQ(old_func->params.size(), replacement_func->params.size())
        << "Number of parameters of old and replacement PrimFunc must match";

    GlobalVar replacement_gv = GetOrCreateGlobalVarForFunc(replacement_func, op_kind);

    auto call_tir_inputs_tuple = ffi::GetRef<Tuple>(call->args[1].as<TupleNode>());
    Tuple updated_inputs = UpdateInputs(call_tir_inputs_tuple, buffer_transforms, axis_separators,
                                        input_axis_separators);

    TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1) << "call_tir ty_args.size() is expected to be 1";
    Type updated_ret_ty = UpdateOutputType(call->ty_args[0], buffer_transforms);
    auto updated_call =
        builder_->Normalize(Call(Type::Missing(), call_tir_op_, {replacement_gv, updated_inputs},
                                 call->attrs, {updated_ret_ty}));

    // Now transform each of the outputs to previous layout.
    return TransformOutputs(updated_call, buffer_transforms, call->ty_args[0], axis_separators,
                            input_axis_separators);
  }

  ffi::Array<TensorType> GetTensorTypePerOutput(const Type& output_ty) {
    if (const auto* tensor_ty = output_ty.as<TensorTypeNode>())
      return {ffi::GetRef<TensorType>(tensor_ty)};
    const auto* tuple_ty = output_ty.as<TupleTypeNode>();
    TVM_FFI_ICHECK(tuple_ty);

    ffi::Array<TensorType> tensor_tys;
    tensor_tys.reserve(tuple_ty->fields.size());
    for (const auto& ty : tuple_ty->fields) {
      const auto* tensor_ty = ty.as<TensorTypeNode>();
      TVM_FFI_ICHECK(tensor_ty) << "Nested tuples in output of call_tir is not supported yet";
      tensor_tys.push_back(ffi::GetRef<TensorType>(tensor_ty));
    }
    return tensor_tys;
  }

  bool IsScalarConstant(const Expr& expr) {
    if (expr->IsInstance<ConstantNode>() && expr.as<ConstantNode>()->is_scalar()) {
      return true;
    }
    return false;
  }

  Expr TransformLayout(const Expr& expr, const IndexMap& index_map,
                       const ffi::Array<IntImm>& axis_separators,
                       const ffi::Array<IntImm>& input_axis_separators) {
    if (IsScalarConstant(expr) || index_map.get() == nullptr) {
      return expr;
    }
    ffi::ObjectPtr<LayoutTransformAttrs> attrs = ffi::make_object<LayoutTransformAttrs>();
    // We want to avoid two layout_transform ops to share the same index map even if they are
    // identical. The scope of vars used in index map initial indices is local to the op. Not doing
    // so would confuse the structural equality check.
    attrs->index_map = DeepCopyIndexMap(index_map);
    attrs->axis_separators = std::move(axis_separators);
    attrs->input_axis_separators = std::move(input_axis_separators);
    return Call(Type::Missing(), layout_transform_op_, {expr}, Attrs{std::move(attrs)}, {});
  }

  /*!
   * \brief Adds the \p remove_pad op to the module if it has not already been added before.
   * \returns The global var associated with the remove_pad PrimFunc.
   */
  GlobalVar GetOrCreateRemovePadOp(const ffi::Array<PrimExpr>& old_shape, DLDataType dtype) {
    int t_shape = old_shape.size();
    if (remove_pad_map_.count(t_shape) != 0) {
      return remove_pad_map_[t_shape];
    }
    // Create dynamic shapes for input and output tensors
    ffi::Array<PrimExpr> dyn_padded_shape, dyn_old_shape;
    for (int i = 0; i < t_shape; i++) {
      tirx::Var var1("p" + std::to_string(i), old_shape[i].ty());
      tirx::Var var2("i" + std::to_string(i), old_shape[i].ty());
      dyn_padded_shape.push_back(var1.as_or_throw<PrimExpr>());
      dyn_old_shape.push_back(var2.as_or_throw<PrimExpr>());
    }

    // Input tensor of remove_pad op
    te::Tensor placeholder_tensor = te::placeholder(dyn_padded_shape, PrimType(dtype), "input");
    // Output tensor of remove_pad op
    te::Tensor output_tensor = te::compute(
        dyn_old_shape,
        [&placeholder_tensor](const ffi::Array<tirx::PrimVar>& indices) {
          return placeholder_tensor(indices);
        },
        "output", topi::kElementWise);

    ffi::String op_name = "remove_pad";
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
                              const TensorType& old_tensor_ty,
                              const ffi::Array<IntImm>& axis_separator,
                              const ffi::Array<IntImm>& input_axis_separator) {
    if (IsScalarConstant(expr) || index_map.get() == nullptr) {
      return expr;
    }
    ffi::Array<PrimExpr> old_shape = GetShapeFromTensorType(old_tensor_ty);
    ffi::Array<Range> initial_ranges = ConstructRangeFromShape(old_shape);
    arith::Analyzer analyzer;
    auto [inverse_index_map, padding_predicate] =
        index_map.NonSurjectiveInverse(initial_ranges, analyzer);

    if (tirx::is_zero(padding_predicate)) {
      return TransformLayout(expr, inverse_index_map, axis_separator, input_axis_separator);
    } else {
      auto padded_expr = builder_->Normalize(
          TransformLayout(expr, inverse_index_map, axis_separator, input_axis_separator));
      const auto& tensor_ty = padded_expr->ty.as_or_throw<TensorType>();

      GlobalVar gv_remove_pad = GetOrCreateRemovePadOp(old_shape, tensor_ty->dtype.value()->dtype);
      return Call(Type::Missing(), call_tir_op_, {gv_remove_pad, Tuple({padded_expr})}, {},
                  {old_tensor_ty});
    }
  }

  /*!
   * \brief Adds the \p replacement_func to the module if it has not already been added before.
   * \returns The global var associated with the PrimFunc.
   */
  GlobalVar GetOrCreateGlobalVarForFunc(const PrimFunc& replacement_func,
                                        const ffi::String& op_kind) {
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
  Tuple UpdateInputs(const Tuple& inputs, const ffi::Array<IndexMap>& transforms,
                     const ffi::Optional<ffi::Array<ffi::Array<IntImm>>>& axis_separators,
                     const ffi::Optional<ffi::Array<ffi::Array<IntImm>>>& input_axis_separators) {
    if (transforms.empty()) return inputs;

    ffi::Array<Expr> updated_inputs;
    int index = 0;
    for (const auto& input : inputs->fields) {
      ffi::Array<IntImm> axis_separator;
      ffi::Array<IntImm> input_axis_separator;
      if (axis_separators.has_value()) {
        ffi::Array<ffi::Array<IntImm>> axis_separators_value = axis_separators.value();
        axis_separator = axis_separators_value[index];
      }
      if (input_axis_separators.has_value()) {
        ffi::Array<ffi::Array<IntImm>> input_axis_separators_value = input_axis_separators.value();
        input_axis_separator = input_axis_separators_value[index];
      }
      auto transform = transforms[index++];
      updated_inputs.push_back(
          TransformLayout(input, transform, axis_separator, input_axis_separator));
    }
    return Tuple(updated_inputs);
  }

  /*! \brief Updates the call_tir output type after applying buffer transforms. */
  Type UpdateOutputType(const Type& out_ty, const ffi::Array<IndexMap>& buffer_transforms) {
    if (buffer_transforms.empty()) return out_ty;

    if (out_ty->IsInstance<TensorTypeNode>())
      return UpdateOutputType(out_ty.as_or_throw<TensorType>(),
                              buffer_transforms[buffer_transforms.size() - 1]);

    TVM_FFI_ICHECK(out_ty->IsInstance<TupleTypeNode>())
        << "Expect output type of call_tir to be either TupleType or "
           "TensorType, but got "
        << out_ty;

    const auto& tuple_ty = out_ty.as_or_throw<TupleType>();
    ffi::Array<Type> ty_fields;
    size_t first_output_index = buffer_transforms.size() - tuple_ty->fields.size();
    size_t i = 0;
    for (const auto& si : tuple_ty->fields) {
      TVM_FFI_ICHECK(si->IsInstance<TensorTypeNode>())
          << "Fields of TupleType must be TensorType for call_tir "
             "output structinfo, but got "
          << si;
      ty_fields.push_back(UpdateOutputType(si.as_or_throw<TensorType>(),
                                           buffer_transforms[first_output_index + i++]));
    }
    return TupleType(ty_fields);
  }

  /*! \brief Returns the TensorType after applying the \p transform on its shape */
  Type UpdateOutputType(const TensorType& tensor_ty, const IndexMap& transform) {
    if (transform.get() == nullptr) return tensor_ty;
    auto shape = GetShapeFromTensorType(tensor_ty);
    arith::Analyzer analyzer;
    auto new_shape = transform->MapShape(shape, analyzer);
    if (tensor_ty->vdevice.has_value()) {
      return TensorType(ShapeExpr(new_shape), tensor_ty->dtype, tensor_ty->vdevice.value());
    }
    return TensorType(ShapeExpr(new_shape), tensor_ty->dtype);
  }

  Expr TransformOutputs(
      const Expr& expr, const ffi::Array<IndexMap>& buffer_transforms, const Type& old_ty,
      const ffi::Optional<ffi::Array<ffi::Array<IntImm>>>& axis_separators,
      const ffi::Optional<ffi::Array<ffi::Array<IntImm>>>& input_axis_separators) {
    if (buffer_transforms.empty()) return expr;

    ffi::Array<TensorType> old_output_ty = GetTensorTypePerOutput(old_ty);

    ffi::Array<IntImm> axis_sep, input_axis_sep;
    size_t num_outputs = old_output_ty.size();
    if (num_outputs == 0) return expr;

    size_t first_output_index = buffer_transforms.size() - num_outputs;
    // If there is a single output, return the transformed output.
    if (num_outputs == 1) {
      IndexMap output_map = buffer_transforms[first_output_index];
      if (axis_separators.has_value()) {
        ffi::Array<ffi::Array<IntImm>> axis_separators_value = axis_separators.value();
        axis_sep = axis_separators_value[first_output_index];
      }
      if (input_axis_separators.has_value()) {
        ffi::Array<ffi::Array<IntImm>> input_axis_separators_value = input_axis_separators.value();
        input_axis_sep = input_axis_separators_value[first_output_index];
      }
      return TransformLayoutInverse(expr, output_map, old_output_ty[0], axis_sep, input_axis_sep);
    }

    // In case of more than one output, we would have to get each item of the output tuple,
    // transform it and return a tuple of all transformed outputs.
    ffi::Array<Expr> transformed_outputs;
    for (size_t i = 0; i + first_output_index < buffer_transforms.size(); ++i) {
      const auto& output_map = buffer_transforms[i + first_output_index];
      if (axis_separators.has_value()) {
        ffi::Array<ffi::Array<IntImm>> axis_separators_value = axis_separators.value();
        axis_sep = axis_separators_value[i + first_output_index];
      }
      if (input_axis_separators.has_value()) {
        ffi::Array<ffi::Array<IntImm>> input_axis_separators_value = input_axis_separators.value();
        input_axis_sep = input_axis_separators_value[i + first_output_index];
      }
      auto output = builder_->Normalize(TupleGetItem(expr, static_cast<int>(i)));
      transformed_outputs.push_back(
          TransformLayoutInverse(output, output_map, old_output_ty[i], axis_sep, input_axis_sep));
    }
    return Tuple(transformed_outputs);
  }

 private:
  /*! \brief Cache to keep track of the GlobalVar associated with the new PrimFunc added */
  ffi::Map<PrimFunc, GlobalVar> cache_;
  /*! \brief Input IRModule */
  const IRModule& mod_;
  /*! \brief Map from shape_dim.size to the remove_pad GlobalVar */
  std::unordered_map<int, GlobalVar> remove_pad_map_;
  /*! \brief Map from kOperatorName attribute to the replacement PrimFunc */
  const ffi::Map<ffi::String, PrimFunc>& op_impl_map_;
  /*! \brief Map from kOperatorName attribute to the layout transforms on i/o buffers */
  const ffi::Map<ffi::String, ffi::Array<IndexMap>>& op_buffer_transforms__;
  /*! \brief Map from kOperatorName attribute to the axis separatos on i/o buffers */
  const ffi::Map<ffi::String, ffi::Optional<ffi::Array<ffi::Array<IntImm>>>>&
      op_buffer_axis_separators__;
  /*! \brief Map from kOperatorName attribute to the input axis separatos */
  const ffi::Map<ffi::String, ffi::Optional<ffi::Array<ffi::Array<IntImm>>>>&
      op_buffer_input_axis_separators__;

  const Op& call_tir_op_ = Op::Get("relax.call_tir");
  const Op& layout_transform_op_ = Op::Get("relax.layout_transform");
};

namespace transform {

Pass AlterOpImpl(
    const ffi::Map<ffi::String, tirx::PrimFunc>& op_impl_map,
    const ffi::Map<ffi::String, ffi::Array<IndexMap>>& op_buffer_transforms_,
    const ffi::Map<ffi::String, ffi::Optional<ffi::Array<ffi::Array<IntImm>>>>& axis_separators_,
    const ffi::Map<ffi::String, ffi::Optional<ffi::Array<ffi::Array<IntImm>>>>&
        input_axis_separators_) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return AlterOpImplMutator(mod, op_impl_map, op_buffer_transforms_, axis_separators_,
                              input_axis_separators_)
        .Run();
  };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"AlterOpImpl",  //
                          /*required=*/{});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AlterOpImpl", AlterOpImpl);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
