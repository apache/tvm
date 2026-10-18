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
 * \file src/tirx/ir/specialize.cc
 * \brief Specialize parameters of Function.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_set>

#include "../transform/ir_utils.h"

namespace tvm {
namespace tirx {

using VarMap = std::unordered_map<Var, Expr>;

/**************** Helper functions ****************/

/*! \brief Helper function to check whether the given var is in function parameter list. */
inline bool IsParam(const Function& func, const Var& param) {
  return std::any_of(func->params.begin(), func->params.end(),
                     [&](const Var& var) { return var.same_as(param); });
}

/**************** Specializer ****************/

// Try fold constants if op's child get specialized to constant.
#define DEFINE_SPECIALIZER_BINARY_OP_MUTATE(BinaryNode, BinaryFunc)                     \
  UnchangedOr<PrimExpr> Mutate_(const BinaryNode* op, InplaceMode inplace_mode) final { \
    auto a_result = Mutate(op->a, inplace_mode);                                        \
    bool a_unchanged = a_result.UnchangedOrSameAs(op->a);                               \
    PrimExpr a = std::move(a_result).ValueOrUnchanged(op->a);                           \
    auto b_result = Mutate(op->b, inplace_mode);                                        \
    bool b_unchanged = b_result.UnchangedOrSameAs(op->b);                               \
    PrimExpr b = std::move(b_result).ValueOrUnchanged(op->b);                           \
    if (a_unchanged && b_unchanged) {                                                   \
      return ffi::Unchanged();                                                          \
    } else {                                                                            \
      return BinaryFunc(a, b, op->loc);                                                 \
    }                                                                                   \
  }
#define DEFINE_SPECIALIZER_UNARY_OP_MUTATE(UnaryNode, UnaryFunc)                       \
  UnchangedOr<PrimExpr> Mutate_(const UnaryNode* op, InplaceMode inplace_mode) final { \
    auto a_result = Mutate(op->a, inplace_mode);                                       \
    bool a_unchanged = a_result.UnchangedOrSameAs(op->a);                              \
    PrimExpr a = std::move(a_result).ValueOrUnchanged(op->a);                          \
    if (a_unchanged) {                                                                 \
      return ffi::Unchanged();                                                         \
    } else {                                                                           \
      return UnaryFunc(a, op->loc);                                                    \
    }                                                                                  \
  }

/*! \brief Mutator to specialize function and remove const parameters */
class FunctionSpecializer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit FunctionSpecializer(const VarMap& var_map) {
    for (const auto& [var, value] : var_map) {
      if (!var.as<TensorVar>()) VarRemapSet(var, value);
    }
  }

  static Function Specialize(Function f, const VarMap& var_map) {
    auto specializer = ffi::make_object<FunctionSpecializer>(var_map);
    for (const Var& param : f->params) {
      auto tensor = param.as<TensorVar>();
      auto replacement = var_map.find(param);
      if (!tensor || replacement == var_map.end()) {
        continue;
      }
      if (auto replacement_var = replacement->second.as<Var>()) {
        if (auto replacement_tensor = replacement_var.value().as<TensorVar>()) {
          if (IsParam(f, replacement_var.value())) {
            specializer->VarRemapSet(tensor.value(), replacement_tensor.value());
          } else {
            specializer->constrained_tensor_params_.insert(param.get());
            specializer->tensor_storage_scopes_.emplace(tensor.value(),
                                                        replacement_tensor.value().scope());
          }
        }
      }
    }

    // Updating parameters
    ffi::Array<Var> params;
    bool param_updated = false;
    for (const auto& var : f->params) {
      Var new_var = var;
      if (auto tensor = var.as<TensorVar>()) {
        TensorVar new_tensor = specializer->MutateTensor(tensor.value());
        new_var = new_tensor;
        if (!new_tensor.same_as(tensor.value())) {
          param_updated = true;
          specializer->VarRemapSet(tensor.value(), new_tensor);
          specializer->defined_tensors_.insert(tensor.value().get());
        }
      }
      // Remove parmeters which has been specialized.
      if (var_map.find(var) == var_map.end() ||
          specializer->constrained_tensor_params_.count(var.get())) {
        params.push_back(new_var);
      } else {
        param_updated = true;
      }
    }

    auto planner = ffi::make_object<TensorPlanner>(specializer.get());
    planner->Visit(f->body);

    auto body_result =
        specializer->Mutate(f->body, f.unique() ? InplaceMode::kAllow : InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(f->body);
    auto body = std::move(body_result).ValueOrUnchanged(f->body);

    if (param_updated || !body_unchanged) {
      return Function(params, body, f->ret_type, f->attrs, f->loc);
    } else {
      return f;
    }
  }

 private:
  class TensorPlanner : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;

    explicit TensorPlanner(FunctionSpecializer* specializer) : specializer_(specializer) {}

   private:
    ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
      if (op->ty.as<TensorTypeNode>()) {
        if (def_region_kind() == kTVMFFIDefRegionKindSimple) {
          const TensorVar tensor = ffi::GetRef<Var>(op).as_or_throw<TensorVar>();
          specializer_->MutateAllocTensor(tensor);
        } else {
          specializer_->ValidateTensorUse(ffi::GetRef<Var>(op).as_or_throw<TensorVar>());
        }
      }
      return StmtExprVisitor::Visit_(op);
    }

    ffi::Optional<VisitInterrupt> Visit_(const BindNode* op) final {
      if (const auto* call = op->value.as<CallNode>();
          call &&
          (call->op.same_as(tirx::alloc_tensor_op()) || call->op.same_as(tirx::decl_tensor_op()))) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
            kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(op->var); }));
        // The definition above owns traversal of the complete tensor type.
        // Value operands include declaration data or allocation placement.
        return Visit(call->args);
      }
      return StmtExprVisitor::Visit_(op);
    }

    FunctionSpecializer* specializer_;
  };

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (!result.IsUnchanged()) {
      op = ffi::AnyView(result).as<CallNode>();
      if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
    }
    if (!op->op.same_as(tirx::tensor_data_ptr_op()) || op->args.size() != 1) return result;
    PtrType type = op->args[0].as_or_throw<TensorVar>().ty()->DataPtrType();
    if (ffi::StructuralEqual()(op->ty, type)) return result;
    if (inplace_mode == InplaceMode::kAllow) {
      const_cast<CallNode*>(op)->ty = std::move(type);
      return result;
    }
    auto copy = ffi::make_object<CallNode>(*op);
    copy->ty = std::move(type);
    return Expr(std::move(copy));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto result = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow);
    if (result.UnchangedOrSameAs(ffi::GetRef<PrimExpr>(op))) return ffi::Unchanged();
    auto load = std::move(result).ValueUnchecked().as_or_throw<TensorLoad>();
    if (auto tensor = load->source.as<TensorVar>()) {
      return TensorLoad(tensor.value(), load->indices, load->loc);
    }
    return load;
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode* op, InplaceMode inplace_mode) final {
    auto result = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow);
    if (result.UnchangedOrSameAs(ffi::GetRef<Expr>(op))) return ffi::Unchanged();
    auto region = std::move(result).ValueUnchecked().as_or_throw<TensorRegion>();
    if (auto tensor = region->source.as<TensorVar>()) {
      return TensorRegion(tensor.value(), region->region, TensorRegionType(), region->loc);
    }
    return region;
  }

  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::AddNode, add);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::SubNode, sub);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::MulNode, mul);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::DivNode, div);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::ModNode, truncmod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::FloorDivNode, floordiv);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::FloorModNode, floormod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::MaxNode, max);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::MinNode, min);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::EQNode, equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::NENode, not_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::LTNode, less);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::LENode, less_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::GTNode, greater);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::GENode, greater_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::AndNode, logical_and);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::OrNode, logical_or);
  DEFINE_SPECIALIZER_UNARY_OP_MUTATE(prim::NotNode, logical_not);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::LShiftNode, left_shift);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::RShiftNode, right_shift);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::BitwiseAndNode, bitwise_and);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::BitwiseOrNode, bitwise_or);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(prim::BitwiseXorNode, bitwise_xor);
  DEFINE_SPECIALIZER_UNARY_OP_MUTATE(prim::BitwiseNotNode, prim::BitwiseNot);
  TensorVar MutateTensor(const TensorVar& tensor) {
    ffi::Any mapped = VarRemapGet(tensor);
    if (mapped.type_index() != ffi::TypeIndex::kTVMFFINone) {
      return std::move(mapped).as_or_throw<UnchangedOr<TensorVar>>().ValueOrUnchanged(tensor);
    }

    ffi::Optional<runtime::MemoryScope> specialized_storage_scope;
    if (auto it = tensor_storage_scopes_.find(tensor); it != tensor_storage_scopes_.end()) {
      specialized_storage_scope = it->second;
    }

    ffi::Array<PrimExpr> shape =
        tensor.shape().Map([this](const PrimExpr& e) { return Mutate(e).ValueOrUnchanged(e); });
    ffi::Optional<Layout> layout = tensor.ty()->layout;
    auto mapped_layout = Mutate(ffi::AnyView(layout), InplaceMode::kDisallow);
    bool layout_changed = !mapped_layout.UnchangedOrSameAs(layout);
    layout = std::move(mapped_layout).ValueOrUnchanged(layout).cast<ffi::Optional<Layout>>();

    bool storage_scope_changed = specialized_storage_scope.has_value() &&
                                 specialized_storage_scope.value() != tensor.scope();
    if (tensor.shape().same_as(shape) && !layout_changed && !storage_scope_changed) {
      return tensor;
    } else {
      auto n = ffi::make_object<TensorTypeNode>(*tensor.ty().get());
      n->shape = std::move(shape);
      if (layout_changed) {
        n->layout = std::move(layout);
      }
      if (storage_scope_changed) {
        n->scope = specialized_storage_scope.value();
      }
      return TensorVar(tensor->name, TensorType(std::move(n)), tensor->loc);
    }
  }

  void MutateAllocTensor(const TensorVar& alloc_buf) {
    TVM_FFI_ICHECK(defined_tensors_.insert(alloc_buf.get()).second)
        << "Multiple points of definition found for tensor " << alloc_buf;
    VarRemapSet(alloc_buf, MutateTensor(alloc_buf));
  }

  void ValidateTensorUse(const TensorVar& old_tensor) {
    if (VarRemapGet(old_tensor).type_index() != ffi::TypeIndex::kTVMFFINone) return;

    auto mutated = MutateTensor(old_tensor);
    TVM_FFI_ICHECK(mutated.same_as(old_tensor))
        << "TensorVar " << old_tensor << " (shape = " << old_tensor.shape() << ")"
        << " was used without a declaration, "
        << "and would be specialized into " << mutated << " (shape = " << mutated.shape() << ").  "
        << "While usage of an undeclared tensor is currently allowed in TIR, "
        << "mutation must occur at the tensor's point of definition "
        << "(see discussion on https://github.com/apache/tvm/pull/14565 for more details).  "
        << "Please add a definition for this tensor, "
        << "either as a TensorType-annotated Function parameter, "
        << "in a block's tensor allocations, "
        << "or in a DeclTensor statement.";
  }

  /*! \brief Definition identities used only to validate declaration order. */
  std::unordered_set<const VarNode*> defined_tensors_;
  /*! \brief Storage constraints supplied by concrete, non-parameter tensors. */
  std::unordered_map<TensorVar, runtime::MemoryScope, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      tensor_storage_scopes_;
  /*! \brief Tensor parameters constrained by a concrete, non-parameter tensor. */
  std::unordered_set<const VarNode*> constrained_tensor_params_;
};

/*!
 * \brief Update Specialize var map with tensor matching.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_buf The matching tensor.
 * \param var_map The var mapping to be updated.
 * \note This function will match target tensor's shape, strides and element_offset
 *   For example, we define a tensor in Function:
 *   A: T.Tensor([m, n])
 *
 *   Then we match it with a tensor B =  Var("tensor", TensorType((8, 16), "float32"))
 *
 *   It means we have two var mappings here: m = 8 and n = 16
 *
 *   If the tensor signature is not a Var, the mapping will fail.
 *   e.g. A: T.Tensor([m * 2, n + 1])
 */
void UpdateSpecializeVarMap(const Function& func, const Var& param, const TensorVar& specific_buf,
                            VarMap* var_map) {
  // preliminaries
  prim::ExprDeepEqual equal;

  auto opt_tensor = param.as<TensorVar>();
  TVM_FFI_CHECK(opt_tensor, ValueError)
      << "specialize expects param to have a TensorType annotation";
  const TensorVar& buf_to_specialize = opt_tensor.value();

  // build var mapping using specific_buf's parameters
  auto build_var_mapping = [&](const Expr& new_expr, const Expr& old_expr) {
    auto expr_equal = [&](const Expr& lhs, const Expr& rhs) {
      if (auto lhs_prim = lhs.as<PrimExpr>()) {
        auto rhs_prim = rhs.as<PrimExpr>();
        return rhs_prim && equal(lhs_prim.value(), rhs_prim.value());
      }
      return lhs.same_as(rhs);
    };
    Expr resolved = old_expr;
    if (auto prim_expr = old_expr.as<PrimExpr>()) {
      auto mutator = ffi::make_object<FunctionSpecializer>(*var_map);
      resolved = mutator->Mutate(prim_expr.value()).ValueOrUnchanged(prim_expr.value());
    }
    if (!expr_equal(new_expr, resolved)) {
      auto maybe_var = old_expr.as<Var>();
      TVM_FFI_CHECK(maybe_var, TypeError)
          << "The signature of target tensor exprected an independent Var, but got " << old_expr
          << ".";
      const Var& var = maybe_var.value();
      auto it = var_map->find(var);
      if (it != var_map->end()) {
        TVM_FFI_CHECK(expr_equal(it->second, new_expr), ValueError)
            << "The assigned value of var " << var << " mismatched. " << it->second << " vs. "
            << new_expr << ".";
      } else {
        var_map->insert_or_assign(var, new_expr);
      }
    }
  };

  // Check tensor dimensions
  TVM_FFI_CHECK(specific_buf.shape().size() == buf_to_specialize.shape().size(), ValueError)
      << "The tensor dimensions mismatched" << buf_to_specialize.shape().size() << " vs. "
      << specific_buf.shape().size() << ".";

  for (size_t i = 0; i < specific_buf.shape().size(); ++i) {
    build_var_mapping(specific_buf.shape()[i], buf_to_specialize.shape()[i]);
  }
  auto specific = MatchShapeStridePattern(specific_buf.ty());
  auto original = MatchShapeStridePattern(buf_to_specialize.ty());
  TVM_FFI_CHECK(specific.has_value() && original.has_value(), ValueError)
      << "Tensor specialization requires shape-aligned strided memory layouts";
  auto new_strides = specific.value().get<1>();
  auto old_strides = original.value().get<1>();
  for (size_t i = 0; i < old_strides.size(); ++i) {
    build_var_mapping(new_strides[i], old_strides[i]);
  }
  build_var_mapping(specific.value().get<2>(), original.value().get<2>());
  // The specializer distinguishes a concrete tensor constraint (which keeps
  // the parameter) from another function parameter (which aliases it).
  build_var_mapping(specific_buf, buf_to_specialize);

  // Check data_alignment.
  // These two signatures are int, so we do not need map them.
  TVM_FFI_CHECK_EQ(specific_buf.ty()->data_alignment, buf_to_specialize.ty()->data_alignment,
                   ValueError)
      << "The tensor data_alignment mismatched" << buf_to_specialize.ty()->data_alignment << " vs. "
      << specific_buf.ty()->data_alignment << ".";
}

/*!
 * \brief Update Specialize var map with parameter value.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_expr The parameter value.
 * \param var_map The var mapping to be updated.
 */
void UpdateSpecializeVarMap(const Function& func, const Var& param, const Expr& specific_expr,
                            VarMap* var_map) {
  // check param is in Function's parameters
  TVM_FFI_CHECK(IsParam(func, param), ValueError)
      << "Specialize expects param to be in Function's params";
  // Specialize a scalar parameter rather than a tensor parameter.
  TVM_FFI_CHECK(!param.as<TensorVar>(), ValueError)
      << "Specialize expects param to not have a TensorType annotation";
  // build var mapping using specific_expr
  var_map->insert_or_assign(param, specific_expr);
}

/**************** Implementation ****************/

Function Specialize(Function func, const ffi::Map<Var, ffi::Variant<TensorVar, Expr>>& param_map) {
  VarMap var_map;
  for (const auto& kv : param_map) {
    const Var& param = kv.first;
    const ffi::Variant<TensorVar, Expr>& instance = kv.second;
    if (auto opt_tensor = instance.as<TensorVar>()) {
      UpdateSpecializeVarMap(func, param, opt_tensor.value(), &var_map);
    } else if (auto opt_expr = instance.as<Expr>()) {
      UpdateSpecializeVarMap(func, param, opt_expr.value(), &var_map);
    } else {
      TVM_FFI_THROW(TypeError) << "specialize expected instance to be TensorVar or Expr";
    }
  }
  return FunctionSpecializer::Specialize(func, std::move(var_map));
}

/**************** FFI ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Specialize", [](Function func,
                                              const ffi::Map<Var, ffi::Any>& param_map) {
    VarMap var_map;
    for (const auto& [param, instance] : param_map) {
      if (auto tensor = instance.as<TensorVar>()) {
        UpdateSpecializeVarMap(func, param, tensor.value(), &var_map);
      } else if (const ExprNode* expr = instance.as<ExprNode>()) {
        UpdateSpecializeVarMap(func, param, ffi::GetRef<Expr>(expr), &var_map);
      } else if (instance.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
        UpdateSpecializeVarMap(func, param, instance.cast<PrimExpr>(), &var_map);
      } else {
        TVM_FFI_THROW(TypeError) << "specialize expected instance to be TensorVar or Expr, but got "
                                 << instance.GetTypeKey();
      }
    }
    return FunctionSpecializer::Specialize(func, std::move(var_map));
  });
}

}  // namespace tirx
}  // namespace tvm
