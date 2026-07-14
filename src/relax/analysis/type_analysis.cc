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
 * \file type_analysis.cc
 * \brief Implementations of foundational Relax type analysis.
 *
 * \note Update this file when you added a new Type.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type_functor.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace relax {

//--------------------------
// GetStaticType
//--------------------------
class StaticTypeDeriver : public TypeFunctor<Type(const Type&)> {
 public:
  Type VisitType_(const AnyTypeNode* op) final { return AnyType(op->span); }

  Type VisitType_(const PrimTypeNode* op) final { return tvm::PrimType(op->dtype); }

  Type VisitType_(const ShapeTypeNode* op) final { return ShapeType(op->ndim, op->span); }

  Type VisitType_(const TensorTypeNode* op) final {
    return TensorType(op->dtype, op->ndim, op->vdevice, op->span);
  }

  // module: distributed
  Type VisitType_(const distributed::DTensorTypeNode* op) final { return AnyType(); }
  // end-module: distributed

  Type VisitType_(const TupleTypeNode* op) final {
    ffi::Array<Type> fields =
        op->fields.Map([this](const Type& ty) { return this->VisitType(ty); });
    return TupleType(fields, op->span);
  }

  Type VisitType_(const FuncTypeNode* op) final {
    if (op->IsOpaque()) return PackedFuncType(op->span);
    ffi::Array<Type> params =
        op->params.value().Map([this](const Type& ty) { return this->VisitType(ty); });
    Type ret = this->VisitType(op->ret);
    return FuncType(params, ret, op->purity, op->span);
  }
};

Type GetStaticType(const Type& info) { return StaticTypeDeriver()(info); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.GetStaticType",
                        [](const Type& info) { return GetStaticType(info); });
}

//--------------------------
// TypeFromStaticType
//--------------------------

Type TypeFromStaticType(const Type& type) {
  if (type.as<AnyTypeNode>()) {
    return AnyType(type->span);
  } else if (const PrimTypeNode* prim_type = type.as<PrimTypeNode>()) {
    return tvm::PrimType(prim_type->dtype);
  } else if (const tvm::PrimTypeNode* prim_type = type.as<tvm::PrimTypeNode>()) {
    return tvm::PrimType(prim_type->dtype);
  } else if (const ShapeTypeNode* shape_type = type.as<ShapeTypeNode>()) {
    return ShapeType(shape_type->ndim, type->span);
  } else if (const TensorTypeNode* tensor_type = type.as<TensorTypeNode>()) {
    return TensorType(tensor_type->dtype, tensor_type->ndim);
  } else if (const TupleTypeNode* tuple_type = type.as<TupleTypeNode>()) {
    ffi::Array<Type> fields;
    for (const Type& field : tuple_type->fields) {
      fields.push_back(TypeFromStaticType(field));
    }
    return TupleType(fields, type->span);
  } else if (const FuncTypeNode* func_type = type.as<FuncTypeNode>()) {
    if (func_type->IsOpaque()) return FuncType::OpaqueFunc(func_type->ret, func_type->purity);
    ffi::Array<Type> params =
        func_type->params.value().Map([](const Type& param) { return TypeFromStaticType(param); });
    Type ret = TypeFromStaticType(func_type->ret);
    return FuncType(params, ret, func_type->purity, func_type->span);
  } else if (const tvm::FuncTypeNode* func_type = type.as<tvm::FuncTypeNode>()) {
    ffi::Array<Type> params =
        func_type->arg_types.Map([](const Type& param) { return TypeFromStaticType(param); });
    Type ret = TypeFromStaticType(func_type->ret_type);
    // TODO(relax-team): Maybe add purity into the type as well
    return FuncType(params, ret, true, func_type->span);
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported type: " << type;
    return Type::Missing();
  }
}

//--------------------------
// EraseToWellDefined
//--------------------------
class WellDefinedEraser : public TypeMutator, public ExprMutatorBase {
 public:
  WellDefinedEraser(std::function<ffi::Optional<Expr>(const Var& var)> f_var_map,
                    arith::AnalyzerObj* ana)
      : f_var_map_(f_var_map), ana_(ana) {}

  Type VisitType_(const PrimTypeNode* op) final { return ffi::GetRef<Type>(op); }

  Type VisitType_(const ShapeTypeNode* op) final {
    bool has_undefined = false;
    ffi::Optional<ffi::Array<PrimExpr>> values;

    if (op->values.has_value()) {
      std::swap(has_undefined_, has_undefined);
      values = op->values.value().Map([&](PrimExpr val) { return VisitPrimitiveExpr(val); });
      std::swap(has_undefined_, has_undefined);
    }
    // erase symbolic shape if we have undefined.
    if (!has_undefined) {
      if (values.same_as(op->values)) {
        return ffi::GetRef<Type>(op);
      } else {
        return ShapeType(values.value(), op->span);
      }
    } else {
      return ShapeType(op->ndim, op->span);
    }
  }

  Type VisitType_(const TensorTypeNode* op) final {
    bool has_undefined = false;
    ffi::Optional<Expr> shape;

    if (op->shape.has_value()) {
      std::swap(has_undefined_, has_undefined);
      shape = relax::ExprMutatorBase::VisitExpr(op->shape.value());
      std::swap(has_undefined_, has_undefined);
    }

    VDevice vdev = op->vdevice.value_or(VDevice());

    // erase symbolic shape if we have undefined.
    if (!has_undefined) {
      if (shape.same_as(op->shape)) {
        return ffi::GetRef<Type>(op);
      } else {
        if (shape.has_value()) {
          return TensorType(shape.value(), op->dtype, vdev, op->span);
        } else {
          return TensorType(op->dtype, op->ndim, vdev, op->span);
        }
      }
    } else {
      return TensorType(op->dtype, op->ndim, vdev, op->span);
    }
  }

  Type VisitType_(const FuncTypeNode* op) final {
    // NOTE: we always require func type to be well-defined.
    //
    // All the occuring symbolic variables are defined in parameters'
    // type annotations. So there is no needed to erase.
    return ffi::GetRef<Type>(op);
  }

  using relax::ExprMutatorBase::VisitExpr_;

  PrimExpr VisitPrimitiveExpr(const PrimExpr& expr) {
    PrimExpr val = tirx::Substitute(expr, [this](const Var& var) -> ffi::Optional<Expr> {
      if (var.as<DataflowVarNode>()) {
        has_undefined_ = true;
        return std::nullopt;
      }
      ffi::Optional<Expr> ret = f_var_map_ == nullptr ? std::nullopt : f_var_map_(var);
      has_undefined_ = has_undefined_ || !ret.has_value();
      if (!ret.has_value()) return std::nullopt;

      PrimExpr value = ret.value().as_or_throw<PrimExpr>();
      if (value->IsInstance<IntImmNode>()) {
        return tvm::cast(PrimType::Int(64), value);
      }
      TVM_FFI_ICHECK(value.ty().MatchesElementType(DLDataTypeCode::kDLInt, 64))
          << "Can only provide i64 expressions in shape";
      return value;
    });
    if (!val.same_as(expr)) {
      return ana_->Simplify(val);
    } else {
      return val;
    }
  }

  Expr VisitExpr_(const ShapeExprNode* op) final {
    ffi::Array<PrimExpr> values =
        op->values.Map([this](const PrimExpr& expr) { return VisitPrimitiveExpr(expr); });
    return values.same_as(op->values) ? ffi::GetRef<Expr>(op) : ShapeExpr(values, op->span);
  }

  Expr VisitExpr_(const VarNode* var) final {
    Var id = ffi::GetRef<Var>(var);
    ffi::Optional<Expr> ret;
    if (f_var_map_ != nullptr) {
      ret = f_var_map_(id);
    }

    has_undefined_ = has_undefined_ || !ret.has_value();
    if (ret.has_value()) {
      TVM_FFI_ICHECK((ret.as<VarNode>() && !ret.as<DataflowVarNode>()) || ret.as<ShapeExprNode>())
          << "Only allow Expr in Type to be ShapeExpr or Var";
    }
    return ret.value_or(ffi::GetRef<Expr>(var));
  }

  Expr VisitExpr_(const DataflowVarNode* var) final {
    has_undefined_ = true;
    return ffi::GetRef<Expr>(var);
  }

 private:
  bool has_undefined_ = false;
  std::function<ffi::Optional<Expr>(const Var& var)> f_var_map_;
  arith::AnalyzerObj* ana_;
};

Type EraseToWellDefined(const Type& info,
                        std::function<ffi::Optional<Expr>(const Var& var)> f_var_map) {
  arith::Analyzer analyzer;
  return EraseToWellDefined(info, f_var_map, analyzer);
}

Type EraseToWellDefined(const Type& info,
                        std::function<ffi::Optional<Expr>(const Var& var)> f_var_map,
                        const arith::Analyzer& ana) {
  return WellDefinedEraser(f_var_map, ana.get()).VisitType(info);
}

Type EraseToWellDefined(const Type& info, ffi::Map<Var, Expr> var_map) {
  arith::Analyzer analyzer;
  return EraseToWellDefined(info, var_map, analyzer);
}

Type EraseToWellDefined(const Type& info, ffi::Map<Var, Expr> var_map, const arith::Analyzer& ana) {
  std::function<ffi::Optional<Expr>(const Var& var)> f_var_map = nullptr;

  if (!var_map.empty()) {
    f_var_map = [&](const Var& var) -> ffi::Optional<Expr> {
      auto it = var_map.find(var);
      if (it != var_map.end()) return (*it).second;
      return std::nullopt;
    };
  }

  return EraseToWellDefined(info, f_var_map, ana);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.EraseToWellDefined",
                        [](const Type& info, ffi::Map<tirx::Var, PrimExpr> shape_var_map,
                           ffi::Map<Var, Expr> var_map) {
                          for (const auto& [var, value] : shape_var_map) {
                            TVM_FFI_CHECK(var.as<tirx::PrimVar>(), TypeError)
                                << "Expected an exact primitive Var, but received " << var;
                            var_map.Set(var, value);
                          }
                          return EraseToWellDefined(info, var_map);
                        });
}

//--------------------------
// IsBaseOf
//--------------------------
class TypeBaseChecker : public TypeFunctor<BaseCheckResult(const Type&, const Type&)> {
 public:
  explicit TypeBaseChecker(arith::AnalyzerObj* ana) : analyzer_(ana) {}

  BaseCheckResult VisitType(const Type& lhs, const Type& other) override {
    // quick path
    // Note: subclass may disable this quick path if we need to go over all type.
    if (lhs.same_as(other)) return BaseCheckResult::kPass;
    return TypeFunctor::VisitType(lhs, other);
  }

  // AnyType is base of every Relax type
  BaseCheckResult VisitType_(const AnyTypeNode* lhs, const Type& other) final {
    return BaseCheckResult::kPass;
  }

  BaseCheckResult VisitType_(const PrimTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<PrimTypeNode>();
    if (rhs == nullptr) {
      if (other.as<AnyTypeNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    if (lhs->dtype != rhs->dtype) {
      return BaseCheckResult::kFailL0;
    }

    return BaseCheckResult::kPass;
  }

  BaseCheckResult VisitType_(const ShapeTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<ShapeTypeNode>();
    if (rhs == nullptr) {
      if (other.as<AnyTypeNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    // lhs have unknown ndim
    if (lhs->IsUnknownNdim()) return BaseCheckResult::kPass;

    // ndim must match
    if (lhs->ndim != rhs->ndim) {
      if (rhs->IsUnknownNdim()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    // lhs does not have symbolic value
    if (!lhs->values.has_value()) return BaseCheckResult::kPass;
    // rhs does not have symbolic value but lhs do.
    if (!rhs->values.has_value()) return BaseCheckResult::kFailL2;

    // shape match check
    return ShapeMatchCheck(lhs->values.value(), rhs->values.value());
  }

  BaseCheckResult VisitType_(const TensorTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<TensorTypeNode>();
    if (rhs == nullptr) {
      if (other.as<AnyTypeNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    // dtype mismatch
    if (!lhs->IsUnknownDtype() && !rhs->IsUnknownDtype() &&
        lhs->dtype.value() != rhs->dtype.value()) {
      if (rhs->IsUnknownDtype()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    if (!lhs->IsUnknownDtype() && rhs->IsUnknownDtype()) return BaseCheckResult::kFailL1;

    // ndim mismatch
    if (!lhs->IsUnknownNdim() && lhs->ndim != rhs->ndim) {
      if (rhs->IsUnknownNdim()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    // vdevice mismatch
    if (lhs->vdevice.has_value() && !rhs->vdevice.has_value()) return BaseCheckResult::kFailL1;
    if (lhs->vdevice.has_value() && rhs->vdevice.has_value()) {
      VDevice lhs_vdevice = lhs->vdevice.value();
      VDevice rhs_vdevice = rhs->vdevice.value();
      if (lhs_vdevice->target.defined() && !rhs_vdevice->target.defined())
        return BaseCheckResult::kFailL1;
      // mismatch in either the target, vdevice_id, or memory_scope
      if ((lhs_vdevice->target.defined() && rhs_vdevice->target.defined()) &&
          (lhs_vdevice->target != rhs_vdevice->target ||
           lhs_vdevice->vdevice_id != rhs_vdevice->vdevice_id ||
           lhs_vdevice->memory_scope != rhs_vdevice->memory_scope))
        return BaseCheckResult::kFailL0;
    }

    // lhs does not have defined shape and everything else matches
    if (!lhs->shape.has_value()) return BaseCheckResult::kPass;
    // rhs does not have symbolic value but lhs don't
    if (!rhs->shape.has_value()) return BaseCheckResult::kFailL2;

    // shape match check
    return ShapeMatchCheck(lhs->shape.value(), rhs->shape.value());
  }

  // module: distributed
  BaseCheckResult VisitType_(const distributed::DTensorTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<distributed::DTensorTypeNode>();
    if (rhs == nullptr) {
      if (other.as<AnyTypeNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    BaseCheckResult tensor_ty_check_result = this->VisitType(lhs->tensor_ty, rhs->tensor_ty);
    BaseCheckResult other_check_result;
    if (!struct_equal_(lhs->device_mesh, rhs->device_mesh) ||
        !struct_equal_(lhs->placement, rhs->placement)) {
      other_check_result = BaseCheckResult::kFailL1;
    } else {
      other_check_result = BaseCheckResult::kPass;
    }
    return CombineCheck(tensor_ty_check_result, other_check_result);
  }
  // end-module: distributed

  BaseCheckResult VisitType_(const TupleTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<TupleTypeNode>();
    if (rhs == nullptr) {
      if (other.as<AnyTypeNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    return ArrayCheck(lhs->fields, rhs->fields);
  }

  BaseCheckResult VisitType_(const FuncTypeNode* lhs, const Type& other) override {
    auto* rhs = other.as<FuncTypeNode>();
    if (rhs == nullptr) {
      if (other.as<AnyTypeNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    // Check purity: Pure functions are a subtype of impure functions
    if (lhs->purity && !rhs->purity) {
      return BaseCheckResult::kFailL0;
    }

    // lhs opaque handling
    if (lhs->IsOpaque()) {
      if (lhs->derive_func.has_value()) {
        // function proving is best effort.
        return lhs->derive_func.same_as(rhs->derive_func) ? BaseCheckResult::kPass
                                                          : BaseCheckResult::kFailL2;
      }
      // no derivation function, only depends on ret
      return this->VisitType(lhs->ret, rhs->ret);
    }

    // Function check is best effort.
    // rhs is opaque but lhs is not
    if (rhs->IsOpaque()) return BaseCheckResult::kFailL2;

    // NOTE: lhs->params, rhs->params may contain different symbolic
    // vars that needs to be re-mapped to each other.
    // This can only be done through structural equality check and not ArrayCheck.
    //
    // So we check structural equality here and if two are structurally
    // equal return true.
    //
    // otherwise we do best effort BaseArrayCheck.
    //
    // This still does not handle cases where some arguments are sub of another
    // while other parameters needs to get remapped.
    //
    // Given we only do best effort checking in these cases, and such cases
    // are likely not a primary concern atm, we take this approach here.
    if (struct_equal_(ffi::GetRef<Type>(lhs), other)) return BaseCheckResult::kPass;

    auto param_check = FuncParamsCheck(lhs->params.value(), rhs->params.value());
    auto ret_check = this->VisitType(lhs->ret, rhs->ret);
    return CombineCheck(param_check, ret_check);
  }

 protected:
  // analyzer
  arith::AnalyzerObj* analyzer_;
  // struct equal checker
  ffi::StructuralEqual struct_equal_;

  // customizable functions.
  /*!
   * \brief Check symbolic shape value equivalence.
   * \param lhs The left hand shape.
   * \param rhs The right hand shape.
   * \return CheckResult.
   */
  virtual BaseCheckResult PrimExprMatchCheck(const PrimExpr& lhs, const PrimExpr& rhs) {
    // get static shape checking right.
    auto* int_lhs = lhs.as<IntImmNode>();
    auto* int_rhs = rhs.as<IntImmNode>();
    if (int_lhs && int_rhs) {
      if (int_lhs->value == int_rhs->value) {
        return BaseCheckResult::kPass;
      } else {
        return BaseCheckResult::kFailL0;
      }
    }
    return analyzer_->CanProveEqual(lhs, rhs) ? BaseCheckResult::kPass : BaseCheckResult::kFailL2;
  }
  /*!
   * \brief CheckShape value.
   * \param lhs The left hand shape.
   * \param rhs The right hand shape.
   * \return CheckResult.
   */
  virtual BaseCheckResult ShapeMatchCheck(const ffi::Array<PrimExpr>& lhs,
                                          const ffi::Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) return BaseCheckResult::kFailL0;

    BaseCheckResult ret = BaseCheckResult::kPass;
    for (size_t i = 0; i < lhs.size(); ++i) {
      auto cmp_ret = PrimExprMatchCheck(lhs[i], rhs[i]);
      if (ret == BaseCheckResult::kFailL0) return ret;
      ret = CombineCheck(cmp_ret, ret);
    }
    return ret;
  }

  /*!
   * \brief CheckShape value.
   * \param lhs The left hand shape.
   * \param rhs The right hand shape.
   * \return Check result.
   */
  virtual BaseCheckResult ShapeMatchCheck(const Expr& lhs, const Expr& rhs) {
    if (lhs.same_as(rhs)) return BaseCheckResult::kPass;
    auto* lhs_shape = lhs.as<ShapeExprNode>();
    auto* rhs_shape = rhs.as<ShapeExprNode>();
    if (lhs_shape && rhs_shape) {
      return ShapeMatchCheck(lhs_shape->values, rhs_shape->values);
    } else {
      return BaseCheckResult::kFailL2;
    }
  }

  /*!
   * \brief CheckShape function parameters.
   * \param lhs The left hand params.
   * \param rhs The right hand params.
   * \return Check result.
   */
  virtual BaseCheckResult FuncParamsCheck(const ffi::Array<Type>& lhs,
                                          const ffi::Array<Type>& rhs) {
    auto res = ArrayCheck(lhs, rhs);
    // treat L1 failures in params checking as L2.
    if (res == BaseCheckResult::kFailL1) res = BaseCheckResult::kFailL2;
    return res;
  }
  // helper functions
  /*!
   * \brief Combine check results.
   * \param lhs The left operand.
   * \param rhs The righr operand.
   * \return The check result.
   */
  static BaseCheckResult CombineCheck(BaseCheckResult lhs, BaseCheckResult rhs) {
    if (lhs == BaseCheckResult::kFailL0 || rhs == BaseCheckResult::kFailL0) {
      return BaseCheckResult::kFailL0;
    }
    if (lhs == BaseCheckResult::kFailL1 || rhs == BaseCheckResult::kFailL1) {
      return BaseCheckResult::kFailL1;
    }
    if (lhs == BaseCheckResult::kFailL2 || rhs == BaseCheckResult::kFailL2) {
      return BaseCheckResult::kFailL2;
    }
    return BaseCheckResult::kPass;
  }

  /*!
   * \brief Generic helper function to check arrays.
   * \param lhs The left operand.
   * \param rhs The right operand.
   */
  BaseCheckResult ArrayCheck(const ffi::Array<Type>& lhs, const ffi::Array<Type>& rhs) {
    if (lhs.size() != rhs.size()) return BaseCheckResult::kFailL0;
    BaseCheckResult ret = BaseCheckResult::kPass;

    for (size_t i = 0; i < lhs.size(); ++i) {
      auto cmp_ret = this->VisitType(lhs[i], rhs[i]);
      if (ret == BaseCheckResult::kFailL0) return ret;
      ret = CombineCheck(cmp_ret, ret);
    }
    return ret;
  }
};

BaseCheckResult TypeBaseCheck(const Type& base, const Type& derived) {
  arith::Analyzer analyzer;
  return TypeBaseCheck(base, derived, analyzer);
}

BaseCheckResult TypeBaseCheck(const Type& base, const Type& derived, const arith::Analyzer& ana) {
  return TypeBaseChecker(ana.get())(base, derived);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.TypeBaseCheck",
                        [](const Type& base, const Type& derived) -> int {
                          return static_cast<int>(TypeBaseCheck(base, derived));
                        });
}

bool IsBaseOf(const Type& base, const Type& derived) {
  arith::Analyzer analyzer;
  return IsBaseOf(base, derived, analyzer);
}

bool IsBaseOf(const Type& base, const Type& derived, const arith::Analyzer& ana) {
  return TypeBaseCheck(base, derived, ana) == BaseCheckResult::kPass;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.TypeIsBaseOf", [](const Type& base, const Type& derived) {
    return IsBaseOf(base, derived);
  });
}

class TypeBasePreconditionCollector : public TypeFunctor<PrimExpr(const Type&, const Type&)> {
 public:
  explicit TypeBasePreconditionCollector() {}

  PrimExpr VisitType(const Type& lhs, const Type& other) override {
    if (lhs.same_as(other)) {
      // Early bail-out if the Type has reference equality.
      return IntImm::Bool(true);
    } else {
      return TypeFunctor::VisitType(lhs, other);
    }
  }

  PrimExpr VisitType_(const AnyTypeNode* lhs, const Type& other) final {
    return IntImm::Bool(true);
  }

  PrimExpr VisitType_(const PrimTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<PrimTypeNode>();
    if (rhs == nullptr) {
      return IntImm::Bool(false);
    }

    if (lhs->dtype != rhs->dtype) {
      return IntImm::Bool(false);
    }

    return IntImm::Bool(true);
  }

  PrimExpr VisitType_(const ShapeTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<ShapeTypeNode>();
    if (rhs == nullptr) {
      return IntImm::Bool(false);
    }
    // lhs have unknown ndim
    if (lhs->IsUnknownNdim()) {
      return IntImm::Bool(true);
    }

    // ndim must match
    if (lhs->ndim != rhs->ndim) {
      return IntImm::Bool(false);
    }

    if (lhs->values.has_value() && rhs->values.has_value()) {
      return ArrayCheck(lhs->values.value(), rhs->values.value());
    } else if (lhs->values.has_value() && !rhs->values.has_value()) {
      return IntImm::Bool(false);
    } else {
      return IntImm::Bool(true);
    }
  }

  PrimExpr VisitType_(const TensorTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<TensorTypeNode>();
    if (rhs == nullptr) {
      return IntImm::Bool(false);
    }
    // dtype mismatch
    if (!lhs->IsUnknownDtype() && !rhs->IsUnknownDtype() &&
        lhs->dtype.value() != rhs->dtype.value()) {
      return IntImm::Bool(false);
    }
    if (!lhs->IsUnknownDtype() && rhs->IsUnknownDtype()) {
      return IntImm::Bool(false);
    }

    // ndim mismatch
    if (!lhs->IsUnknownNdim() && lhs->ndim != rhs->ndim) {
      return IntImm::Bool(false);
    }

    // vdevice mismatch
    if (lhs->vdevice.has_value() && !rhs->vdevice.has_value()) {
      return IntImm::Bool(false);
    }
    if (lhs->vdevice.has_value() && rhs->vdevice.has_value()) {
      VDevice lhs_vdevice = lhs->vdevice.value();
      VDevice rhs_vdevice = rhs->vdevice.value();
      if (lhs_vdevice->target.defined() && !rhs_vdevice->target.defined()) {
        return IntImm::Bool(false);
      }
      // mismatch in either the target, vdevice_id, or memory_scope
      if ((lhs_vdevice->target.defined() && rhs_vdevice->target.defined()) &&
          (lhs_vdevice->target != rhs_vdevice->target ||
           lhs_vdevice->vdevice_id != rhs_vdevice->vdevice_id ||
           lhs_vdevice->memory_scope != rhs_vdevice->memory_scope)) {
        return IntImm::Bool(false);
      }
    }

    if (lhs->shape.same_as(rhs->shape)) {
      return IntImm::Bool(true);
    } else if (lhs->shape.has_value() && !rhs->shape.has_value()) {
      return IntImm::Bool(false);
    }

    auto* lhs_shape = lhs->shape.as<ShapeExprNode>();
    auto* rhs_shape = rhs->shape.as<ShapeExprNode>();
    if (lhs_shape && rhs_shape) {
      return ArrayCheck(lhs_shape->values, rhs_shape->values);
    } else if (lhs_shape && !rhs_shape) {
      return IntImm::Bool(false);
    }

    return IntImm::Bool(true);
  }

  PrimExpr VisitType_(const distributed::DTensorTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<distributed::DTensorTypeNode>();
    if (rhs == nullptr) {
      return IntImm::Bool(false);
    }

    ffi::StructuralEqual struct_equal;
    if (!struct_equal(lhs->device_mesh, rhs->device_mesh) ||
        !struct_equal(lhs->placement, rhs->placement)) {
      return IntImm::Bool(false);
    }

    return this->VisitType(lhs->tensor_ty, rhs->tensor_ty);
  }

  PrimExpr VisitType_(const TupleTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<TupleTypeNode>();
    if (rhs == nullptr) {
      return IntImm::Bool(false);
    }
    return ArrayCheck(lhs->fields, rhs->fields);
  }

  PrimExpr VisitType_(const FuncTypeNode* lhs, const Type& other) override {
    auto* rhs = other.as<FuncTypeNode>();
    if (rhs == nullptr) {
      return IntImm::Bool(false);
    }

    // Check purity: Pure functions are a subtype of impure functions
    if (lhs->purity && !rhs->purity) {
      return IntImm::Bool(false);
    }

    if (lhs->derive_func.has_value() && !lhs->derive_func.same_as(rhs->derive_func)) {
      return IntImm::Bool(false);
    }
    if (lhs->params.has_value() && !rhs->params.has_value()) {
      return IntImm::Bool(false);
    }

    PrimExpr all_match = VisitType(lhs->ret, rhs->ret);

    PrimExpr param_check;
    if (lhs->params.has_value()) {
      param_check = ArrayCheck(lhs->params.value(), rhs->params.value());
    } else {
      param_check = IntImm::Bool(true);
    }

    PrimExpr ret_check = VisitType(lhs->ret, rhs->ret);

    return param_check && ret_check;
  }

 private:
  PrimExpr ArrayCheck(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) {
      return IntImm::Bool(false);
    }

    PrimExpr all_equal = IntImm::Bool(true);
    for (size_t i = 0; i < lhs.size(); i++) {
      all_equal = all_equal && (lhs[i] == rhs[i]);
    }
    return all_equal;
  }

  PrimExpr ArrayCheck(const ffi::Array<Type>& lhs, const ffi::Array<Type>& rhs) {
    if (lhs.size() != rhs.size()) {
      return IntImm::Bool(false);
    }

    PrimExpr all_pass = IntImm::Bool(true);

    for (size_t i = 0; i < lhs.size(); ++i) {
      all_pass = all_pass && VisitType(lhs[i], rhs[i]);
    }
    return all_pass;
  }
};

PrimExpr TypeBaseCheckPrecondition(const Type& base, const Type& derived) {
  TypeBasePreconditionCollector visitor;
  return visitor(base, derived);
}

//--------------------------
// DeriveType
//--------------------------

// NOTE: we are reusing TypeBaseChecker here to populate a mapping
// from the expressions in arg(rhs) to var in param.
class CallRetTypeDeriver : public TypeBaseChecker {
 public:
  explicit CallRetTypeDeriver(arith::AnalyzerObj* ana) : TypeBaseChecker(ana) {}

  // No short cut, so we can recursively populate all pairs.
  BaseCheckResult VisitType(const Type& lhs, const Type& other) final {
    return TypeFunctor::VisitType(lhs, other);
  }

  Type Derive(const FuncType& finfo, const Call& call, const BlockBuilder& ctx) {
    // opaque derivation
    if (finfo->IsOpaque()) {
      if (finfo->derive_func.has_value()) {
        // derive using custom derivation function.
        return finfo->derive_func.value()(call, ctx);
      } else {
        // directly return the normal value.
        return finfo->ret;
      }
    }

    // Normal function signature derivation.
    auto params = finfo->params.value();
    if (params.size() != call->args.size()) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Number of arguments and parameters mismatch:"
          << " Function " << call->op << " has type " << finfo << " and accepts " << params.size()
          << " parameters, but was called with " << call->args.size() << " arguments ("
          << call->args << ")";
    }
    // Visit each param arg pair, check and populate the var map
    for (size_t i = 0; i < params.size(); ++i) {
      TVM_FFI_VISIT_BEGIN();
      auto arg_ty = GetType(call->args[i]);
      BaseCheckResult res = this->VisitType(params[i], arg_ty);
      // Report error if we find L1 level failure
      // L2 level is best effort so we don't report.
      // The behavior of L2 can be customized later.
      if (res == BaseCheckResult::kFailL0 || res == BaseCheckResult::kFailL1) {
        TVM_FFI_VISIT_THROW(ValueError, call->args[i])
            << "Argument " << i << " type mismatch:"
            << " expected " << params[i] << ", given " << arg_ty;
      }
      TVM_FFI_VISIT_END(call->args[i]);
    }
    // map the ret using the populated var map.
    return EraseToWellDefined(finfo->ret, var_map_);
  }

 protected:
  // Whether to populate map in params.
  bool populate_mapping_{true};
  // for simplicity, we make these fields public so the user can access them.
  ffi::Map<Var, Expr> var_map_;

  using TypeBaseChecker::ShapeMatchCheck;

  // Match shape values in between param(lhs) and arg(rhs)
  BaseCheckResult PrimExprMatchCheck(const PrimExpr& param, const PrimExpr& arg) final {
    if (!populate_mapping_) {
      return TypeBaseChecker::PrimExprMatchCheck(param, arg);
    }

    if (auto var = param.as<tirx::PrimVar>()) {
      auto it = var_map_.find(var.value());
      // not populated
      if (it == var_map_.end()) {
        var_map_.Set(var.value(), arg);
        return BaseCheckResult::kPass;
      } else {
        // Best effort prove.
        PrimExpr mapped_value = (*it).second.as_or_throw<PrimExpr>();
        if (analyzer_->CanProveEqual(mapped_value, arg)) return BaseCheckResult::kPass;
        return BaseCheckResult::kFailL2;
      }
    } else {
      // Best effort
      // Do not attempt to do prove when param contains a symbolic expr.
      // such expression might depends on a later defined var in params created by dyn fusion.
      // example: f(a: Tensor[(n+1)], s: Shape[(n,)]), the (n+1) case here.
      return TypeBaseChecker::PrimExprMatchCheck(param, arg);
    }
  }

  BaseCheckResult ShapeMatchCheck(const Expr& lhs, const Expr& rhs) final {
    if (!populate_mapping_) {
      return TypeBaseChecker::ShapeMatchCheck(lhs, rhs);
    }

    if (auto* ptr = lhs.as<VarNode>();
        ptr && !lhs.as<DataflowVarNode>() && !lhs.as<tirx::PrimVar>()) {
      auto var = ffi::GetRef<Var>(ptr);
      auto it = var_map_.find(var);
      // not populated
      if (it == var_map_.end()) {
        var_map_.Set(var, rhs);
        return BaseCheckResult::kPass;
      } else {
        // Best effort prove.
        Expr mapped_value = (*it).second;
        if (CanProveShapeEqual(mapped_value, rhs, ffi::GetRef<arith::Analyzer>(analyzer_))) {
          return BaseCheckResult::kPass;
        }
        return BaseCheckResult::kFailL2;
      }
    }
    auto lhs_shape = lhs.as<ShapeExprNode>();
    auto rhs_shape = rhs.as<ShapeExprNode>();
    TVM_FFI_ICHECK(lhs_shape) << "lhs must have a shape";
    if (!rhs_shape) return BaseCheckResult::kFailL2;
    return ShapeMatchCheck(lhs_shape->values, rhs_shape->values);
  }

  BaseCheckResult FuncParamsCheck(const ffi::Array<Type>& lhs, const ffi::Array<Type>& rhs) final {
    // Set populate mapping to false
    // so we do not pick up symbolic vars in params with function type.
    //
    // @R.function
    // def f(g: R.Func([R.Tensor[(n,)]], R.Tensor[(n+1,)]),
    //       x: R.Tensor[(m,)]) -> R.Tensor[(m,)]:
    //     ...
    //
    // For example, in the above function f, we should avoid
    // pick up n in g's signature.
    bool populate_mapping = false;
    std::swap(populate_mapping_, populate_mapping);
    auto ret = TypeBaseChecker::FuncParamsCheck(lhs, rhs);
    std::swap(populate_mapping_, populate_mapping);
    return ret;
  }
};

Type DeriveCallRetType(const FuncType& finfo, const Call& call, const BlockBuilder& ctx) {
  arith::Analyzer analyzer;
  return DeriveCallRetType(finfo, call, ctx, analyzer);
}

Type DeriveCallRetType(const FuncType& finfo, const Call& call, const BlockBuilder& ctx,
                       const arith::Analyzer& ana) {
  // The deriver's TVM_FFI_VISIT_THROW seeds a VisitErrorContext on the error;
  // the outer pass wrapper catches it and enriches the message with the access
  // path. Nothing to do here but propagate.
  return CallRetTypeDeriver(ana.get()).Derive(finfo, call, ctx);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.DeriveCallRetType",
                        [](const FuncType& finfo, const Call& call, const BlockBuilder& ctx) {
                          return DeriveCallRetType(finfo, call, ctx);
                        });
}

//--------------------------
// UnifyToLCA
//--------------------------
class TypeLCAFinder : public TypeFunctor<Type(const Type&, const Type&)> {
 public:
  explicit TypeLCAFinder(arith::AnalyzerObj* ana) : analyzer_(ana) {}

  Type VisitType(const Type& lhs, const Type& other) final {
    // quick path
    if (lhs.same_as(other)) return lhs;
    return TypeFunctor::VisitType(lhs, other);
  }

  // AnyType is base of every Relax type, unify to Any.
  Type VisitType_(const AnyTypeNode* lhs, const Type& other) final {
    return ffi::GetRef<Type>(lhs);
  }

  Type VisitType_(const PrimTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<PrimTypeNode>();
    if (rhs == nullptr) return AnyType(lhs->span);
    if (lhs->dtype != rhs->dtype) {
      // PrimType will be treated as their boxed Any values
      // as a result we can unify to Any.
      return AnyType(lhs->span);
    }
    return ffi::GetRef<Type>(lhs);
  }

  Type VisitType_(const ShapeTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<ShapeTypeNode>();
    if (rhs == nullptr) return AnyType(lhs->span);

    int ndim = lhs->ndim == rhs->ndim ? lhs->ndim : kUnknownNDim;
    if (lhs->ndim != rhs->ndim || !lhs->values.has_value() || !rhs->values.has_value() ||
        !CanProveShapeEqual(lhs->values.value(), rhs->values.value(),
                            ffi::GetRef<arith::Analyzer>(analyzer_))) {
      // prefers return same when possible
      if (!lhs->values.has_value() && lhs->ndim == ndim) {
        return ffi::GetRef<Type>(lhs);
      } else {
        return ShapeType(ndim, lhs->span);
      }
    }
    // equals to each other
    return ffi::GetRef<Type>(lhs);
  }

  Type VisitType_(const TensorTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<TensorTypeNode>();
    if (rhs == nullptr) return AnyType(lhs->span);

    // find the target dtype, ndim, and vdevice.
    ffi::Optional<PrimType> dtype = (!lhs->IsUnknownDtype() && !rhs->IsUnknownDtype() &&
                                     lhs->dtype.value() == rhs->dtype.value())
                                        ? ffi::Optional<PrimType>(lhs->dtype.value())
                                        : std::nullopt;
    int ndim = lhs->ndim == rhs->ndim ? lhs->ndim : kUnknownNDim;
    VDevice vdev = VDevice();
    if (lhs->vdevice.has_value() && rhs->vdevice.has_value() &&
        lhs->vdevice.value() == rhs->vdevice.value()) {
      vdev = lhs->vdevice.value();
    }
    // if ndim mismatch or one side of shape is missing
    // then we cannot keep in symbolic shape
    if (lhs->ndim != rhs->ndim || !lhs->shape.has_value() || !rhs->shape.has_value() ||
        !CanProveShapeEqual(lhs->shape.value(), rhs->shape.value(),
                            ffi::GetRef<arith::Analyzer>(analyzer_))) {
      // reuse lhs when possible
      if (!lhs->shape.has_value() && lhs->dtype == dtype && lhs->ndim == ndim &&
          (!lhs->vdevice.has_value() || vdev.defined())) {
        return ffi::GetRef<Type>(lhs);
      } else {
        return TensorType(dtype, ndim, vdev, lhs->span);
      }
    }
    // symbolic shape and vdevice match but dtype mismatch
    if (lhs->dtype != dtype || (lhs->vdevice.has_value() && !vdev.defined())) {
      return TensorType(lhs->shape.value(), dtype, vdev, lhs->span);
    } else {
      return ffi::GetRef<Type>(lhs);
    }
  }

  Type VisitType_(const TupleTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<TupleTypeNode>();
    if (rhs == nullptr) return AnyType(lhs->span);
    ffi::Optional<ffi::Array<Type>> fields = UnifyArray(lhs->fields, rhs->fields);
    // tuple length not the same.
    if (!fields.has_value()) return AnyType(lhs->span);

    // same length tuple.
    if (!fields.same_as(lhs->fields)) {
      return TupleType(fields.value(), lhs->span);
    } else {
      return ffi::GetRef<Type>(lhs);
    }
  }

  Type VisitType_(const FuncTypeNode* lhs, const Type& other) final {
    auto* rhs = other.as<FuncTypeNode>();
    if (rhs == nullptr) return AnyType(lhs->span);

    // the unified function is pure only if both are pure
    bool purity = lhs->purity && rhs->purity;

    // lhs opaque handling
    if (lhs->IsOpaque()) {
      if (lhs->derive_func.has_value()) {
        if (lhs->derive_func.same_as(rhs->derive_func)) {
          return ffi::GetRef<Type>(lhs);
        } else {
          // Create a new opaque with object return
          return FuncType::OpaqueFunc(AnyType(), purity, lhs->span);
        }
      } else {
        // no derivation function, only depends on ret
        Type ret = this->VisitType(lhs->ret, rhs->ret);
        if (ret.same_as(lhs->ret)) return ffi::GetRef<Type>(lhs);
        return FuncType::OpaqueFunc(ret, purity, lhs->span);
      }
    }
    // rhs is opaque, lhs is not
    if (rhs->IsOpaque()) {
      // unify ret value, note that rhs's ret is context free(because it is opaque)
      // so result of the unify is also context-free.
      Type ret = this->VisitType(lhs->ret, rhs->ret);
      return FuncType::OpaqueFunc(ret, purity, lhs->span);
    }

    // Both lhs and rhs are not opaque
    // NOTE: lhs->params, rhs->params may contain different symbolic
    // vars that needs to be re-mapped to each other.
    // This can only be done through structural equality check.
    //
    // So we check structural equality here and if two are structurally
    // equal return true.
    //
    // otherwise we do best effort of unify types without considering var remap.
    //
    // This still does not handle cases where some arguments are sub of another
    // while other parameters needs to get remapped.
    //
    // Given we only do best effort checking in these cases, and such cases
    // are likely not a primary concern atm, we take this approach here.
    if (struct_equal_(ffi::GetRef<Type>(lhs), ffi::GetRef<Type>(rhs))) {
      return ffi::GetRef<Type>(lhs);
    }

    auto params = UnifyArray(lhs->params.value(), rhs->params.value());
    auto ret = this->VisitType(lhs->ret, rhs->ret);

    if (params.same_as(lhs->params) && ret.same_as(lhs->ret)) {
      return ffi::GetRef<Type>(lhs);
    } else {
      // fail to unify the params
      if (!params.has_value()) {
        return FuncType::OpaqueFunc(ret, purity, lhs->span);
      } else {
        return FuncType(params.value(), ret, purity, lhs->span);
      }
    }
  }

 private:
  // analyzer
  arith::AnalyzerObj* analyzer_;
  // struct equal checker
  ffi::StructuralEqual struct_equal_;

  // check arrays
  ffi::Optional<ffi::Array<Type>> UnifyArray(const ffi::Array<Type>& lhs,
                                             const ffi::Array<Type>& rhs) {
    if (lhs.same_as(rhs)) return lhs;
    if (lhs.size() != rhs.size()) return std::nullopt;
    size_t index = 0;
    return lhs.Map([&](const Type& a) { return this->VisitType(a, rhs[index++]); });
  }
};

Type TypeLCA(const Type& lhs, const Type& rhs) {
  arith::Analyzer analyzer;
  return TypeLCA(lhs, rhs, analyzer);
}

Type TypeLCA(const Type& lhs, const Type& rhs, const arith::Analyzer& ana) {
  return TypeLCAFinder(ana.get())(lhs, rhs);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.TypeLCA",
                        [](const Type& lhs, const Type& rhs) { return TypeLCA(lhs, rhs); });
}

//--------------------------
// TIRVarsInType
//--------------------------

class TIRVarsDetector : public TypeVisitor {
 public:
  enum class VarType {
    Definition,
    Usage,
  };
  explicit TIRVarsDetector(VarType collection_type) : collection_type(collection_type) {}

  ffi::Array<tirx::Var> GetTIRVars() const { return tir_vars_; }

 private:
  void VisitTypePrimExprField(PrimExpr expr) {
    if (collection_type == VarType::Definition) {
      if (auto opt = expr.as<tirx::PrimVar>()) {
        RecordTIRVar(opt.value());
      }
    } else if (collection_type == VarType::Usage) {
      for (const tirx::Var& tir_var : tirx::UndefinedVars(expr)) {
        if (auto prim_var = tir_var.as<tirx::PrimVar>()) {
          RecordTIRVar(prim_var.value());
        }
      }
    } else {
      TVM_FFI_THROW(InternalError)
          << "Invalid value for VarType enum, " << static_cast<int>(collection_type);
    }
  }

  void VisitShape(ffi::Array<PrimExpr> shape) {
    for (const PrimExpr& expr : shape) {
      VisitTypePrimExprField(expr);
    }
  }

  void VisitType_(const PrimTypeNode* prim_ty) final {}

  void VisitType_(const ShapeTypeNode* shape_ty) final {
    if (shape_ty->values.has_value()) {
      VisitShape(shape_ty->values.value());
    }
  }

  void VisitType_(const TensorTypeNode* tensor_ty) final {
    if (tensor_ty->shape.has_value()) {
      VisitType(GetType(tensor_ty->shape.value()));
    }
  }

  void RecordTIRVar(const tirx::Var& tir_var) {
    auto insert_res = used_tir_vars_dedup_.insert(tir_var.get());
    if (insert_res.second) {
      tir_vars_.push_back(tir_var);
    }
  }

  ffi::Array<tirx::Var> tir_vars_;
  std::unordered_set<const tirx::VarNode*> used_tir_vars_dedup_;

  VarType collection_type;
};

ffi::Array<tirx::Var> TIRVarsInType(const Type& ty) {
  TIRVarsDetector detector(TIRVarsDetector::VarType::Usage);
  detector(ty);
  return detector.GetTIRVars();
}

ffi::Array<tirx::Var> DefinableTIRVarsInType(const Type& ty) {
  TIRVarsDetector detector(TIRVarsDetector::VarType::Definition);
  detector(ty);
  return detector.GetTIRVars();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.analysis.TIRVarsInType", TIRVarsInType)
      .def("relax.analysis.DefinableTIRVarsInType", DefinableTIRVarsInType);
}

class NonNegativeExpressionCollector : relax::TypeVisitor {
 public:
  static ffi::Array<PrimExpr> Collect(const Type& ty) {
    NonNegativeExpressionCollector visitor;
    visitor(ty);
    return visitor.expressions_;
  }

 private:
  void VisitType_(const TensorTypeNode* op) override {
    if (op->shape.has_value()) {
      VisitType(GetType(op->shape.value()));
    }
  }

  void VisitType_(const PrimTypeNode* op) override {
    // Unlike the expressions in TensorType or ShapeType,
    // PrimType may contain negative values.  This override
    // prevents calling VisitTypeExprField from the default
    // TypeVisitor implementation.
  }

  void VisitTypeExprField(const PrimExpr& size_expr) override {
    if (auto size_int = size_expr.as<IntImmNode>(); size_int && size_int->value >= 0) {
      // Avoid cluttering the result with non-negative integers
      return;
    }

    if (!dedup_lookup_.count(size_expr)) {
      expressions_.push_back(size_expr);
      dedup_lookup_.insert(size_expr);
    }
  }

  ffi::Array<PrimExpr> expressions_;
  std::unordered_set<PrimExpr, ffi::StructuralHash, ffi::StructuralEqual> dedup_lookup_;
};

ffi::Array<PrimExpr> CollectNonNegativeExpressions(const Type& ty) {
  return NonNegativeExpressionCollector::Collect(ty);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.CollectNonNegativeExpressions",
                        CollectNonNegativeExpressions);
}

class SymbolicVarCollector : public relax::ExprVisitor, public relax::TypeVisitor {
 public:
  static ffi::Array<tirx::Var> Free(const Expr& expr) {
    SymbolicVarCollector collector;
    collector.relax::ExprVisitor::VisitExpr(expr);
    ffi::Array<tirx::Var> ret{collector.free_symbolic_var_.begin(),
                              collector.free_symbolic_var_.end()};
    return ret;
  }

  static ffi::Array<tirx::Var> Defined(const Expr& expr) {
    SymbolicVarCollector collector;
    collector.relax::ExprVisitor::VisitExpr(expr);
    ffi::Array<tirx::Var> ret{collector.defined_symbolic_var_.begin(),
                              collector.defined_symbolic_var_.end()};
    return ret;
  }

 private:
  using relax::ExprVisitor::VisitExpr;
  using relax::ExprVisitor::VisitExpr_;

  // Possible mode of visitor, used as bit-flags
  enum VisitMode {
    /*! \brief Do nothing on encountering a symbolic variable */
    kNone = 0,

    /*! \brief Provide a variable definition on first occurrence.
     *
     * If a symbolic variable occurs at a site where a definition can
     * be provided, mark the variable as having a definition.
     */
    kProvideDefinition = 1,

    /*! \brief Require a variable definition on occurrence.
     *
     * If a symbolic variable occurs, and has not previously been
     * defined, mark the variable as being free/undefined.
     */
    kRequireDefinition = 2,
  };

  void VisitExpr_(const FunctionNode* op) final {
    WithMode(VisitMode::kProvideDefinition, [&]() {
      for (Var param : op->params) {
        relax::TypeVisitor::VisitType(GetType(param));
      }
    });

    WithMode(VisitMode::kRequireDefinition, [&]() {
      for (Var param : op->params) {
        relax::TypeVisitor::VisitType(GetType(param));
      }
    });

    relax::ExprVisitor::VisitExpr_(op);
  }

  void VisitBinding_(const MatchCastNode* binding) final {
    WithMode(VisitMode(VisitMode::kProvideDefinition | VisitMode::kRequireDefinition),
             [&]() { this->VisitType(binding->ty); });

    relax::ExprVisitor::VisitBinding_(binding);
  }

  void VisitExprDepTypeField(const Type& ty) { return this->VisitType(ty); }

  void VisitType_(const FuncTypeNode* op) final {
    if (op->params.has_value()) {
      // Visit the parameters once to collect bindings, and another
      // time to collect usages.  Otherwise, a symbolic variable
      // defined by a later parameter may be treated as undefined when
      // used by an earlier parameter.
      WithMode(VisitMode::kProvideDefinition, [&]() {
        for (Type param : op->params.value()) {
          this->VisitType(param);
        }
      });

      WithMode(VisitMode::kRequireDefinition, [&]() {
        for (Type param : op->params.value()) {
          this->VisitType(param);
        }
      });
    }
    this->VisitType(op->ret);
  }

  void VisitTypeExprField(const Expr& expr) final {
    if (auto* shape = expr.as<relax::ShapeExprNode>()) {
      for (const auto& val : shape->values) {
        this->VisitTypeExprField(val);
      }
      return;
    } else if (auto prim_value = expr.as<PrimExpr>()) {
      this->VisitTypeExprField(prim_value.value());
      return;
    }
    relax::ExprVisitor::VisitExpr(expr);
  }

  void VisitTypeExprField(const PrimExpr& expr) final {
    if (mode_ & VisitMode::kProvideDefinition) {
      if (auto var = expr.as<tirx::PrimVar>()) {
        defined_symbolic_var_.insert(var.value());
      }
    }
    if (mode_ & VisitMode::kRequireDefinition) {
      relax::ExprVisitor::VisitExpr(expr);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    if (!op->ty.as<PrimTypeNode>()) {
      return;
    }
    tirx::PrimVar var = ffi::GetRef<Var>(op).as_or_throw<tirx::PrimVar>();
    // default mode, check defined.
    if (defined_symbolic_var_.count(var) == 0) {
      free_symbolic_var_.insert(var);
    }
  }

  void VisitExpr_(const DataflowVarNode*) final {}

  void VisitVarDef_(const VarNode* op) final {
    if (op->ty.as<PrimTypeNode>()) {
      defined_symbolic_var_.insert(ffi::GetRef<Var>(op).as_or_throw<tirx::PrimVar>());
    }
    relax::ExprVisitor::VisitVarDef_(op);
  }

  void VisitVarDef_(const DataflowVarNode* op) final {
    relax::ExprVisitor::VisitVarDef_(static_cast<const VarNode*>(op));
  }

  // Run callback with mode.
  template <typename FType>
  void WithMode(VisitMode mode, FType callback) {
    std::swap(mode_, mode);
    callback();
    std::swap(mode_, mode);
  }

  /*! \brief The current visit mode. */
  VisitMode mode_ = VisitMode::kRequireDefinition;
  /*! \brief The set of defined symbolic vars. */
  std::unordered_set<tirx::Var> defined_symbolic_var_;
  /*! \brief The set of free/undefined symbolic vars. */
  std::unordered_set<tirx::Var> free_symbolic_var_;
};

ffi::Array<tirx::Var> DefinedSymbolicVars(const Expr& expr) {
  return SymbolicVarCollector::Defined(expr);
}
ffi::Array<tirx::Var> FreeSymbolicVars(const Expr& expr) {
  return SymbolicVarCollector::Free(expr);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.analysis.DefinedSymbolicVars", DefinedSymbolicVars)
      .def("relax.analysis.FreeSymbolicVars", FreeSymbolicVars);
}

}  // namespace relax
}  // namespace tvm
