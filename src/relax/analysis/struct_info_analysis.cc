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
 * \file struct_info_analysis.cc
 * \brief Implementations of foundation struct info analysis
 *
 * \note Update this file when you added a new StructInfo.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
namespace relax {

//--------------------------
// GetStaticType
//--------------------------
class StaticTypeDeriver : public StructInfoFunctor<Type(const StructInfo&)> {
 public:
  Type VisitStructInfo_(const ObjectStructInfoNode* op) final { return ObjectType(op->span); }

  Type VisitStructInfo_(const PrimStructInfoNode* op) final {
    return PrimType(op->dtype, op->span);
  }

  Type VisitStructInfo_(const ShapeStructInfoNode* op) final {
    return ShapeType(op->ndim, op->span);
  }

  Type VisitStructInfo_(const TensorStructInfoNode* op) final {
    return DynTensorType(op->ndim, op->dtype);
  }

  // module: distributed
  Type VisitStructInfo_(const distributed::DTensorStructInfoNode* op) final { return ObjectType(); }
  // end-module: distributed

  Type VisitStructInfo_(const TupleStructInfoNode* op) final {
    Array<Type> fields =
        op->fields.Map([this](const StructInfo& sinfo) { return this->VisitStructInfo(sinfo); });
    return TupleType(fields, op->span);
  }

  Type VisitStructInfo_(const FuncStructInfoNode* op) final {
    if (op->IsOpaque()) return PackedFuncType(op->span);
    Array<Type> params = op->params.value().Map(
        [this](const StructInfo& sinfo) { return this->VisitStructInfo(sinfo); });
    Type ret = this->VisitStructInfo(op->ret);
    return FuncType(params, ret, {}, {}, op->span);
  }
};

Type GetStaticType(const StructInfo& info) { return StaticTypeDeriver()(info); }

TVM_REGISTER_GLOBAL("relax.analysis.GetStaticType").set_body_typed([](const StructInfo& info) {
  return GetStaticType(info);
});

//--------------------------
// StructInfoFromType
//--------------------------

StructInfo StructInfoFromType(const Type& type) {
  if (type.as<ObjectTypeNode>()) {
    return ObjectStructInfo(type->span);
  } else if (const PrimTypeNode* prim_type = type.as<PrimTypeNode>()) {
    return PrimStructInfo(prim_type->dtype, prim_type->span);
  } else if (const ShapeTypeNode* shape_type = type.as<ShapeTypeNode>()) {
    return ShapeStructInfo(shape_type->ndim, type->span);
  } else if (const DynTensorTypeNode* tensor_type = type.as<DynTensorTypeNode>()) {
    return TensorStructInfo(tensor_type->dtype, tensor_type->ndim);
  } else if (const TupleTypeNode* tuple_type = type.as<TupleTypeNode>()) {
    Array<StructInfo> fields;
    for (const Type& field : tuple_type->fields) {
      fields.push_back(StructInfoFromType(field));
    }
    return TupleStructInfo(fields, type->span);
  } else if (const FuncTypeNode* func_type = type.as<FuncTypeNode>()) {
    Array<StructInfo> params =
        func_type->arg_types.Map([](const Type& param) { return StructInfoFromType(param); });
    StructInfo ret = StructInfoFromType(func_type->ret_type);
    // TODO(relax-team): Maybe add purity into the type as well
    return FuncStructInfo(params, ret, true, func_type->span);
  } else {
    LOG(FATAL) << "Unsupported type: " << type;
    return StructInfo();
  }
}

//--------------------------
// EraseToWellDefined
//--------------------------
class WellDefinedEraser : public StructInfoMutator,
                          public ExprMutatorBase,
                          public tir::ExprMutator {
 public:
  WellDefinedEraser(std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map,
                    std::function<Optional<Expr>(const Var& var)> f_var_map, arith::Analyzer* ana)
      : f_shape_var_map_(f_shape_var_map), f_var_map_(f_var_map), ana_(ana) {}

  StructInfo VisitStructInfo_(const PrimStructInfoNode* op) final {
    bool has_undefined = false;
    Optional<PrimExpr> value;

    if (op->value.defined()) {
      std::swap(has_undefined_, has_undefined);
      value = VisitPrimExpr(op->value.value());
      std::swap(has_undefined_, has_undefined);
    }

    // erase symbolic shape if we have undefined.
    if (!has_undefined) {
      if (value.same_as(op->value)) {
        return GetRef<StructInfo>(op);
      } else {
        return PrimStructInfo(value.value(), op->span);
      }
    } else {
      return PrimStructInfo(op->dtype, op->span);
    }
  }

  StructInfo VisitStructInfo_(const ShapeStructInfoNode* op) final {
    bool has_undefined = false;
    Optional<Array<PrimExpr>> values;

    if (op->values.defined()) {
      std::swap(has_undefined_, has_undefined);
      values = op->values.value().Map([&](PrimExpr val) { return this->VisitPrimExpr(val); });
      std::swap(has_undefined_, has_undefined);
    }
    // erase symbolic shape if we have undefined.
    if (!has_undefined) {
      if (values.same_as(op->values)) {
        return GetRef<StructInfo>(op);
      } else {
        return ShapeStructInfo(values.value(), op->span);
      }
    } else {
      return ShapeStructInfo(op->ndim, op->span);
    }
  }

  StructInfo VisitStructInfo_(const TensorStructInfoNode* op) final {
    bool has_undefined = false;
    Optional<Expr> shape;

    if (op->shape.defined()) {
      std::swap(has_undefined_, has_undefined);
      shape = relax::ExprMutatorBase::VisitExpr(op->shape.value());
      std::swap(has_undefined_, has_undefined);
    }

    VDevice vdev = op->vdevice.value_or(VDevice());

    // erase symbolic shape if we have undefined.
    if (!has_undefined) {
      if (shape.same_as(op->shape)) {
        return GetRef<StructInfo>(op);
      } else {
        if (shape.defined()) {
          return TensorStructInfo(shape.value(), op->dtype, vdev, op->span);
        } else {
          return TensorStructInfo(op->dtype, op->ndim, vdev, op->span);
        }
      }
    } else {
      return TensorStructInfo(op->dtype, op->ndim, vdev, op->span);
    }
  }

  StructInfo VisitStructInfo_(const FuncStructInfoNode* op) final {
    // NOTE: we always require func struct info to be well-defined.
    //
    // All the occuring symbolic variables are defined in parameters'
    // struct info annotations. So there is no needed to erase.
    return GetRef<StructInfo>(op);
  }

  using relax::ExprMutatorBase::VisitExpr_;
  using tir::ExprMutator::VisitExpr_;

  // connect things up
  PrimExpr VisitPrimExpr(const PrimExpr& expr) {
    // apply eager simplification
    PrimExpr val = tir::ExprMutator::VisitExpr(expr);
    if (!val.same_as(expr)) {
      return ana_->Simplify(val);
    } else {
      return val;
    }
  }

  Expr VisitExpr_(const VarNode* var) final {
    Optional<Expr> ret;
    if (f_var_map_ != nullptr) {
      ret = f_var_map_(GetRef<Var>(var));
    }
    has_undefined_ = has_undefined_ || !ret.defined();
    if (ret.defined()) {
      ICHECK(ret.as<VarNode>() || ret.as<ShapeExprNode>())
          << "Only allow Expr in StructInfo to be ShapeExpr or Var";
    }
    return ret.value_or(GetRef<Expr>(var));
  }

  PrimExpr VisitExpr_(const tir::VarNode* var) final {
    Optional<PrimExpr> ret;
    if (f_shape_var_map_ != nullptr) {
      ret = f_shape_var_map_(GetRef<tir::Var>(var));
    }
    has_undefined_ = has_undefined_ || !ret.defined();

    if (ret.defined()) {
      PrimExpr value = ret.value();
      if (value->IsInstance<IntImmNode>()) {
        return tvm::cast(DataType::Int(64), value);
      }
      ICHECK(value.dtype() == DataType::Int(64)) << "Can only provide i64 expressions in shape";
      return value;
    } else {
      return GetRef<PrimExpr>(var);
    }
  }

 private:
  bool has_undefined_ = false;
  std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map_;
  std::function<Optional<Expr>(const Var& var)> f_var_map_;
  arith::Analyzer* ana_;
};

StructInfo EraseToWellDefined(
    const StructInfo& info, std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map,
    std::function<Optional<Expr>(const Var& var)> f_var_map, arith::Analyzer* ana) {
  if (ana == nullptr) {
    arith::Analyzer inst;
    return WellDefinedEraser(f_shape_var_map, f_var_map, &inst).VisitStructInfo(info);
  } else {
    return WellDefinedEraser(f_shape_var_map, f_var_map, ana).VisitStructInfo(info);
  }
}

StructInfo EraseToWellDefined(const StructInfo& info, Map<tir::Var, PrimExpr> shape_var_map,
                              Map<Var, Expr> var_map, arith::Analyzer* ana) {
  std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map = nullptr;
  std::function<Optional<Expr>(const Var& var)> f_var_map = nullptr;

  if (!shape_var_map.empty()) {
    f_shape_var_map = [&](const tir::Var& var) -> Optional<PrimExpr> {
      auto it = shape_var_map.find(var);
      if (it != shape_var_map.end()) return (*it).second;
      return NullOpt;
    };
  }

  if (!var_map.empty()) {
    f_var_map = [&](const Var& var) -> Optional<Expr> {
      auto it = var_map.find(var);
      if (it != var_map.end()) return (*it).second;
      return NullOpt;
    };
  }

  return EraseToWellDefined(info, f_shape_var_map, f_var_map, ana);
}

TVM_REGISTER_GLOBAL("relax.analysis.EraseToWellDefined")
    .set_body_typed([](const StructInfo& info, Map<tir::Var, PrimExpr> shape_var_map,
                       Map<Var, Expr> var_map) {
      return EraseToWellDefined(info, shape_var_map, var_map);
    });

//--------------------------
// IsBaseOf
//--------------------------
class StructInfoBaseChecker
    : public StructInfoFunctor<BaseCheckResult(const StructInfo&, const StructInfo&)> {
 public:
  explicit StructInfoBaseChecker(arith::Analyzer* ana) : analyzer_(ana) {}

  BaseCheckResult VisitStructInfo(const StructInfo& lhs, const StructInfo& other) override {
    // quick path
    // Note: subclass may disable this quick path if we need to go over all struct info.
    if (lhs.same_as(other)) return BaseCheckResult::kPass;
    return StructInfoFunctor::VisitStructInfo(lhs, other);
  }

  // Object is base of everything
  BaseCheckResult VisitStructInfo_(const ObjectStructInfoNode* lhs, const StructInfo& other) final {
    return BaseCheckResult::kPass;
  }

  BaseCheckResult VisitStructInfo_(const PrimStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<PrimStructInfoNode>();
    if (rhs == nullptr) {
      if (other.as<ObjectStructInfoNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    if (lhs->dtype != rhs->dtype) {
      return BaseCheckResult::kFailL0;
    }

    if (!lhs->value.defined()) return BaseCheckResult::kPass;
    if (!rhs->value.defined()) return BaseCheckResult::kFailL2;

    return PrimValueMatchCheck(lhs->value.value(), rhs->value.value());
  }

  BaseCheckResult VisitStructInfo_(const ShapeStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<ShapeStructInfoNode>();
    if (rhs == nullptr) {
      if (other.as<ObjectStructInfoNode>()) return BaseCheckResult::kFailL1;
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
    if (!lhs->values.defined()) return BaseCheckResult::kPass;
    // rhs does not have symbolic value but lhs do.
    if (!rhs->values.defined()) return BaseCheckResult::kFailL2;

    // shape match check
    return ShapeMatchCheck(lhs->values.value(), rhs->values.value());
  }

  BaseCheckResult VisitStructInfo_(const TensorStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TensorStructInfoNode>();
    if (rhs == nullptr) {
      if (other.as<ObjectStructInfoNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    // dtype mismatch
    if (!lhs->IsUnknownDtype() && lhs->dtype != rhs->dtype) {
      if (rhs->IsUnknownDtype()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    // ndim mismatch
    if (!lhs->IsUnknownNdim() && lhs->ndim != rhs->ndim) {
      if (rhs->IsUnknownNdim()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    // vdevice mismatch
    if (lhs->vdevice.defined() && !rhs->vdevice.defined()) return BaseCheckResult::kFailL1;
    if (lhs->vdevice.defined() && rhs->vdevice.defined()) {
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
    if (!lhs->shape.defined()) return BaseCheckResult::kPass;
    // rhs does not have symbolic value but lhs don't
    if (!rhs->shape.defined()) return BaseCheckResult::kFailL2;

    // shape match check
    return ShapeMatchCheck(lhs->shape.value(), rhs->shape.value());
  }

  // module: distributed
  BaseCheckResult VisitStructInfo_(const distributed::DTensorStructInfoNode* lhs,
                                   const StructInfo& other) final {
    auto* rhs = other.as<distributed::DTensorStructInfoNode>();
    if (rhs == nullptr) {
      if (other.as<ObjectStructInfoNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    BaseCheckResult tensor_sinfo_check_result =
        this->VisitStructInfo(lhs->tensor_sinfo, rhs->tensor_sinfo);
    BaseCheckResult other_check_result;
    if (!struct_equal_(lhs->device_mesh, rhs->device_mesh) ||
        !struct_equal_(lhs->placement, rhs->placement)) {
      other_check_result = BaseCheckResult::kFailL1;
    } else {
      other_check_result = BaseCheckResult::kPass;
    }
    return CombineCheck(tensor_sinfo_check_result, other_check_result);
  }
  // end-module: distributed

  BaseCheckResult VisitStructInfo_(const TupleStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TupleStructInfoNode>();
    if (rhs == nullptr) {
      if (other.as<ObjectStructInfoNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }
    return ArrayCheck(lhs->fields, rhs->fields);
  }

  BaseCheckResult VisitStructInfo_(const FuncStructInfoNode* lhs,
                                   const StructInfo& other) override {
    auto* rhs = other.as<FuncStructInfoNode>();
    if (rhs == nullptr) {
      if (other.as<ObjectStructInfoNode>()) return BaseCheckResult::kFailL1;
      return BaseCheckResult::kFailL0;
    }

    // Check purity: Pure functions are a subtype of impure functions
    if (lhs->purity && !rhs->purity) {
      return BaseCheckResult::kFailL0;
    }

    // lhs opaque handling
    if (lhs->IsOpaque()) {
      if (lhs->derive_func.defined()) {
        // function proving is best effort.
        return lhs->derive_func.same_as(rhs->derive_func) ? BaseCheckResult::kPass
                                                          : BaseCheckResult::kFailL2;
      }
      // no derivation function, only depends on ret
      return this->VisitStructInfo(lhs->ret, rhs->ret);
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
    if (struct_equal_(GetRef<StructInfo>(lhs), other)) return BaseCheckResult::kPass;

    auto param_check = FuncParamsCheck(lhs->params.value(), rhs->params.value());
    auto ret_check = this->VisitStructInfo(lhs->ret, rhs->ret);
    return CombineCheck(param_check, ret_check);
  }

 protected:
  // analyzer
  arith::Analyzer* analyzer_;
  // struct equal checker
  StructuralEqual struct_equal_;

  // customizable functions.
  /*!
   * \brief Check symbolic shape value equivalence.
   * \param lhs The left hand shape.
   * \param rhs The right hand shape.
   * \return CheckResult.
   */
  virtual BaseCheckResult PrimValueMatchCheck(const PrimExpr& lhs, const PrimExpr& rhs) {
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
  virtual BaseCheckResult ShapeMatchCheck(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) return BaseCheckResult::kFailL0;

    BaseCheckResult ret = BaseCheckResult::kPass;
    for (size_t i = 0; i < lhs.size(); ++i) {
      auto cmp_ret = PrimValueMatchCheck(lhs[i], rhs[i]);
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
  virtual BaseCheckResult FuncParamsCheck(const Array<StructInfo>& lhs,
                                          const Array<StructInfo>& rhs) {
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
  BaseCheckResult ArrayCheck(const Array<StructInfo>& lhs, const Array<StructInfo>& rhs) {
    if (lhs.size() != rhs.size()) return BaseCheckResult::kFailL0;
    BaseCheckResult ret = BaseCheckResult::kPass;

    for (size_t i = 0; i < lhs.size(); ++i) {
      auto cmp_ret = this->VisitStructInfo(lhs[i], rhs[i]);
      if (ret == BaseCheckResult::kFailL0) return ret;
      ret = CombineCheck(cmp_ret, ret);
    }
    return ret;
  }
};

BaseCheckResult StructInfoBaseCheck(const StructInfo& base, const StructInfo& derived,
                                    arith::Analyzer* ana) {
  if (ana == nullptr) {
    arith::Analyzer inst;
    return StructInfoBaseChecker(&inst)(base, derived);
  } else {
    return StructInfoBaseChecker(ana)(base, derived);
  }
}

TVM_REGISTER_GLOBAL("relax.analysis.StructInfoBaseCheck")
    .set_body_typed([](const StructInfo& base, const StructInfo& derived) -> int {
      return static_cast<int>(StructInfoBaseCheck(base, derived));
    });

bool IsBaseOf(const StructInfo& base, const StructInfo& derived, arith::Analyzer* ana) {
  return StructInfoBaseCheck(base, derived, ana) == BaseCheckResult::kPass;
}

TVM_REGISTER_GLOBAL("relax.StructInfoIsBaseOf")
    .set_body_typed([](const StructInfo& base, const StructInfo& derived) {
      return IsBaseOf(base, derived);
    });

//--------------------------
// DeriveStructInfo
//--------------------------

// NOTE: we are reusing StructInfoBaseChecker here to populate a mapping
// from the expressions in arg(rhs) to var in param.
class CallRetStructInfoDeriver : public StructInfoBaseChecker {
 public:
  explicit CallRetStructInfoDeriver(arith::Analyzer* ana) : StructInfoBaseChecker(ana) {}

  // No short cut, so we can recursively populate all pairs.
  BaseCheckResult VisitStructInfo(const StructInfo& lhs, const StructInfo& other) final {
    return StructInfoFunctor::VisitStructInfo(lhs, other);
  }

  StructInfo Derive(const FuncStructInfo& finfo, const Call& call, const BlockBuilder& ctx) {
    // opaque derivation
    if (finfo->IsOpaque()) {
      if (finfo->derive_func.defined()) {
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
      ctx->ReportFatal(Diagnostic::Error(call->span)
                       << "number of arguments and parameters mismatch:"
                       << " expected " << params.size() << ", given " << call->args.size());
    }
    // Visit each param arg pair, check and populate the var map
    for (size_t i = 0; i < params.size(); ++i) {
      auto arg_sinfo = GetStructInfo(call->args[i]);
      BaseCheckResult res = this->VisitStructInfo(params[i], arg_sinfo);
      // Report error if we find L1 level failure
      // L2 level is best effort so we don't report.
      // The behavior of L2 can be customized later.
      if (res == BaseCheckResult::kFailL0 || res == BaseCheckResult::kFailL1) {
        ctx->ReportFatal(Diagnostic::Error(call->span)
                         << "Argument " << i << " type mismatch:"
                         << " expected " << params[i] << ", given " << arg_sinfo);
      }
    }
    // map the ret using the populated var map.
    return EraseToWellDefined(finfo->ret, shape_var_map_, var_map_);
  }

 protected:
  // Whether to populate map in params.
  bool populate_mapping_{true};
  // for simplicity, we make these fields public so the user can access them.
  Map<tir::Var, PrimExpr> shape_var_map_;
  Map<Var, Expr> var_map_;

  using StructInfoBaseChecker::ShapeMatchCheck;

  // Match shape values in between param(lhs) and arg(rhs)
  BaseCheckResult PrimValueMatchCheck(const PrimExpr& param, const PrimExpr& arg) final {
    if (!populate_mapping_) {
      return StructInfoBaseChecker::PrimValueMatchCheck(param, arg);
    }

    if (auto* ptr = param.as<tir::VarNode>()) {
      auto var = GetRef<tir::Var>(ptr);
      auto it = shape_var_map_.find(var);
      // not populated
      if (it == shape_var_map_.end()) {
        shape_var_map_.Set(var, arg);
        return BaseCheckResult::kPass;
      } else {
        // Best effort prove.
        PrimExpr mapped_value = (*it).second;
        if (analyzer_->CanProveEqual(mapped_value, arg)) return BaseCheckResult::kPass;
        return BaseCheckResult::kFailL2;
      }
    } else {
      // Best effort
      // Do not attempt to do prove when param contains a symbolic expr.
      // such expression might depends on a later defined var in params created by dyn fusion.
      // example: f(a: Tensor[(n+1)], s: Shape[(n,)]), the (n+1) case here.
      return StructInfoBaseChecker::PrimValueMatchCheck(param, arg);
    }
  }

  BaseCheckResult ShapeMatchCheck(const Expr& lhs, const Expr& rhs) final {
    if (!populate_mapping_) {
      return StructInfoBaseChecker::ShapeMatchCheck(lhs, rhs);
    }

    if (auto* ptr = lhs.as<VarNode>()) {
      auto var = GetRef<Var>(ptr);
      auto it = var_map_.find(var);
      // not populated
      if (it == var_map_.end()) {
        var_map_.Set(var, rhs);
        return BaseCheckResult::kPass;
      } else {
        // Best effort prove.
        Expr mapped_value = (*it).second;
        if (CanProveShapeEqual(mapped_value, rhs, analyzer_)) return BaseCheckResult::kPass;
        return BaseCheckResult::kFailL2;
      }
    }
    auto lhs_shape = lhs.as<ShapeExprNode>();
    auto rhs_shape = rhs.as<ShapeExprNode>();
    ICHECK(lhs_shape) << "lhs must have a shape";
    if (!rhs_shape) return BaseCheckResult::kFailL2;
    return ShapeMatchCheck(lhs_shape->values, rhs_shape->values);
  }

  BaseCheckResult FuncParamsCheck(const Array<StructInfo>& lhs,
                                  const Array<StructInfo>& rhs) final {
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
    auto ret = StructInfoBaseChecker::FuncParamsCheck(lhs, rhs);
    std::swap(populate_mapping_, populate_mapping);
    return ret;
  }
};

StructInfo DeriveCallRetStructInfo(const FuncStructInfo& finfo, const Call& call,
                                   const BlockBuilder& ctx, arith::Analyzer* ana) {
  if (ana == nullptr) {
    arith::Analyzer inst;
    return CallRetStructInfoDeriver(&inst).Derive(finfo, call, ctx);
  } else {
    return CallRetStructInfoDeriver(ana).Derive(finfo, call, ctx);
  }
}

TVM_REGISTER_GLOBAL("relax.analysis.DeriveCallRetStructInfo")
    .set_body_typed([](const FuncStructInfo& finfo, const Call& call, const BlockBuilder& ctx) {
      return DeriveCallRetStructInfo(finfo, call, ctx);
    });

//--------------------------
// UnifyToLCA
//--------------------------
class StructInfoLCAFinder
    : public StructInfoFunctor<StructInfo(const StructInfo&, const StructInfo&)> {
 public:
  explicit StructInfoLCAFinder(arith::Analyzer* ana) : analyzer_(ana) {}

  StructInfo VisitStructInfo(const StructInfo& lhs, const StructInfo& other) final {
    // quick path
    if (lhs.same_as(other)) return lhs;
    return StructInfoFunctor::VisitStructInfo(lhs, other);
  }

  // Object is based of everything, unify to object.
  StructInfo VisitStructInfo_(const ObjectStructInfoNode* lhs, const StructInfo& other) final {
    return GetRef<StructInfo>(lhs);
  }

  StructInfo VisitStructInfo_(const PrimStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<PrimStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);
    if (lhs->dtype == rhs->dtype) return GetRef<StructInfo>(lhs);
    // PrimType will be treated as their boxed(object) values
    // as a result we can unify to object.
    return ObjectStructInfo(lhs->span);
  }

  StructInfo VisitStructInfo_(const ShapeStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<ShapeStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);

    int ndim = lhs->ndim == rhs->ndim ? lhs->ndim : kUnknownNDim;
    if (lhs->ndim != rhs->ndim || !lhs->values.defined() || !rhs->values.defined() ||
        !CanProveShapeEqual(lhs->values.value(), rhs->values.value(), analyzer_)) {
      // prefers return same when possible
      if (!lhs->values.defined() && lhs->ndim == ndim) {
        return GetRef<StructInfo>(lhs);
      } else {
        return ShapeStructInfo(ndim, lhs->span);
      }
    }
    // equals to each other
    return GetRef<StructInfo>(lhs);
  }

  StructInfo VisitStructInfo_(const TensorStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TensorStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);

    // find the target dtype, ndim, and vdevice.
    DataType dtype = lhs->dtype == rhs->dtype ? lhs->dtype : DataType::Void();
    int ndim = lhs->ndim == rhs->ndim ? lhs->ndim : kUnknownNDim;
    VDevice vdev = VDevice();
    if (lhs->vdevice.defined() && rhs->vdevice.defined() &&
        lhs->vdevice.value() == rhs->vdevice.value()) {
      vdev = lhs->vdevice.value();
    }
    // if ndim mismatch or one side of shape is missing
    // then we cannot keep in symbolic shape
    if (lhs->ndim != rhs->ndim || !lhs->shape.defined() || !rhs->shape.defined() ||
        !CanProveShapeEqual(lhs->shape.value(), rhs->shape.value(), analyzer_)) {
      // reuse lhs when possible
      if (!lhs->shape.defined() && lhs->dtype == dtype && lhs->ndim == ndim &&
          (!lhs->vdevice.defined() || vdev.defined())) {
        return GetRef<StructInfo>(lhs);
      } else {
        return TensorStructInfo(dtype, ndim, vdev, lhs->span);
      }
    }
    // symbolic shape and vdevice match but dtype mismatch
    if (lhs->dtype != dtype || (lhs->vdevice.defined() && !vdev.defined())) {
      return TensorStructInfo(lhs->shape.value(), dtype, vdev, lhs->span);
    } else {
      return GetRef<StructInfo>(lhs);
    }
  }

  StructInfo VisitStructInfo_(const TupleStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<TupleStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);
    Optional<Array<StructInfo>> fields = UnifyArray(lhs->fields, rhs->fields);
    // tuple length not the same.
    if (!fields.defined()) return ObjectStructInfo(lhs->span);

    // same length tuple.
    if (!fields.same_as(lhs->fields)) {
      return TupleStructInfo(fields.value(), lhs->span);
    } else {
      return GetRef<StructInfo>(lhs);
    }
  }

  StructInfo VisitStructInfo_(const FuncStructInfoNode* lhs, const StructInfo& other) final {
    auto* rhs = other.as<FuncStructInfoNode>();
    if (rhs == nullptr) return ObjectStructInfo(lhs->span);

    // the unified function is pure only if both are pure
    bool purity = lhs->purity && rhs->purity;

    // lhs opaque handling
    if (lhs->IsOpaque()) {
      if (lhs->derive_func.defined()) {
        if (lhs->derive_func.same_as(rhs->derive_func)) {
          return GetRef<StructInfo>(lhs);
        } else {
          // Create a new opaque with object return
          return FuncStructInfo::OpaqueFunc(ObjectStructInfo(), purity, lhs->span);
        }
      } else {
        // no derivation function, only depends on ret
        StructInfo ret = this->VisitStructInfo(lhs->ret, rhs->ret);
        if (ret.same_as(lhs->ret)) return GetRef<StructInfo>(lhs);
        return FuncStructInfo::OpaqueFunc(ret, purity, lhs->span);
      }
    }
    // rhs is opaque, lhs is not
    if (rhs->IsOpaque()) {
      // unify ret value, note that rhs's ret is context free(because it is opaque)
      // so result of the unify is also context-free.
      StructInfo ret = this->VisitStructInfo(lhs->ret, rhs->ret);
      return FuncStructInfo::OpaqueFunc(ret, purity, lhs->span);
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
    if (struct_equal_(GetRef<StructInfo>(lhs), GetRef<StructInfo>(rhs))) {
      return GetRef<StructInfo>(lhs);
    }

    auto params = UnifyArray(lhs->params.value(), rhs->params.value());
    auto ret = this->VisitStructInfo(lhs->ret, rhs->ret);

    if (params.same_as(lhs->params) && ret.same_as(lhs->ret)) {
      return GetRef<StructInfo>(lhs);
    } else {
      // fail to unify the params
      if (!params.defined()) {
        return FuncStructInfo::OpaqueFunc(ret, purity, lhs->span);
      } else {
        return FuncStructInfo(params.value(), ret, purity, lhs->span);
      }
    }
  }

 private:
  // analyzer
  arith::Analyzer* analyzer_;
  // struct equal checker
  StructuralEqual struct_equal_;

  // check arrays
  Optional<Array<StructInfo>> UnifyArray(const Array<StructInfo>& lhs,
                                         const Array<StructInfo>& rhs) {
    if (lhs.same_as(rhs)) return lhs;
    if (lhs.size() != rhs.size()) return NullOpt;
    size_t index = 0;
    return lhs.Map([&](const StructInfo& a) { return this->VisitStructInfo(a, rhs[index++]); });
  }
};

StructInfo StructInfoLCA(const StructInfo& lhs, const StructInfo& rhs, arith::Analyzer* ana) {
  if (ana == nullptr) {
    arith::Analyzer inst;
    return StructInfoLCAFinder(&inst)(lhs, rhs);
  } else {
    return StructInfoLCAFinder(ana)(lhs, rhs);
  }
}

TVM_REGISTER_GLOBAL("relax.analysis.StructInfoLCA")
    .set_body_typed([](const StructInfo& lhs, const StructInfo& rhs) {
      return StructInfoLCA(lhs, rhs);
    });

//--------------------------
// TIRVarsInStructInfo
//--------------------------

class TIRVarsDetector : public StructInfoVisitor {
 public:
  enum class VarType {
    Definition,
    Usage,
  };
  TIRVarsDetector(VarType collection_type) : collection_type(collection_type) {}

  Array<tir::Var> GetTIRVars() const { return tir_vars_; }

 private:
  void VisitShape(Array<PrimExpr> shape) {
    for (const PrimExpr& value : shape) {
      if (collection_type == VarType::Definition) {
        if (auto opt = value.as<tir::Var>()) {
          RecordTIRVar(opt.value());
        }
      } else if (collection_type == VarType::Usage) {
        for (const tir::Var& tir_var : tir::UndefinedVars(value)) {
          RecordTIRVar(tir_var);
        }
      } else {
        LOG(FATAL) << "Invalid value for VarType enum, " << static_cast<int>(collection_type);
      }
    }
  }

  void VisitStructInfo_(const ShapeStructInfoNode* shape_sinfo) final {
    if (shape_sinfo->values.defined()) {
      VisitShape(shape_sinfo->values.value());
    }
  }

  void VisitStructInfo_(const TensorStructInfoNode* tensor_sinfo) final {
    if (tensor_sinfo->shape.defined()) {
      VisitStructInfo(GetStructInfo(tensor_sinfo->shape.value()));
    }
  }

  void RecordTIRVar(const tir::Var& tir_var) {
    auto insert_res = used_tir_vars_dedup_.insert(tir_var.get());
    if (insert_res.second) {
      tir_vars_.push_back(tir_var);
    }
  }

  Array<tir::Var> tir_vars_;
  std::unordered_set<const tir::VarNode*> used_tir_vars_dedup_;

  VarType collection_type;
};

Array<tir::Var> TIRVarsInStructInfo(const StructInfo& sinfo) {
  TIRVarsDetector detector(TIRVarsDetector::VarType::Usage);
  detector(sinfo);
  return detector.GetTIRVars();
}

Array<tir::Var> DefinableTIRVarsInStructInfo(const StructInfo& sinfo) {
  TIRVarsDetector detector(TIRVarsDetector::VarType::Definition);
  detector(sinfo);
  return detector.GetTIRVars();
}

TVM_REGISTER_GLOBAL("relax.analysis.TIRVarsInStructInfo").set_body_typed(TIRVarsInStructInfo);

TVM_REGISTER_GLOBAL("relax.analysis.DefinableTIRVarsInStructInfo")
    .set_body_typed(DefinableTIRVarsInStructInfo);

class SymbolicVarCollector : public relax::ExprVisitor,
                             public relax::StructInfoVisitor,
                             public tir::ExprVisitor {
 public:
  static Array<tir::Var> Free(const Expr& expr) {
    SymbolicVarCollector collector;
    collector.VisitExpr(expr);
    Array<tir::Var> ret{collector.free_symbolic_var_.begin(), collector.free_symbolic_var_.end()};
    return ret;
  }

  static Array<tir::Var> Defined(const Expr& expr) {
    SymbolicVarCollector collector;
    collector.VisitExpr(expr);
    Array<tir::Var> ret{collector.defined_symbolic_var_.begin(),
                        collector.defined_symbolic_var_.end()};
    return ret;
  }

 private:
  using relax::ExprVisitor::VisitExpr;
  using relax::ExprVisitor::VisitExpr_;
  using tir::ExprVisitor::VisitExpr;
  using tir::ExprVisitor::VisitExpr_;

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
        relax::StructInfoVisitor::VisitStructInfo(GetStructInfo(param));
      }
    });

    WithMode(VisitMode::kRequireDefinition, [&]() {
      for (Var param : op->params) {
        relax::StructInfoVisitor::VisitStructInfo(GetStructInfo(param));
      }
    });

    relax::ExprVisitor::VisitExpr_(op);
  }

  void VisitBinding_(const MatchCastNode* binding) final {
    WithMode(VisitMode(VisitMode::kProvideDefinition | VisitMode::kRequireDefinition),
             [&]() { this->VisitStructInfo(binding->struct_info); });

    relax::ExprVisitor::VisitBinding_(binding);
  }

  void VisitExprDepStructInfoField(const StructInfo& struct_info) {
    return this->VisitStructInfo(struct_info);
  }

  void VisitStructInfo_(const FuncStructInfoNode* op) final {
    if (op->params.defined()) {
      // Visit the parameters once to collect bindings, and another
      // time to collect usages.  Otherwise, a symbolic variable
      // defined by a later parameter may be treated as undefined when
      // used by an earlier parameter.
      WithMode(VisitMode::kProvideDefinition, [&]() {
        for (StructInfo param : op->params.value()) {
          this->VisitStructInfo(param);
        }
      });

      WithMode(VisitMode::kRequireDefinition, [&]() {
        for (StructInfo param : op->params.value()) {
          this->VisitStructInfo(param);
        }
      });
    }
    this->VisitStructInfo(op->ret);
  }

  void VisitStructInfoExprField(const Expr& expr) final {
    relax::ExprVisitor::VisitExpr(expr);
    if (auto* shape = expr.as<relax::ShapeExprNode>()) {
      for (const auto& val : shape->values) {
        this->VisitStructInfoExprField(val);
      }
    }
    if (auto prim_value = expr.as<relax::PrimValue>()) {
      this->VisitStructInfoExprField(prim_value.value()->value);
    }
  }

  void VisitStructInfoExprField(const PrimExpr& expr) final {
    if (mode_ & VisitMode::kProvideDefinition) {
      if (auto var = expr.as<tir::Var>()) {
        defined_symbolic_var_.insert(var.value());
      }
    }
    if (mode_ & VisitMode::kRequireDefinition) {
      tir::ExprVisitor::VisitExpr(expr);
    }
  }

  void VisitExpr_(const tir::VarNode* op) final {
    tir::Var var = GetRef<tir::Var>(op);
    // default mode, check defined.
    if (defined_symbolic_var_.count(var) == 0) {
      free_symbolic_var_.insert(var);
    }
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
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> defined_symbolic_var_;
  /*! \brief The set of free/undefined symbolic vars. */
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> free_symbolic_var_;
};

Array<tir::Var> DefinedSymbolicVars(const Expr& expr) {
  return SymbolicVarCollector::Defined(expr);
}
Array<tir::Var> FreeSymbolicVars(const Expr& expr) { return SymbolicVarCollector::Free(expr); }

TVM_REGISTER_GLOBAL("relax.analysis.DefinedSymbolicVars").set_body_typed(DefinedSymbolicVars);

TVM_REGISTER_GLOBAL("relax.analysis.FreeSymbolicVars").set_body_typed(FreeSymbolicVars);

}  // namespace relax
}  // namespace tvm
