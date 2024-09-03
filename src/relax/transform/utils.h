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
 * \file src/relax/transform/utils.h
 * \brief Additional utility classes and functions for working with the Relax IR.
 */
#ifndef TVM_RELAX_TRANSFORM_UTILS_H_
#define TVM_RELAX_TRANSFORM_UTILS_H_

#include <builtin_fp16.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/expr_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../relay/analysis/graph_partitioner.h"
#include "../../support/array.h"
#include "../op/nn/convolution.h"
#include "../op/nn/nn.h"
#include "../op/nn/pooling.h"
#include "../op/tensor/binary.h"
#include "../op/tensor/create.h"
#include "../op/tensor/datatype.h"
#include "../op/tensor/index.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"
#include "../op/tensor/search.h"
#include "../op/tensor/set.h"
#include "../op/tensor/statistical.h"
#include "../op/tensor/ternary.h"
#include "../op/tensor/unary.h"

namespace tvm {
namespace relax {

/*!
 * \brief A simple wrapper around ExprFunctor for a single argument case.
 *  The result of visit is memoized.
 */
template <typename OutputType>
class MemoizedExprTranslator : public ::tvm::relax::ExprFunctor<OutputType(const Expr&)> {
  using BaseFunctor = ::tvm::relax::ExprFunctor<OutputType(const Expr&)>;

 public:
  /*! \brief virtual destructor */
  virtual ~MemoizedExprTranslator() {}

  /*!
   * \brief The memoized call.
   * \param n The expression node.
   * \return The result of the call
   */
  virtual OutputType VisitExpr(const Expr& n) {
    ICHECK(n.defined());
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return it->second;
    }
    auto res = BaseFunctor::VisitExpr(n);
    memo_[n] = res;
    return res;
  }

  virtual OutputType VisitExpr_(const VarNode* vn) {
    ICHECK(memo_.count(GetRef<Expr>(vn)));
    return memo_[GetRef<Expr>(vn)];
  }

  virtual OutputType VisitBinding_(const VarBindingNode* binding) {
    ICHECK_EQ(memo_.count(binding->var), 0);
    auto v = VisitExpr(binding->value);
    memo_[binding->var] = v;
    return v;
  }

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, OutputType, ObjectPtrHash, ObjectPtrEqual> memo_;
};

/*!
 * \brief Dead code elimination
 * Currently it removes:
 *   1. Unused local VarBindings in a DataflowBlock.
 *      The used var set is set to empty at the beginning of each DataflowBlock.
 *      We reverse scan the DataflowBlock, if a VarBinding
 *        - bindings to a dataflowvar, or
 *        - is used in the used var set
 *      We keep it and add its var to the used var set. Otherwise, we remove it.
 *   2. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 * \param mod The target module
 * \param entry_functions list of entry functions
 * \return The updated module.
 */
TVM_DLL IRModule DeadCodeElimination(const IRModule& mod, Array<runtime::String> entry_funcs);

/*!
 * \brief Get the external symbol of the Relax function name.
 *
 * \param func The provided function.
 * \return An external symbol.
 */
inline std::string GetExtSymbol(const Function& func) {
  const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(name_node.defined()) << "Fail to retrieve external symbol.";
  return std::string(name_node.value());
}

/*!
 * \brief Fuse ops or functions according to the given partition, and grouped them into a new
 * function.
 *
 * \param mod The input module.
 * \param partition A mapping from a subexpression to the containing group.
 * \param lift_constants Whether or not to lift bound constants to parameters of the
 * grouped function.
 * \param entry_function_names The names of the entry functions.
 * \return A new module containing grouped functions.
 */
IRModule MakeGroupedFunctions(
    IRModule mod,
    const std::unordered_map<const Object*, relay::GraphPartitioner::Group*>& partition,
    bool lift_constants = true, const Array<String>& entry_function_names = {});

/*!
 * \brief Check if the given StructInfo is a scalar tensor. The sinfo should be an instance of
 * TensorStructInfo; its shape must be ShapeExpr.
 * \param sinfo The StructInfo to be checked.
 * \return true if the given StructInfo is a scalar tensor.
 */
bool IsScalarTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a scalar tensor. Now the shape of the tensor expr must be
 * ShapeExpr.
 * \param expr The expr to be checked.
 * \return true if the given expr is a scalar tensor.
 */
bool IsScalarTensor(const Expr& expr);

/*!
 * \brief Check if the given StructInfo is a nested tensor StructInfo satisfying the given
 * condition f_condition.
 * \param sinfo The StructInfo to be checked.
 * \param f_condition The condition function for each leaf StructInfo with signature
 * `bool f_condition(TensorStructInfo)`.
 * \tparam FType The condition function type.
 * \return true if the given StructInfo is a nested tensor satisfying the given f_condition.
 */
template <typename FType>
bool IsNestedTensorConditioned(const StructInfo& sinfo, FType f_condition) {
  if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
    return f_condition(GetRef<TensorStructInfo>(tensor_sinfo));
  } else if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
    return !std::any_of(
        tuple_sinfo->fields.begin(), tuple_sinfo->fields.end(),
        [&](const StructInfo& field) { return !IsNestedTensorConditioned(field, f_condition); });
  }
  return false;
}

/*!
 * \brief Check if the given StructInfo is a nested tensor.
 * \param sinfo The StructInfo to be checked.
 * \return true if the given StructInfo is a nested tensor.
 */
bool IsNestedTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a nested tensor.
 * \param expr The expr to be checked.
 * \return true if the given expr is a nested tensor.
 */
bool IsNestedTensor(const Expr& expr);

// TODO(@bohan): implements some postorder function accepts a visitor closure
class VarReplacer : public ExprMutator {
 public:
  using VarMap = std::unordered_map<Id, Var, ObjectPtrHash, ObjectPtrEqual>;

  explicit VarReplacer(const VarMap& var_remap) : var_remap_(var_remap) {}

  static Expr Replace(const Expr& expr, const VarMap& var_remap) {
    VarReplacer replacer(var_remap);
    return replacer(expr);
  }

 private:
  Expr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var->vid);
    return it == var_remap_.end() ? var : it->second;
  }

  const VarMap& var_remap_;
};

/*!
 * \brief Renew the definition of symbolic vars in Relax.
 * \details This mutator is used to prevent the same symbolic var from being used in different
 *          functions, which is malformed.
 */
class SymbolicVarRenewMutator : public ExprMutator, tir::ExprMutator {
 public:
  static Function Renew(const Function& function) {
    SymbolicVarRenewMutator mutator;
    return Downcast<Function>(mutator.VisitExpr(function));
  }
  SymbolicVarRenewMutator() = default;

 protected:
  using relax::ExprMutator::VisitExpr;
  using relax::ExprMutator::VisitExpr_;
  using tir::ExprMutator::VisitExpr_;

  PrimExpr VisitPrimExpr(const PrimExpr& expr) final { return tir::ExprMutator::VisitExpr(expr); }

  // TODO(Siyuan): enhance the method to the following steps:
  // 1. Visit and replace all tir::Vars at the definition point
  // 2. Revisit the function again and update the use side.
  PrimExpr VisitExpr_(const tir::VarNode* op) final {
    auto it = var_map_.find(GetRef<tir::Var>(op));
    if (it != var_map_.end()) {
      return (*it).second;
    } else {
      auto n = make_object<tir::VarNode>(*op);
      tir::Var v(n);
      var_map_.Set(GetRef<tir::Var>(op), v);
      return v;
    }
  }

  Expr VisitExpr_(const FunctionNode* op) {
    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : op->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      if (!param.same_as(new_param)) {
        var_remap_[param->vid] = new_param;
        all_params_unchanged = false;
      }
    }

    Expr body = this->VisitWithNewScope(op->body, params);

    if (all_params_unchanged && body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      auto new_ret_sinfo = this->VisitExprDepStructInfoField(op->ret_struct_info);
      return Function(params, body, new_ret_sinfo, op->is_pure, op->attrs);
    }
  }

  Map<tir::Var, tir::Var> var_map_;
};

/*!
 * \brief Copy a function while renewing the relax Vars and the tir Vars.
 * \details All variables that are bound inside the original function would be copied to satisfy
 * the restriction in the well-formed check: Variables in Relax must be bound exactly once.
 */
class FunctionCopier : public SymbolicVarRenewMutator {
 public:
  FunctionCopier() = default;
  Function Copy(Function func) { return Downcast<Function>(VisitExpr(func)); }
  Map<Var, Var> GetVarMap() { return relax_var_map_; }

 private:
  using relax::ExprMutator::VisitExpr;

  Var VisitVarDef_(const DataflowVarNode* var) override {
    Var new_var = SymbolicVarRenewMutator::VisitVarDef_(var);
    Var copied_var = DataflowVar(new_var->name_hint(), GetStructInfo(new_var), new_var->span);
    var_remap_[var->vid] = copied_var;
    relax_var_map_.Set(GetRef<Var>(var), copied_var);
    return copied_var;
  }

  Var VisitVarDef_(const VarNode* var) override {
    Var new_var = SymbolicVarRenewMutator::VisitVarDef_(var);
    Var copied_var = Var(new_var->name_hint(), GetStructInfo(new_var), new_var->span);
    var_remap_[var->vid] = copied_var;
    relax_var_map_.Set(GetRef<Var>(var), copied_var);
    return copied_var;
  }

  Map<Var, Var> relax_var_map_;
};

/*!
 * \brief Create a Constant with a scalar
 *
 * \param dtype The data type.
 * \param value The value of the scalar.
 * \return A Constant.
 */
template <typename T>
inline Constant MakeConstantScalar(T value, DataType dtype) {
  runtime::NDArray arr = runtime::NDArray::Empty({}, dtype, {kDLCPU, 0});
  if (dtype == DataType::Float(32)) {
    *static_cast<float*>(arr->data) = static_cast<float>(value);
  } else if (dtype == DataType::Float(64)) {
    *static_cast<double*>(arr->data) = static_cast<double>(value);
  } else if (dtype == DataType::Int(32)) {
    *static_cast<int32_t*>(arr->data) = static_cast<int32_t>(value);
  } else if (dtype == DataType::Int(64)) {
    *static_cast<int64_t*>(arr->data) = static_cast<int64_t>(value);
  } else if (dtype == DataType::UInt(1)) {
    *static_cast<bool*>(arr->data) = static_cast<bool>(value);
  } else if (dtype == DataType::UInt(8)) {
    *static_cast<uint8_t*>(arr->data) = static_cast<uint8_t>(value);
  } else if (dtype == DataType::UInt(16)) {
    *static_cast<uint16_t*>(arr->data) = static_cast<uint16_t>(value);
  } else if (dtype == DataType::UInt(32)) {
    *static_cast<uint32_t*>(arr->data) = static_cast<uint32_t>(value);
  } else if (dtype == DataType::UInt(64)) {
    *static_cast<uint64_t*>(arr->data) = static_cast<uint64_t>(value);
  } else if (dtype == DataType::Int(8)) {
    *static_cast<int8_t*>(arr->data) = static_cast<int8_t>(value);
  } else if (dtype == DataType::Int(16)) {
    *static_cast<int16_t*>(arr->data) = static_cast<int16_t>(value);
  } else if (dtype == DataType::Int(32)) {
    *static_cast<int32_t*>(arr->data) = static_cast<int32_t>(value);
  } else if (dtype == DataType::Int(64)) {
    *static_cast<int64_t*>(arr->data) = static_cast<int64_t>(value);
  } else if (dtype == DataType::Float(16)) {
    // convert to float16 storage is uint16_t
    *static_cast<uint16_t*>(arr->data) =
        __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(static_cast<float>(value));
  } else if (dtype == DataType::BFloat(16)) {
    // convert to bfloat16 storage is uint16_t
    *static_cast<uint16_t*>(arr->data) =
        __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 7>(static_cast<float>(value));
  } else {
    LOG(FATAL) << "Unsupported dtype " << dtype;
  }
  return Constant(arr);
}

inline Array<Integer> GetOrderedPositiveAxes(const Array<Integer>& axes, int ndim) {
  std::vector<int64_t> ret;
  ret.reserve(axes.size());
  for (const auto& axis : axes) {
    int64_t axis_val = axis->value;
    if (axis_val < 0) {
      axis_val += ndim;
    }
    ICHECK(axis_val >= 0 && axis_val < ndim) << "axis " << axis << " is out of bounds for array of "
                                             << "dimension " << ndim;
    ret.push_back(axis_val);
  }
  std::sort(ret.begin(), ret.end());
  return support::AsArray<int64_t, Integer>(ret);
}

inline String GetCodegenName(const std::string& composite_name) {
  auto delim_pos = composite_name.find(".");
  ICHECK(delim_pos != std::string::npos) << "The pattern name for a composite function should "
                                            "start with a compiler name followed by period.";
  return composite_name.substr(0, delim_pos);
}

inline int GetDeviceIndex(const IRModule& mod, const VDevice& vdevice) {
  Array<GlobalInfo> vdevices = mod->global_infos["vdevice"];
  for (int i = 0; i < static_cast<int>(vdevices.size()); ++i) {
    if (vdevices[i] == vdevice) {
      return i;
    }
  }
  LOG(FATAL) << "The vdevice is not in the ir_module.";
  return -1;
}

/* \brief Eliminate common subexpressions
 *
 * Utility for simplifying relax expressions by removing common
 * subexpressions.
 *
 * \param expr The expression to be updated
 *
 * \param call_only If true, only eliminate relax::Call nodes.  If
 * false, eliminate any common subexpressions.
 *
 * \ret The updated expression
 */
Expr EliminateCommonSubexpr(const Expr& expr, bool call_only = false);

/* \brief Remove use of trivial bindings
 *
 * Utility for simplifying relax expressions by folding var bindings
 * and match shape nodes.  May include other forms of simplification
 * in the future.  Ideally should be used before constant folding and
 * eliminating unused bindings.
 *
 * \param expr The expression to be canonicalized
 *
 * \ret The canonicalized expression
 */
Expr CanonicalizeBindings(Expr expr);

/* \brief Remove use of trivial bindings
 *
 * Utility for converting from individual model parameters to a single
 * parameter with a tuple of parameters.  If the `kNumInput` attribute
 * is absent, no model parameters are present, so no updates are made.
 *
 * \param func The function to be updated.
 *
 * \param param_tuple_name The name of the tuple parameter.  If
 * unspecified, defaults to "model_params"
 *
 * \ret The updated function.
 */
Function BundleModelParams(const Function& func, Optional<String> param_tuple_name = NullOpt);

/*! \brief Compose two functions
 *
 * Given two functions `func_a` and `func_b`, produce `func_c` such
 * that `func_c(x)` is equivalent to `func_b(func_a(x))`.
 *
 * If the output if `func_a` is not usable as the input of `func_b`,
 * an error will be raised.
 *
 * \param func_a The first function to be composed.
 * \param func_b The second function to be composed.
 * \return The composed function
 */
TVM_DLL Function ComposeFunctions(Function func_a, Function func_b);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_UTILS_H_
