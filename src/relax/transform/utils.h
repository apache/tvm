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
 * \brief Remove unused global relax functions in an IRModule.
 * \param mod The target module
 * \param entry_functions list of entry functions
 * \return The updated module.
 */
TVM_DLL IRModule RemoveUnusedFunctions(IRModule mod, Array<runtime::String> entry_funcs);

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
 * \return A new module containing grouped functions.
 */
IRModule MakeGroupedFunctions(
    IRModule mod,
    const std::unordered_map<const Object*, relay::GraphPartitioner::Group*>& partition,
    bool lift_constants = true);

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

  Expr VisitExpr_(const DataflowVarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var->vid);
    return it == var_remap_.end() ? var : it->second;
  }

  const VarMap& var_remap_;
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

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_UTILS_H_
