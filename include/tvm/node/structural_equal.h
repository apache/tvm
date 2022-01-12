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
 * \file tvm/node/structural_equal.h
 * \brief Structural equality comparison.
 */
#ifndef TVM_NODE_STRUCTURAL_EQUAL_H_
#define TVM_NODE_STRUCTURAL_EQUAL_H_

#include <tvm/node/functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/data_type.h>

#include <string>

namespace tvm {

/*!
 * \brief Equality definition of base value class.
 */
class BaseValueEqual {
 public:
  bool operator()(const double& lhs, const double& rhs) const {
    // fuzzy float pt comparison
    constexpr double atol = 1e-9;
    if (lhs == rhs) return true;
    double diff = lhs - rhs;
    return diff > -atol && diff < atol;
  }

  bool operator()(const int64_t& lhs, const int64_t& rhs) const { return lhs == rhs; }
  bool operator()(const uint64_t& lhs, const uint64_t& rhs) const { return lhs == rhs; }
  bool operator()(const int& lhs, const int& rhs) const { return lhs == rhs; }
  bool operator()(const bool& lhs, const bool& rhs) const { return lhs == rhs; }
  bool operator()(const std::string& lhs, const std::string& rhs) const { return lhs == rhs; }
  bool operator()(const DataType& lhs, const DataType& rhs) const { return lhs == rhs; }
  template <typename ENum, typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  bool operator()(const ENum& lhs, const ENum& rhs) const {
    return lhs == rhs;
  }
};

/*!
 * \brief Content-aware structural equality comparator for objects.
 *
 *  The structural equality is recursively defined in the DAG of IR nodes via SEqual.
 *  There are two kinds of nodes:
 *
 *  - Graph node: a graph node in lhs can only be mapped as equal to
 *    one and only one graph node in rhs.
 *  - Normal node: equality is recursively defined without the restriction
 *    of graph nodes.
 *
 *  Vars(tir::Var, TypeVar) and non-constant relay expression nodes are graph nodes.
 *  For example, it means that `%1 = %x + %y; %1 + %1` is not structurally equal
 *  to `%1 = %x + %y; %2 = %x + %y; %1 + %2` in relay.
 *
 *  A var-type node(e.g. tir::Var, TypeVar) can be mapped as equal to another var
 *  with the same type if one of the following condition holds:
 *
 *  - They appear in a same definition point(e.g. function argument).
 *  - They points to the same VarNode via the same_as relation.
 *  - They appear in a same usage point, and map_free_vars is set to be True.
 */
class StructuralEqual : public BaseValueEqual {
 public:
  // inheritate operator()
  using BaseValueEqual::operator();
  /*!
   * \brief Compare objects via strutural equal.
   * \param lhs The left operand.
   * \param rhs The right operand.
   * \return The comparison result.
   */
  TVM_DLL bool operator()(const ObjectRef& lhs, const ObjectRef& rhs) const;
};

/*!
 * \brief A Reducer class to reduce the structural equality result of two objects.
 *
 * The reducer will call the SEqualReduce function of each objects recursively.
 * Importantly, the reducer may not directly use recursive calls to resolve the
 * equality checking. Instead, it can store the necessary equality conditions
 * and check later via an internally managed stack.
 */
class SEqualReducer : public BaseValueEqual {
 public:
  /*! \brief Internal handler that defines custom behaviors.. */
  class Handler {
   public:
    /*!
     * \brief Reduce condition to equality of lhs and rhs.
     *
     * \param lhs The left operand.
     * \param rhs The right operand.
     * \param map_free_vars Whether do we allow remap variables if possible.
     *
     * \return false if there is an immediate failure, true otherwise.
     * \note This function may save the equality condition of (lhs == rhs) in an internal
     *       stack and try to resolve later.
     */
    virtual bool SEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars) = 0;
    /*!
     * \brief Lookup the graph node equal map for vars that are already mapped.
     *
     *  This is an auxiliary method to check the Map<Var, Value> equality.
     * \param lhs an lhs value.
     *
     * \return The corresponding rhs value if any, nullptr if not available.
     */
    virtual ObjectRef MapLhsToRhs(const ObjectRef& lhs) = 0;
    /*!
     * \brief Mark current comparison as graph node equal comparison.
     */
    virtual void MarkGraphNode() = 0;
  };

  using BaseValueEqual::operator();

  /*! \brief default constructor */
  SEqualReducer() = default;
  /*!
   * \brief Constructor with a specific handler.
   * \param handler The equal handler for objects.
   * \param map_free_vars Whether or not to map free variables.
   */
  explicit SEqualReducer(Handler* handler, bool map_free_vars)
      : handler_(handler), map_free_vars_(map_free_vars) {}
  /*!
   * \brief Reduce condition to comparison of two objects.
   * \param lhs The left operand.
   * \param rhs The right operand.
   * \return the immediate check result.
   */
  bool operator()(const ObjectRef& lhs, const ObjectRef& rhs) const {
    return handler_->SEqualReduce(lhs, rhs, map_free_vars_);
  }
  /*!
   * \brief Reduce condition to comparison of two definitions,
   *        where free vars can be mapped.
   *
   *  Call this function to compare definition points such as function params
   *  and var in a let-binding.
   *
   * \param lhs The left operand.
   * \param rhs The right operand.
   * \return the immediate check result.
   */
  bool DefEqual(const ObjectRef& lhs, const ObjectRef& rhs) {
    return handler_->SEqualReduce(lhs, rhs, true);
  }
  /*!
   * \brief Reduce condition to comparison of two arrays.
   * \param lhs The left operand.
   * \param rhs The right operand.
   * \return the immediate check result.
   */
  template <typename T>
  bool operator()(const Array<T>& lhs, const Array<T>& rhs) const {
    // quick specialization for Array to reduce amount of recursion
    // depth as array comparison is pretty common.
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!(operator()(lhs[i], rhs[i]))) return false;
    }
    return true;
  }
  /*!
   * \brief Implementation for equality rule of var type objects(e.g. TypeVar, tir::Var).
   * \param lhs The left operand.
   * \param rhs The right operand.
   * \return the result.
   */
  bool FreeVarEqualImpl(const runtime::Object* lhs, const runtime::Object* rhs) const {
    // var need to be remapped, so it belongs to graph node.
    handler_->MarkGraphNode();
    // We only map free vars if they corresponds to the same address
    // or map free_var option is set to be true.
    return lhs == rhs || map_free_vars_;
  }

  /*! \return Get the internal handler. */
  Handler* operator->() const { return handler_; }

 private:
  /*! \brief Internal class pointer. */
  Handler* handler_;
  /*! \brief Whether or not to map free vars. */
  bool map_free_vars_;
};

}  // namespace tvm
#endif  // TVM_NODE_STRUCTURAL_EQUAL_H_
