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

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/data_type.h>

#include <cmath>
#include <string>

namespace tvm {

/*!
 * \brief Equality definition of base value class.
 */
class BaseValueEqual {
 public:
  bool operator()(const double& lhs, const double& rhs) const {
    if (std::isnan(lhs) && std::isnan(rhs)) {
      // IEEE floats do not compare as equivalent to each other.
      // However, for the purpose of comparing IR representation, two
      // NaN values are equivalent.
      return true;
    } else if (std::isnan(lhs) || std::isnan(rhs)) {
      return false;
    } else if (lhs == rhs) {
      return true;
    } else {
      // fuzzy float pt comparison
      constexpr double atol = 1e-9;
      double diff = lhs - rhs;
      return diff > -atol && diff < atol;
    }
  }

  bool operator()(const int64_t& lhs, const int64_t& rhs) const { return lhs == rhs; }
  bool operator()(const uint64_t& lhs, const uint64_t& rhs) const { return lhs == rhs; }
  bool operator()(const ffi::Optional<int64_t>& lhs, const ffi::Optional<int64_t>& rhs) const {
    return lhs == rhs;
  }
  bool operator()(const ffi::Optional<double>& lhs, const ffi::Optional<double>& rhs) const {
    return lhs == rhs;
  }
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
 *  Vars(tir::Var, relax::Var) nodes are graph nodes.
 *
 *  A var-type node(e.g. tir::Var) can be mapped as equal to another var
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
   * \param map_free_params Whether or not to map free variables.
   * \return The comparison result.
   */
  TVM_DLL bool operator()(const ffi::Any& lhs, const ffi::Any& rhs,
                          const bool map_free_params = false) const;
};

}  // namespace tvm
#endif  // TVM_NODE_STRUCTURAL_EQUAL_H_
