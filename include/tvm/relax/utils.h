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
 * \file tvm/relax/utils.h
 * \brief Utility classes and functions for working with the Relax IR.
 */
#ifndef TVM_RELAX_UTILS_H_
#define TVM_RELAX_UTILS_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace tvm {
namespace relax {

/*!
 * \brief Utility data structure for generating unique names for IR construction.
 */
class NameTable {
 public:
  /*!
   * \brief Generate a unique name with a specified prefix.
   * \param prefix The name prefix.
   * \return The generated name.
   */
  inline std::string GetUniqueName(std::string prefix) {
    std::replace(prefix.begin(), prefix.end(), '.', '_');
    std::string unique_prefix = prefix;
    auto it = alloc_map_.find(prefix);
    if (it != alloc_map_.end()) {
      while (alloc_map_.count(unique_prefix = prefix + std::to_string(++it->second)) > 0) {
      }
    }
    alloc_map_[unique_prefix] = 0;
    return unique_prefix;
  }

  NameTable() = default;

  template <typename Iter, typename Lambda>
  explicit NameTable(Iter begin, Iter end, Lambda f) {
    // static_assert is more reader-friendly than SFINAE when template specialization is not needed.
    static_assert(std::is_convertible<decltype(f(*begin)), std::string>::value,
                  "Lambda f must has a signature of [?](*it) -> string {}");
    for (auto it = begin; it != end; ++it) {
      const std::string& name = f(*it);
      const size_t idx_last_first_num = std::distance(
          std::find_if(name.rbegin(), name.rend(), [](char c) { return !std::isdigit(c); }),
          name.rend());
      // name = {O = others}{D = consecutive digits}
      // let O -> prefix;
      std::string prefix = name.substr(0, idx_last_first_num);
      ICHECK(prefix.size() > 0 && std::isalpha(prefix[0])) << "Invalid variable name: " << name;
      if (0 == alloc_map_.count(prefix)) alloc_map_[prefix] = 0;
      if (idx_last_first_num < name.size()) {  // has some digits.
                                               // let D's nearest natural number -> idx;
                                               // note: stoul("000123") = 123;
        alloc_map_[prefix] =
            std::max(alloc_map_[prefix], std::stoi(name.substr(idx_last_first_num)));
      }
    }
  }

  template <typename Iter>
  explicit NameTable(Iter begin, Iter end)
      : NameTable(begin, end, [](const decltype(*begin)& v) { return v; }) {}

 private:
  std::unordered_map<std::string, int> alloc_map_;
};

/*!
 * \brief Bind the variables to a Relax expression. This is a helper
 * function usually called by other pass functions to help optimizations.
 * If any free variables are introduced into a function, those are added
 * to the function parameters.
 * Additionally this may change the order of parameters if you map a variable
 * to a variable.
 *
 * \param expr The input expression.
 * \param binds The variable to expression map that will be used to help the
 *        binding.
 * \param symbolic_var_map The map from symbolic var to the expr it binds to.
 *
 * \return The updated expression.
 */
TVM_DLL Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds,
                  const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map = {});

/*!
 * \brief Check if the given StructInfo is for a boolean scalar (tensor of rank 0 with a boolean
 * dtype).
 *
 * \param sinfo The input StructInfo.
 * \param permit_unknown_rank If true, it will permit the input type to have unknown rank
 *   (ndim of -1), which will require a dynamic check.
 * \param permit_unknown_dtype If true, it will permit the input type to have an unknown dtype
 *   (namely, void), which will require a dynamic check.
 *
 * \return True iff the input type is a boolean scalar type (or, depending on options, has unknown
 *   rank or dtype)
 */
TVM_DLL bool IsBoolStructInfo(const StructInfo& sinfo, bool permit_unknown_rank = true,
                              bool permit_unknown_dtype = true);

/*!
 * \brief Check if the given expression is a "leaf" node or tuple node for normalization purposes.
 *
 *    The following expressions are defined as leaf nodes: Var, Constant, ShapeExpr,
 *    GlobalVar, Op, ExternFunc.
 *
 *    Tuples are included in this list mainly for convenience in grouping operator arguments.
 *    *Note*: Since tuples can contain nested expressions, it is necessary to ensure that
 *    values nested inside them are also leaves.
 *
 * \param expr The input expression
 *
 * \return True iff the input expression is a "leaf" node (a value allowed to appear
 *    inline without being bound to a var during normalization).
 */
TVM_DLL bool IsLeafOrTuple(const Expr& expr);

/*!
 * \brief Copy the given function. All variables that are bound inside the original function
 *  would be copied to satisfy the restriction in the well-formed check: Variables in
 *  Relax must be bound exactly once. This also ensures that both the function and its copy
 *  can be inserted into the same IRModule, and be asserted on the structural equality
 *  agaisnt IRModule created by TVMScript.
 *
 * \param func The relax function to copy.
 * \return The copied function.
 */
TVM_DLL Function CopyWithNewVars(Function func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_UTILS_H_
