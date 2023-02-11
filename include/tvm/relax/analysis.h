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
 * \file tvm/relax/analysis.h
 * \brief The set of Relax specific analysis on IR.
 */
#ifndef TVM_RELAX_ANALYSIS_H_
#define TVM_RELAX_ANALYSIS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/diagnostic.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/function.h>

#include <functional>
#include <utility>

namespace tvm {
namespace relax {
//-----------------------------------
// Shape expression analysis
//----------------------------------
/*!
 * \brief Can prove the two symbolic shape arrays equals to each other.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana The analyzer used for integer analysis.
 * \return The prove result.
 *
 * \note This function does best effort prove, which means
 *       if result is false, there is still possibility that
 *       two shapes equals to each other during runtime.
 */
TVM_DLL bool CanProveShapeEqual(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs,
                                arith::Analyzer* ana);

/*!
 * \brief Can prove the two symbolic shape expressions equals to each other.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana The analyzer used for integer analysis.
 *
 * \note This function does best effort prove, which means
 *       if result is false, there is still possibility that
 *       two shapes equals to each other during runtime.
 */
TVM_DLL bool CanProveShapeEqual(const Expr& lhs, const Expr& rhs, arith::Analyzer* ana);

//-----------------------------------
// Foundational StructInfo analysis
//-----------------------------------
/*!
 * \brief Get the corresponding static type from a given struct info.
 * \param info The struct info.
 * \return the corresponding static type.
 */
TVM_DLL Type GetStaticType(const StructInfo& info);

/*!
 * \brief Get the corresponding struct info from static type.
 * \param type The input type
 * \return the corresponding struct info.
 */
TVM_DLL StructInfo StructInfoFromType(const Type& type);

/*!
 * \return Derive the call's ret value struct info from inputs.
 * \param finfo The function struct info.
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 * \return The derived struct info of the call.
 * \note  call->op field is ignored during derivation and we only rely on information
 *        presented by func_sinfo.
 */
TVM_DLL StructInfo DeriveCallRetStructInfo(const FuncStructInfo& finfo, const Call& call,
                                           const BlockBuilder& ctx, arith::Analyzer* ana = nullptr);

/*!
 * \brief Erase the info to a corresponding more coarse grained
 *        struct info that is still well-defined(with all the vars in scope).
 *
 * When we are returning a StructInfo to another scope,
 * it is important to remember that StructInfo may carry
 * dependencies on var that is not defined the other scope.
 *
 * In such cases, it is important to call EraseToWellDefined to get
 * another StructInfo that **only** contains the vars that are defined
 * in the target scope.
 *
 * For example, consider the following function
 *
 * \code
 *
 * @R.function
 * def f(x: R.Tensor[(n, m)]):
 *     k = tir.Var("k", "int64")
 *     v0 = opaque_fn(x)
 *     v1 = match_cast(v0, R.Tensor[(n, k)])
 *     v2 : R.Tensor[(n + 1, k + 2)] = pad(v1)
 *     return v2
 *
 * \endcode
 *
 * In the above code, the return value y have shape `(n + 1, k + 2)`,
 * However, at the level of function signature, only n, m are defined,
 * k is undefined here.
 *
 * When we call EraseToWellDefined(R.Tensor[(n + 1, k + 2)], fshape_var_map={n: n, m: m}),
 * we will obtain R.Tensor(ndim=2), which is an erased info that does not depend
 * on k(which is undefined from parameter signature).
 *
 * However, if we call EraseToWellDefined(R.Tensor[(n + 1, m)], fshape_var_map={n: n, m: m}),
 * Then the return value will be R.Tensor[(n + 1, m)], because both n and m are defined.
 *
 * We can also make these var map to return a different expression.
 * For example, EraseToWellDefined(R.Tensor[(n + 1, m)], fshape_var_map={n: 2, m: m})
 * will give us R.Tensor[(3, m)], where n get replaced by 2.
 *
 * Use this function in the following scenarios:
 * - Decide the struct_info of expr with sub-scopes, such as If, SeqExpr
 * - Decide the deduced return struct_info of a function that can be fully decided by params.
 *
 * \param info The struct info.
 * \param f_shape_var_map callback function to specify
 *        whether a symbolic shape var is defined and the value it maps to,
 *        return nullopt if var is undefined.
 * \param f_var_map callback function to specify
 *        whether a var is defined in the target scope and the value it maps to,
 *        return nullopt if var is undefined.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 *
 * \return the corresponding erased struct info.
 */
TVM_DLL StructInfo
EraseToWellDefined(const StructInfo& info,
                   std::function<Optional<PrimExpr>(const tir::Var& var)> f_shape_var_map = nullptr,
                   std::function<Optional<Expr>(const Var& var)> f_var_map = nullptr,
                   arith::Analyzer* ana = nullptr);

/*!
 * \brief EraseToWellDefined variant with map.
 * \param info The struct info.
 * \param shape_var_map map to specify
 *        whether a symbolic shape var is defined and the value it maps to,
 *        return nullopt if var is undefined.
 * \param var_map map to specify
 *        whether a var is defined in the target scope and the value it maps to,
 *        return nullopt if var is undefined.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 *
 * \return the corresponding erased struct info.
 */
TVM_DLL StructInfo EraseToWellDefined(const StructInfo& info, Map<tir::Var, PrimExpr> shape_var_map,
                                      Map<Var, Expr> var_map, arith::Analyzer* ana = nullptr);

/*!
 * \brief Fine grained result of base check.
 *
 * This analysis comes with different levels of checking failures
 * that can help to customize the compilation decisions.
 *
 * For a given pair of lhs_struct_info, rhs_struct_info. We adopt
 * the following terminology:
 * - LSet = {value | value matches lhs_struct_info}
 * - RSet = {value | value matches rhs_struct_info}
 *
 * See the definition of each level below.
 */
enum class BaseCheckResult {
  /*!
   * \brief The two value sets have no intersection at all: Interset(LSet, RSet) = empty
   */
  kFailL0 = 0,
  /*!
   * \brief LSet is not superset of RSet by only looking at static information.
   *
   * \note This level will trigger static type checking error when lhs is param and rhs is arg.
   */
  kFailL1 = 1,
  /*!
   * \brief WLSet is not superset of RSet because of mismatch in value information.
   *
   * L1-level mismatches in params of FuncStructInfo is categorized as
   * If lhs is FuncStructInfo, then L1-level mismatch in its params
   * is categorized as L2-level mismatch for lhs.
   *
   * Design considerations for functions:
   * - (a) We want to be able to erase type/value in function signature
   *       when we unify function struct info and preserve simpler representations.
   * - (b) We automatically insert match_cast at function boundary, so
   *       we can erase (int)->int argument as (object)->int.
   *       The input shape/type mismatch will be detected by runtime checks at function boundary.
   *       This behavior is also consistent with the PackedFunc behavior.
   *
   * \note This level means there is no problem about static known information.
   *       It is OK for the checker to do best effort and return this value.
   */
  kFailL2 = 2,
  /*! \brief LSet is superset of RSet. */
  kPass = 3
};

/*!
 * \brief Run a base check to see if base subsumes derived.
 *
 * This function returns fine-grained base-check result on reasons of failure.
 *
 * \param base The base struct info.
 * \param derived The derived struct info.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 * \return Whether the relation holds.
 *
 * \sa BaseCheckResult
 */
TVM_DLL BaseCheckResult StructInfoBaseCheck(const StructInfo& base, const StructInfo& derived,
                                            arith::Analyzer* ana = nullptr);

/*!
 * \brief Check the relation of two struct info to see if one subsumes another one.
 *
 * \param base The base struct info.
 * \param derived The derived struct info.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 * \return Whether the relation holds.
 */
TVM_DLL bool IsBaseOf(const StructInfo& base, const StructInfo& derived,
                      arith::Analyzer* ana = nullptr);

/*!
 * \brief Unify the two struct info to their least common ancestor.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana Optional context analyzer to prove symbolic expression equality.
 * \return The unified information.
 */
TVM_DLL StructInfo StructInfoLCA(const StructInfo& lhs, const StructInfo& rhs,
                                 arith::Analyzer* ana = nullptr);

/*!
 * \brief Check if the given PrimFunc is essentially doing a reshape operation.
 * The reshape operation also includes expand_dims, squeeze, flatten, etc.
 * \details Here the allowed reshape pattern is: for example, assume the operation is
 *  `B[l_0, l_1, ..., l_b] = A[r_0, r_1, ..., r_a]`, we check if we can prove that the flattened
 * index of l_0, ..., l_b under buffer B equals to the flattened index of r_0, ..., r_a under
 * buffer A.
 * \param func The function to be examined.
 * \return A boolean indicating if the given PrimFunc is doing a reshape.
 * \note According to the description above, the returned result can only be false-negative and
 * cannot be false-positive, since whenever we cannot prove the equality, we return false. This
 * property guarantees the safety of this function.
 */
TVM_DLL bool HasReshapePattern(const tir::PrimFunc& func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_H_
