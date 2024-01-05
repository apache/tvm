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
 * \file src/contrib/msc/core/transform/layout_utils.h
 * \brief Common utilities for layout.
 */
#ifndef TVM_CONTRIB_MSC_CORE_TRANSFORM_LAYOUT_UTILS_H_
#define TVM_CONTRIB_MSC_CORE_TRANSFORM_LAYOUT_UTILS_H_

#include <tvm/ir/source_map.h>
#include <tvm/relax/expr.h>

#include <vector>

#include "../../../../relax/transform/infer_layout_utils.h"
#include "../../../../relax/transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using Expr = tvm::RelayExpr;
using namespace tvm::relax;

/*!
 * \brief Utils for Layout.
 */
class LayoutUtils {
 public:
  /*!
   * \brief Infer NLayout.
   * \return The NLayout.
   */
  TVM_DLL static NLayout InferNLayout(const Expr& expr, const VarLayoutMap& var_layout_map);

  /*!
   * \brief Infer LayoutDecision.
   * \return The LayoutDecision.
   */
  TVM_DLL static LayoutDecision InferLayoutDecision(const Expr& expr,
                                                    const VarLayoutMap& var_layout_map);

  /*!
   * \brief Infer LayoutDecision at given pos.
   * \return The LayoutDecision.
   */
  TVM_DLL static LayoutDecision InferLayoutDecisionAt(const Expr& expr,
                                                      const VarLayoutMap& var_layout_map,
                                                      size_t index = 0);

  /*!
   * \brief Check if the layout is infered.
   * \return Whether the layout is infered.
   */
  TVM_DLL static bool LayoutInfered(const Expr& expr);

  /*!
   * \brief Set the layout to span
   * \return Whether the layout is setted.
   */
  TVM_DLL static bool SetLayout(const Expr& expr, const NLayout& layout);

  /*!
   * \brief Get the layout from span
   * \return The NLayout.
   */
  TVM_DLL static const NLayout GetNLayout(const Expr& expr);

  /*!
   * \brief Get the layout desion from span
   * \return The LayoutDecision.
   */
  TVM_DLL static const LayoutDecision GetLayoutDecision(const Expr& expr);

  /*!
   * \brief Check if the layout has unknown dim tensor.
   * \return Whether the layout has unknown dim tensor.
   */
  TVM_DLL static bool HasUnknownDimTensor(const NLayout& nlayout);

  /*!
   * \brief Check if the args has unknown dim tensor.
   * \return Whether the args has unknown dim tensor.
   */
  TVM_DLL static bool HasUnknownDimTensor(const Array<Expr>& args);

  /*!
   * \brief Insert axes to the Layout
   * \return The new layout.
   */
  TVM_DLL static const LayoutDecision ExpandLayout(const LayoutDecision& src_layout,
                                                   const std::vector<size_t>& expand_axes);

  /*!
   * \brief Delete axes from the Layout
   * \return The new layout.
   */
  TVM_DLL static const LayoutDecision ReduceLayout(const LayoutDecision& src_layout,
                                                   const std::vector<size_t>& reduce_axes);
  /*!
   * \brief Permute axes from the Layout
   * \return The new layout.
   */
  TVM_DLL static const LayoutDecision PermuteLayout(const LayoutDecision& src_layout,
                                                    const Array<Integer>& axes);
  TVM_DLL static const LayoutDecision PermuteLayout(const LayoutDecision& src_layout,
                                                    const std::vector<size_t>& axes);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_TRANSFORM_LAYOUT_UTILS_H_
