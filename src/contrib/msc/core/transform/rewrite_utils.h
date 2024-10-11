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
 * \file src/contrib/msc/core/transform/rewrite_utils.h
 * \brief Common utilities for rewrite.
 */
#ifndef TVM_CONTRIB_MSC_CORE_TRANSFORM_REWRITE_UTILS_H_
#define TVM_CONTRIB_MSC_CORE_TRANSFORM_REWRITE_UTILS_H_

#include <tvm/ir/source_map.h>
#include <tvm/relax/expr.h>

#include <vector>

#include "../../../../relax/transform/utils.h"
#include "../../../../support/scalars.h"
#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using Expr = tvm::RelayExpr;
using namespace tvm::relax;

/*!
 * \brief Utils for Layout.
 */
class RewriteUtils {
 public:
  /*!
   * \brief Emit call with span name.
   * \return The emitted var.
   */
  TVM_DLL static Var ReEmit(BlockBuilder builder, const String& name, const Expr& expr);

  /*!
   * \brief Make and emit a call binding with span.
   * \return The emitted var.
   */
  TVM_DLL static Var MakeCall(BlockBuilder builder, const String& name, Expr op, Array<Expr> args,
                              Attrs attrs = Attrs());

  /*!
   * \brief Make and emit a (shaped)constant with span.
   * \return The constant/reshape.
   */
  TVM_DLL static Expr MakeConstant(BlockBuilder builder, const String& name, double value,
                                   const DataType& dtype, size_t ndim = 0);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_TRANSFORM_REWRITE_UTILS_H_
