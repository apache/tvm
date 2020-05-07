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
 * \file ad_util.cc
 * \brief Utility for tensor-level auto-differentiation.
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <string>
#include "ad_util.h"

namespace tvm {
namespace te {

std::pair<Array<IterVar>, Map<Var, PrimExpr>> CloneIterVars(const Array<IterVar>& vars) {
  Array<IterVar> new_vars;
  Map<Var, PrimExpr> vmap;
  for (const IterVar& iv : vars) {
    IterVar new_v =
      IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""),
          iv->iter_type, iv->thread_tag);
    new_vars.push_back(new_v);
    vmap.Set(iv->var, new_v->var);
  }
  return std::make_pair(std::move(new_vars), std::move(vmap));
}

PrimExpr CloneReduction(const PrimExpr& expr) {
  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    Array<IterVar> new_axis;
    Map<Var, PrimExpr> vmap;
    std::tie(new_axis, vmap) = CloneIterVars(red->axis);

    Array<PrimExpr> src_with_newaxis;
    for (const auto& src : red->source) {
      src_with_newaxis.push_back(tir::Substitute(src, vmap));
    }

    return ReduceNode::make(red->combiner, src_with_newaxis,
        new_axis, tir::Substitute(red->condition, vmap), red->value_index);
  } else {
    return expr;
  }
}

}  // namespace te
}  // namespace tvm
