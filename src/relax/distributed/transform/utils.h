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

#ifndef RELAX_DISTRIBUTED_TRANSFORM_UTILS_H
#define RELAX_DISTRIBUTED_TRANSFORM_UTILS_H

#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/expr_functor.h>
namespace tvm {
namespace relax {
namespace distributed {

/*!
 * \brief Pattern match op to a TIR function and look it up.
 * \return The TIR function, or nullopt if pattern match fails.
 */
inline Optional<tir::PrimFunc> MatchPrimFunc(const IRModule& mod_, const Expr& op) {
  const GlobalVar& global_var = Downcast<GlobalVar>(op);
  // NOTE: as check works for nullptr(returns null)
  Optional<BaseFunc> base_func = mod_->functions.Get(global_var);
  if (auto* pfunc = base_func.as<tir::PrimFuncNode>()) {
    return GetRef<tir::PrimFunc>(pfunc);
  }
  return NullOpt;
}

bool SinfoCompatibleWithDistIR(Array<StructInfo> sinfos);

bool IsDistIRFunc(Function func);

bool IsShardingAnnotatedFunc(Function func);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // RELAX_DISTRIBUTED_TRANSFORM_UTILS_H