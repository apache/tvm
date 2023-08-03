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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/simplify_fixpoint.cc
 * \brief Pass that applies other simplification passes until fixpoint.
 *   Presently, this subsumes the following passes:
 *   * FoldDataflowBlockOutput
 *   * CanonicalizeBIndings
 *   * EliminateCommonSubexpr
 *   * DeadCodeElimination
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

#include "utils.h"

namespace tvm {
namespace relax {

uint64_t Hash(const IRModule& mod) { return SHashHandlerDefault().Hash(mod, true); }

IRModule FixpointSimplification(const IRModule& mod, Array<runtime::String> entry_funcs,
                                bool call_only) {
  // apply passes until it stops changing
  IRModule current_mod = mod;
  transform::Pass cse = transform::EliminateCommonSubexpr(call_only);
  transform::Pass canonicalize_bindings = transform::CanonicalizeBindings();
  transform::Pass dce = transform::DeadCodeElimination(entry_funcs);
  transform::Pass fold_df_output = transform::FoldDataflowBlockOutput();

  while (true) {
    uint64_t last_hash = Hash(current_mod);
    current_mod = std::move(fold_df_output(cse(canonicalize_bindings(dce(current_mod)))));
    uint64_t current_hash = Hash(current_mod);
    if (current_hash == last_hash) {
      break;
    }
  }

  return current_mod;
}

namespace transform {

Pass FixpointSimplification(Array<runtime::String> entry_functions, bool call_only) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    return relax::FixpointSimplification(m, entry_functions, call_only);
  };
  return CreateModulePass(pass_func, 1, "FixpointSimplification", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FixpointSimplification")
    .set_body_typed(FixpointSimplification);

}  // namespace transform
}  // namespace relax
}  // namespace tvm