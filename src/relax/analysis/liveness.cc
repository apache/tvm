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
 * \file tvm/relax/analysis/liveness.cc
 * \brief Implementation of liveness analysis
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {

// just sets of vars. the bool value is unnecessary
using Domain = Map<Var, Bool>;

Domain transfer_func(const GraphBinding& binding, const ObjectRef& input) {
  Domain in_domain = Downcast<Domain>(input);
  Domain new_domain(in_domain);

  // 1. If a var that appears in the RHS of the binding, add it (it's live)
  // 2. Remove the bound var (it is not live prior to being bound)
  Array<Var> vars_used;
  Optional<Var> var_bound;
  if (binding->kind == BindingNodeKind::kSeqBody) {
    vars_used = AllVars(binding->seq->body);
  } else if (binding->kind == BindingNodeKind::kIfCond) {
    Binding b = binding->seq->blocks[binding->block_idx]->bindings[binding->binding_idx];
    Expr cond = Downcast<If>(GetBoundValue(b))->cond;
    vars_used = AllVars(cond);
  } else if (binding->kind == BindingNodeKind::kIfMerge) {
    // no vars are used in the merge
    vars_used = {};
    // define the merge var
    var_bound = binding->seq->blocks[binding->block_idx]->bindings[binding->binding_idx]->var;
  } else {
    // the ordinary binding case
    Binding b = binding->seq->blocks[binding->block_idx]->bindings[binding->binding_idx];
    Expr bound_value = GetBoundValue(b);
    // special case: if the RHS is a function literal, we only care about the free vars
    // (those captured by the closure)
    if (bound_value.as<FunctionNode>()) {
      vars_used = FreeVars(bound_value);
    } else {
      vars_used = AllVars(bound_value);
    }
    var_bound = b->var;
  }

  for (auto var : vars_used) {
    new_domain.Set(var, Bool(true));
  }

  // the var bound is killed
  if (var_bound.defined()) {
    new_domain.erase(var_bound.value());
  }

  // technically, we could kill the args too,
  // but they are not actually *bound* at the first binding

  return new_domain;
}

// simply combine sets of live vars to merge
Domain merge_func(const ObjectRef& domain1, const ObjectRef& domain2) {
  Domain merged;
  for (auto kv : Downcast<Domain>(domain1)) {
    merged.Set(kv.first, kv.second);
  }
  for (auto kv : Downcast<Domain>(domain2)) {
    merged.Set(kv.first, kv.second);
  }
  return merged;
}

Array<Array<Var>> LivenessAnalysis(const Function& func) {
  // initial domain is empty
  Domain init_domain;
  ControlFlowGraph cfg = ExtractCFG(func);
  std::pair<ObjectRef, ObjectRef> results =
      DataflowAnalysis(cfg, init_domain, transfer_func, merge_func, false);

  // we will return the input map but convert the maps into arrays for simplicity
  Array<Domain> in_map = Downcast<Array<Domain>>(results.first);

  Array<Array<Var>> ret;
  for (const Domain& d : in_map) {
    Array<Var> arr;
    for (auto kv : d) {
      arr.push_back(kv.first);
    }
    ret.push_back(arr);
  }
  return ret;
}

TVM_REGISTER_GLOBAL("relax.analysis.LivenessAnalysis").set_body_typed(LivenessAnalysis);

}  // namespace relax
}  // namespace tvm
