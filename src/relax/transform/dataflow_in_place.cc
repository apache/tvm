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

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

std::unordered_map<Var, std::pair<int, int>, ObjectPtrHash, ObjectPtrEqual> analyze_liveness(
    const DataflowBlock& block) {
  std::unordered_map<Var, std::pair<int, int>, ObjectPtrHash, ObjectPtrEqual> ret;
  for (int i = block->bindings.size() - 1; i >= 0; i--) {
    Binding b = block->bindings[i];
    Var defined_var = b->var;
    Expr value;
    if (const auto* var_binding = b.as<VarBindingNode>()) {
      value = var_binding->value;
    } else if (const auto* match_binding = b.as<MatchCastNode>()) {
      value = match_binding->value;
    } else {
      CHECK(false) << "Invalid binding";  // impossible
    }
    Array<Var> used_vars;
    // for a function literal, we consider only the free vars
    // (those captured from the outer scope)
    if (value.as<FunctionNode>()) {
      used_vars = FreeVars(value);
    } else {
      used_vars = AllVars(value);
    }

    for (auto var : used_vars) {
      if (!ret.count(var)) {
        ret[var] = {-1, i};
      }
    }

    if (!ret.count(defined_var)) {
      ret[defined_var] = {i, block->bindings.size()};
    } else {
      // this means the var is used later but we encountered its definition now
      auto last_range = ret[defined_var];
      CHECK_EQ(last_range.first, -1);
      std::pair<int, int> new_range = {i, last_range.second};
      ret[defined_var] = new_range;
    }
  }
  return ret;
}

class AliasAnalyzer {
 public:
  explicit AliasAnalyzer() : alias_map_(), tuple_map_(), captured_by_functions_(), mem_idx_(0) {}

  // alias: map of var to memory locations (we will call these indices and use -1 as an index for
  // "unknown")
  std::unordered_map<Var, std::unordered_set<int>, ObjectPtrHash, ObjectPtrEqual> Analyze(
      const DataflowBlock& block, const Array<Var>& inputs) {
    for (auto input : inputs) {
      int curr_idx = get_fresh_idx();
      alias_map_[input] = {curr_idx};
      if (auto* tup_info = GetStructInfoAs<TupleStructInfoNode>(input)) {
        insert_fresh_tuple(curr_idx, tup_info);
      }
    }

    for (const Binding& binding : block->bindings) {
      Var current_var = binding->var;
      Expr value;
      if (const auto* var_binding = binding.as<VarBindingNode>()) {
        value = var_binding->value;
      } else if (const auto* match_binding = binding.as<MatchCastNode>()) {
        value = match_binding->value;
      } else {
        CHECK(false) << "Invalid binding";  // impossible
      }
      alias_map_[current_var] = get_alias_set(value, current_var);
    }

    return alias_map_;
  }

 private:
  int get_fresh_idx() {
    int ret = mem_idx_;
    mem_idx_++;
    return ret;
  }

  void insert_fresh_tuple(int tup_idx, const TupleStructInfoNode* tup_info) {
    std::vector<std::unordered_set<int>> tuple_set;
    for (int i = 0; i < static_cast<int>(tup_info->fields.size()); i++) {
      int curr_field = get_fresh_idx();
      tuple_set.push_back({curr_field});
      if (auto* nested_tup_info = tup_info->fields[i].as<TupleStructInfoNode>()) {
        insert_fresh_tuple(curr_field, nested_tup_info);
      }
    }
    tuple_map_[tup_idx] = tuple_set;
  }

  // capture the given index and also its tuple components (including recursively)
  // if they exist
  void add_to_captured_set(int idx) {
    captured_by_functions_.insert(idx);
    if (tuple_map_.count(idx)) {
      for (auto comp_set : tuple_map_[idx]) {
        for (auto tup_comp_idx : comp_set) {
          add_to_captured_set(tup_comp_idx);
        }
      }
    }
  }

  // Conservative extremely pessimistic assumption:
  // assume that the result of a non-op call can be aliased to anything
  // ever passed to or returned from any non-op call.
  // For tuples, assume all members are aliased. Yeah, it's bad.
  // (Skip first arg is for handling call_pure_packed, where the first arg is an ExternFunc that we
  // should ignore)
  std::unordered_set<int> handle_mystery_call(const CallNode* call_node, const Var& bound_var,
                                              bool skip_first_arg = false) {
    // the result may or may not be newly allocated
    std::unordered_set<int> ret;
    int res_idx = get_fresh_idx();
    // the result may be a tuple
    if (auto* tup_info_node = GetStructInfoAs<TupleStructInfoNode>(bound_var)) {
      insert_fresh_tuple(res_idx, tup_info_node);
    }
    add_to_captured_set(res_idx);

    for (size_t i = (skip_first_arg) ? 1 : 0; i < call_node->args.size(); i++) {
      auto arg = call_node->args[i];
      auto arg_alias_set = get_alias_set(arg, bound_var);
      for (int alias_idx : arg_alias_set) {
        add_to_captured_set(alias_idx);
      }
    }
    ret.insert(captured_by_functions_.begin(), captured_by_functions_.end());
    return ret;
  }

  std::unordered_set<int> get_alias_set(const Expr& value, const Var& bound_var) {
    std::unordered_set<int> ret;

    // cases for value:
    // constant: it's a fresh index
    // var: look up in alias map (-1 if not present)
    // op call: assume it's fresh (may need to make list of exceptions)
    // tuple: fresh entry in tuple index, recurse to determine indices for values
    // function/packed call: chaos reigns, alias with everything ever passed or returned from func
    //   (if tuple is passed, assume also aliased with all members of the tuple)
    // tuple index: -1 if tuple is not in tuple map, otherwise look up corresponding entry
    // function constant: give them a fresh index (TODO: we can handle in more detail if this is a
    // case we need to support) prim value: fresh index if node: should not happen inside dataflow
    // block
    if (value.as<ConstantNode>() || value.as<PrimValueNode>() || value.as<FunctionNode>()) {
      // TODO(@slyubomirsky): We will probably want special handling for closures
      ret.insert(get_fresh_idx());
    } else if (auto* target_var_node = value.as<VarNode>()) {
      auto target_var = GetRef<Var>(target_var_node);
      if (alias_map_.count(target_var)) {
        ret.insert(alias_map_[target_var].begin(), alias_map_[target_var].end());
      } else {
        ret.insert(-1);
      }
    } else if (auto* target_tuple = value.as<TupleNode>()) {
      // fresh idx but we update the tuple map
      int tup_idx = get_fresh_idx();
      ret.insert(tup_idx);
      std::vector<std::unordered_set<int>> new_tuple_map;
      for (auto field : target_tuple->fields) {
        new_tuple_map.push_back(get_alias_set(field, bound_var));
      }
      tuple_map_[tup_idx] = new_tuple_map;
    } else if (auto* target_tgi = value.as<TupleGetItemNode>()) {
      std::unordered_set<int> tuple_set = get_alias_set(target_tgi->tuple, bound_var);
      // if there's only one possibility for the tuple and it's in the tuple map,
      // index into it
      if (tuple_set.size() == 1) {
        int index = *(tuple_set.begin());
        if (tuple_map_.count(index)) {
          return tuple_map_[index][target_tgi->index];
        } else {
          ret.insert(-1);
        }
      } else {
        ret.insert(-1);
      }
    } else if (auto* call_node = value.as<CallNode>()) {
      if (auto* op_node = call_node->op.as<OpNode>()) {
        // call_pure_packed: treat as non-op call
        if (op_node->name == "relax.call_pure_packed") {
          return handle_mystery_call(call_node, bound_var, true);
        }
        // split: Returns a tuple, treat as allocation
        else if (op_node->name == "relax.split") {
          // tuple is freshly allocated, but also add components to the tuple map
          int tup_idx = get_fresh_idx();
          ret.insert(tup_idx);
          // the LHS (the bound var) will definitely have a tuple struct info
          insert_fresh_tuple(tup_idx, GetStructInfoAs<TupleStructInfoNode>(bound_var));
        }
        // call_tir: can potentially return a tuple
        else if (op_node->name == "relax.call_tir") {
          if (auto* tuple_struct_info = call_node->sinfo_args[0].as<TupleStructInfoNode>()) {
            int tup_idx = get_fresh_idx();
            ret.insert(tup_idx);
            insert_fresh_tuple(tup_idx, tuple_struct_info);
          } else {
            ret.insert(get_fresh_idx());
          }
        }
        // We are assuming most op calls return a single fresh allocation.
        // We may have to track more exceptions
        else {
          ret.insert(get_fresh_idx());
        }
      } else {
        // assume any non-op call can be extremely dangerous and do anything
        return handle_mystery_call(call_node, bound_var);
      }
    }

    return ret;
  }

  std::unordered_map<Var, std::unordered_set<int>, ObjectPtrHash, ObjectPtrEqual> alias_map_;
  std::unordered_map<int, std::vector<std::unordered_set<int>>> tuple_map_;
  std::unordered_set<int> captured_by_functions_;
  int mem_idx_;
};

// check for in-place eligibility:
//  1. see if there's an arg big enough to hold the result
//  2. see if the arg is live past the call
//  3. see if the arg has an alias that's live past the call
//  if conditions are met, we're good to go

// export for testing
namespace transform {

Map<Var, Array<Integer>> DataflowLivenessAnalysis(const DataflowBlock& block) {
  auto liveness_ranges = analyze_liveness(block);
  Map<Var, Array<Integer>> ret;
  for (auto kv : liveness_ranges) {
    ret.Set(kv.first, {kv.second.first, kv.second.second});
  }
  return ret;
}

Map<Var, Array<Integer>> DataflowAliasAnalysis(const DataflowBlock& block, Array<Var> inputs) {
  AliasAnalyzer analyzer;
  auto alias_sets = analyzer.Analyze(block, inputs);
  Map<Var, Array<Integer>> ret;
  for (auto kv : alias_sets) {
    Array<Integer> aliases;
    for (auto alias : kv.second) {
      aliases.push_back(alias);
    }
    ret.Set(kv.first, aliases);
  }
  return ret;
}

TVM_REGISTER_GLOBAL("relax.analysis.DataflowLivenessAnalysis")
    .set_body_typed(DataflowLivenessAnalysis);
TVM_REGISTER_GLOBAL("relax.analysis.DataflowAliasAnalysis").set_body_typed(DataflowAliasAnalysis);
}  // namespace transform
}  // namespace relax
}  // namespace tvm