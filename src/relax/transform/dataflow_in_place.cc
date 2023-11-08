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
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

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
    } else if (value.as<TupleGetItemNode>()) {
      // Special case: we do not consider a tuple index to be a "use."
      // This is a bit of a hack but allows us to do operations that
      // create tuples to be done in-place (otherwise, any index of the tuple
      // would be considered a use and so the tuple would be live later).
      // Hence we keep the array empty.
      ;
    } else {
      used_vars = AllVars(value);
    }

    for (auto var : used_vars) {
      if (!ret.count(var)) {
        ret[var] = {-1, i};
      }
    }

    if (!ret.count(defined_var)) {
      // if it's an output, then it lives past the end of the block
      if (!defined_var.as<DataflowVarNode>()) {
        ret[defined_var] = {i, block->bindings.size()};
      } else {
        // otherwise, it's live only here
        ret[defined_var] = {i, i};
      }
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
  explicit AliasAnalyzer() : alias_map_(), tuple_map_(), mem_idx_(0) {}

  // alias: map of var to memory locations (we will call these indices and use -1 as an index for
  // "unknown")
  std::pair<std::unordered_map<Var, std::unordered_set<int>, ObjectPtrHash, ObjectPtrEqual>,
            std::unordered_map<int, std::vector<std::unordered_set<int>>>>
  Analyze(const DataflowBlock& block, const Array<Var>& inputs) {
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

    return {alias_map_, tuple_map_};
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

  void update_tuple_components(int tup_idx, const std::unordered_set<int>& insert_idxs) {
    if (tuple_map_.count(tup_idx)) {
      auto tuple_comps = tuple_map_[tup_idx];
      for (size_t i = 0; i < tuple_comps.size(); i++) {
        auto comp_set = tuple_comps[i];

        // if a member is a tuple, update its components as well
        for (int member : comp_set) {
          if (tuple_map_.count(member)) {
            update_tuple_components(member, insert_idxs);
          }
        }

        // update after iterating to avoid iterating over the inserted elements
        tuple_map_[tup_idx][i].insert(insert_idxs.begin(), insert_idxs.end());
      }
    }
  }

  // capture the given index and also its tuple components (including recursively)
  // if they exist
  void add_captured_indices(std::unordered_set<int>* captured_set, int idx) {
    captured_set->insert(idx);
    if (tuple_map_.count(idx)) {
      for (auto comp_set : tuple_map_[idx]) {
        for (auto tup_comp_idx : comp_set) {
          add_captured_indices(captured_set, tup_comp_idx);
        }
      }
    }
  }

  // Conservative extremely pessimistic assumption:
  // assume that the result of a non-op call can be aliased to any argument
  // or that it could be a newly allocated value.
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
    add_captured_indices(&ret, res_idx);

    for (size_t i = (skip_first_arg) ? 1 : 0; i < call_node->args.size(); i++) {
      auto arg = call_node->args[i];
      auto arg_alias_set = get_alias_set(arg, bound_var);
      for (int alias_idx : arg_alias_set) {
        add_captured_indices(&ret, alias_idx);
      }
    }
    // if the result is a tuple, the components can also potentially be aliased to any arg
    // or, in fact, to each other
    update_tuple_components(res_idx, ret);
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
      // if -1 is a member of the tuple set, then we have to assume the result is -1
      if (tuple_set.count(-1)) {
        ret.insert(-1);
      } else {
        // otherwise, consider all members that are tuples of appropriate size and index into them
        // (this is safe because the type system will ensure we're not indexing into a tuple
        // of the wrong size)
        for (int member : tuple_set) {
          if (tuple_map_.count(member) &&
              static_cast<int>(tuple_map_[member].size()) > target_tgi->index) {
            auto member_set = tuple_map_[member][target_tgi->index];
            ret.insert(member_set.begin(), member_set.end());
          }
        }
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
  int mem_idx_;
};

int shape_size(const ShapeExpr& shape) {
  int ret = 1;
  for (auto dim : shape->values) {
    if (auto int_dim = dim.as<IntImmNode>()) {
      ret *= static_cast<int>(int_dim->value);
    } else {
      return -1;
    }
  }
  return ret;
}

std::unordered_set<StructInfo, ObjectPtrHash, ObjectPtrEqual> gather_candidate_sinfo(
    const StructInfo& result_sinfo) {
  if (auto* tensor_info = result_sinfo.as<TensorStructInfoNode>()) {
    // don't consider void dtype (don't know the size at compile time)
    if (tensor_info->dtype.is_void()) {
      return {};
    }
    // don't consider cases where we don't know the shape at compile time
    // TODO(@slyubomirsky): variables might be okay if we use the arithmetic analyzer
    if (auto* shape_node = tensor_info->shape.as<ShapeExprNode>()) {
      for (auto dim : shape_node->values) {
        if (!dim.as<IntImmNode>()) {
          return {};
        }
      }
      return {GetRef<TensorStructInfo>(tensor_info)};
    } else {
      return {};
    }
  } else if (auto* tuple_info = result_sinfo.as<TupleStructInfoNode>()) {
    // we can see if the whole tuple matches or go for any of the components
    std::unordered_set<StructInfo, ObjectPtrHash, ObjectPtrEqual> ret;
    for (auto field : tuple_info->fields) {
      auto field_candidates = gather_candidate_sinfo(field);
      ret.insert(field_candidates.begin(), field_candidates.end());
    }
    // at least one field should be eligible to be done in-place
    if (!ret.empty()) {
      ret.insert(GetRef<StructInfo>(tuple_info));
    }
    return ret;
  } else {
    // don't consider any other types
    return {};
  }
}

std::pair<bool, bool> size_matches(const StructInfo& target_info, const StructInfo& arg_info) {
  if (target_info.as<TensorStructInfoNode>() && arg_info.as<TensorStructInfoNode>()) {
    auto target_tensor = Downcast<TensorStructInfo>(target_info);
    auto arg_tensor = Downcast<TensorStructInfo>(arg_info);
    if (target_tensor->shape.defined() && target_tensor->shape.as<ShapeExprNode>() &&
        arg_tensor->shape.defined() && arg_tensor->shape.as<ShapeExprNode>()) {
      if (target_tensor->dtype != arg_tensor->dtype) {
        return {false, false};
      }
      auto target_shape = Downcast<ShapeExpr>(target_tensor->shape);
      auto arg_shape = Downcast<ShapeExpr>(arg_tensor->shape);
      int target_size = shape_size(target_shape);
      int arg_size = shape_size(arg_shape);
      if (target_size == -1 || arg_size == -1 || target_size < arg_size) {
        return {false, false};
      }
      // exact match: number of dims and each dim matches
      if (target_shape->values.size() == arg_shape->values.size()) {
        for (size_t i = 0; i < target_shape->values.size(); i++) {
          if (!arg_shape->values[i].as<IntImmNode>()) {
            return {false, false};
          }
          if (Downcast<IntImm>(target_shape->values[i])->value !=
              Downcast<IntImm>(arg_shape->values[i])->value) {
            return {true, false};
          }
        }
        return {true, true};
      }
      return {true, false};
    } else {
      return {false, false};
    }
  } else if (target_info.as<TupleStructInfoNode>() && arg_info.as<TupleStructInfoNode>()) {
    auto target_tup = Downcast<TupleStructInfo>(target_info);
    auto arg_tup = Downcast<TupleStructInfo>(arg_info);
    if (target_tup->fields.size() != arg_tup->fields.size()) {
      return {false, false};
    }
    bool all_exact = true;
    for (size_t i = 0; i < target_tup->fields.size(); i++) {
      auto element_match = size_matches(target_tup->fields[i], arg_tup->fields[i]);
      if (!element_match.first) {
        return {false, false};
      }
      if (!element_match.second) {
        all_exact = false;
      }
    }
    return {true, all_exact};
  } else if (target_info.as<PrimStructInfoNode>() && arg_info.as<PrimStructInfoNode>()) {
    return {true, true};
  } else {
    return {false, false};
  }
}

// Given an alias index, check if it's a tuple and gather the sets of aliases for the tuple
// members if so (apply recursively if any of those members are tuples).
// Return false if the alias set contains -1, meaning a reference to an unknown or
// possibly dangerous value (no checking we can do for that).
bool gather_sets_to_check_for_liveness(
    const std::unordered_map<Var, std::unordered_set<int>, ObjectPtrHash, ObjectPtrEqual>&
        alias_sets,
    const std::unordered_map<int, std::vector<std::unordered_set<int>>>& tuple_map,
    std::vector<std::unordered_set<int>>* sets_to_check, int alias_idx) {
  if (tuple_map.count(alias_idx)) {
    for (auto member_set : tuple_map.at(alias_idx)) {
      // contains -1 -> unknown and dangerous, we can short-circuit
      if (member_set.count(-1)) {
        return false;
      }
      sets_to_check->push_back(member_set);

      // if a member can be a tuple, check it recursively
      for (int member : member_set) {
        if (tuple_map.count(member)) {
          if (!gather_sets_to_check_for_liveness(alias_sets, tuple_map, sets_to_check, member)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// check that the target is not live past the index and that no alias of it is live past the
// binding index (if the target is a tuple, check the conditions recursively for the members)
bool df_inplace_conditions_met(
    const std::unordered_map<Var, std::pair<int, int>, ObjectPtrHash, ObjectPtrEqual>& live_ranges,
    const std::unordered_map<Var, std::unordered_set<int>, ObjectPtrHash, ObjectPtrEqual>&
        alias_sets,
    const std::unordered_map<int, std::vector<std::unordered_set<int>>>& tuple_map,
    const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& currently_live,
    const Expr& target, int idx) {
  if (auto* var_node = target.as<VarNode>()) {
    auto current_var = GetRef<Var>(var_node);
    // if the var is live past this point, we can't use it for in-place computations anyway
    if (live_ranges.count(current_var)) {
      auto live_range = live_ranges.at(current_var);
      if (live_range.second > idx) {
        return false;
      }
    }

    // no entry for the current var -> it must be something external and we have to assume the worst
    if (!alias_sets.count(current_var)) {
      return false;
    }
    auto alias_set = alias_sets.at(current_var);
    // -1 -> an external value and we must assume the worst
    if (alias_set.count(-1)) {
      return false;
    }
    std::vector<std::unordered_set<int>> sets_to_check = {alias_set};
    std::unordered_set<int> indices_checked;
    // If a possible alias is a tuple, we will also check for aliases of the members
    // (possibly recursively)
    for (int alias_idx : alias_set) {
      if (!gather_sets_to_check_for_liveness(alias_sets, tuple_map, &sets_to_check, alias_idx)) {
        return false;
      }
    }

    for (Var other_var : currently_live) {
      if (!alias_sets.count(other_var) || !live_ranges.count(other_var)) {
        return false;
      }
      // var is not live past this point => don't need to worry
      if (live_ranges.at(other_var).second <= idx) {
        continue;
      }
      auto other_alias_set = alias_sets.at(other_var);
      for (int alias_idx : other_alias_set) {
        for (auto check_set : sets_to_check) {
          if (check_set.count(alias_idx)) {
            return false;
          }
        }
      }
    }
    return true;
  } else if (auto* tup_node = target.as<TupleNode>()) {
    for (auto field : tup_node->fields) {
      if (!df_inplace_conditions_met(live_ranges, alias_sets, tuple_map, currently_live, field,
                                     idx)) {
        return false;
      }
    }
    return true;
  } else {
    return true;
  }
}

// this is obviously not a complete list
static std::unordered_set<std::string> SUPPORTED_OPS = {"relax.add",      "relax.subtract",
                                                        "relax.multiply", "relax.divide",
                                                        "relax.nn.silu",  "relax.nn.relu"};
bool op_supports_inplace(const Op& op) { return SUPPORTED_OPS.count(op->name); }

// check for in-place eligibility:
//  1. see if there's an arg big enough to hold the result
//  2. see if the arg is live past the call
//  3. see if the arg has an alias that's live past the call
//  if conditions are met, we're good to go
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> find_inplace_opportunities(
    const DataflowBlock& block, const Array<Var>& inputs) {
  auto live_ranges = analyze_liveness(block);
  AliasAnalyzer analyzer;
  auto alias_info = analyzer.Analyze(block, inputs);
  auto alias_sets = alias_info.first;
  auto tuple_map = alias_info.second;

  std::vector<std::vector<int>> size_match_list;
  std::vector<std::vector<int>> exact_match_list;

  // sort the live ranges by starting index
  std::vector<Var> live_order;
  for (auto kv : live_ranges) {
    live_order.push_back(kv.first);
  }
  std::sort(live_order.begin(), live_order.end(),
            [&live_ranges](const Var& var1, const Var& var2) -> bool {
              return live_ranges[var1].first < live_ranges[var2].first;
            });

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> currently_live;
  int last_live = 0;

  for (size_t i = 0; i < block->bindings.size(); i++) {
    // include all vars that are currently live
    for (int j = last_live; j < static_cast<int>(live_order.size()); j++) {
      auto live_var = live_order[j];
      auto live_range = live_ranges[live_var];
      if (live_range.first > static_cast<int>(i)) {
        break;
      }
      currently_live.insert(live_var);
      last_live++;
    }

    // if we reach a binding check the conditions
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

    if (auto* call_node = value.as<CallNode>()) {
      if (auto* op_node = call_node->op.as<OpNode>()) {
        if (!op_supports_inplace(GetRef<Op>(op_node))) {
          continue;
        }

        std::unordered_set<int> candidates;
        std::unordered_set<int> exact_match_candidates;

        auto target_sinfo = gather_candidate_sinfo(GetStructInfo(defined_var));
        // can't be done in-place, ignore
        if (target_sinfo.empty()) {
          continue;
        }

        // Check that at least one argument matches size with the result
        for (size_t j = 0; j < call_node->args.size(); j++) {
          auto arg = call_node->args[j];
          for (auto target : target_sinfo) {
            std::pair<bool, bool> match = size_matches(target, GetStructInfo(arg));
            if (match.first) {
              candidates.insert(static_cast<int>(j));
              if (match.second) {
                exact_match_candidates.insert(static_cast<int>(j));
              }
            }
          }
        }
        if (!candidates.size()) {
          continue;
        }

        // Make sure at least one candidate is not live past this point and does not have an alias
        // live past this point
        std::unordered_set<int> remove_candidates;
        for (auto candidate : candidates) {
          if (!df_inplace_conditions_met(live_ranges, alias_sets, tuple_map, currently_live,
                                         call_node->args[candidate], i)) {
            remove_candidates.insert(candidate);
          }
        }
        // (remove now to avoid modifying the list as we iterate on it)
        for (auto candidate : remove_candidates) {
          candidates.erase(candidate);
        }

        // if we have a candidate, then this can be made in-place. Report the appropriate candidates
        if (!candidates.size()) {
          continue;
        }

        // produce a list of candidates for this index
        std::vector<int> size_match_indices = {static_cast<int>(i)};
        size_match_indices.insert(size_match_indices.end(), candidates.begin(), candidates.end());
        size_match_list.push_back(size_match_indices);

        // also gather up the exact match candidates if there are any
        std::vector<int> exact_match_indices = {static_cast<int>(i)};
        for (auto candidate : candidates) {
          if (exact_match_candidates.count(candidate)) {
            exact_match_indices.push_back(candidate);
          }
        }
        if (exact_match_indices.size() > 1) {
          exact_match_list.push_back(exact_match_indices);
        }
      }
    }

    // remove vars whose range has come to an end
    // (keep a separate set to avoid changing the sit while iterating on it)
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> remove;
    for (auto var : currently_live) {
      auto live_range = live_ranges[var];
      if (live_range.second <= static_cast<int>(i)) {
        remove.insert(var);
      }
    }
    for (auto var : remove) {
      currently_live.erase(var);
    }
  }

  return {size_match_list, exact_match_list};
}

tir::Stmt remap_buffers(const tir::Stmt& stmt, const Map<tir::Buffer, tir::Buffer>& buffer_map) {
  class BufferMapper : public tir::StmtExprMutator {
   public:
    explicit BufferMapper(const Map<tir::Buffer, tir::Buffer>& buffer_map)
        : buffer_map_(buffer_map) {}

    tir::Stmt Remap(const tir::Stmt& stmt) { return VisitStmt(stmt); }

    PrimExpr VisitExpr_(const tir::BufferLoadNode* op) final {
      auto node = Downcast<tir::BufferLoad>(tir::StmtExprMutator::VisitExpr_(op));
      auto* node_cow = node.CopyOnWrite();
      node_cow->buffer = AttemptRemap(node->buffer);
      return node;
    }

    tir::Stmt VisitStmt_(const tir::BufferStoreNode* op) final {
      auto node = Downcast<tir::BufferStore>(tir::StmtExprMutator::VisitStmt_(op));
      auto* node_cow = node.CopyOnWrite();
      node_cow->buffer = AttemptRemap(node->buffer);
      return node;
    }

    tir::Stmt VisitStmt_(const tir::BufferRealizeNode* op) final {
      auto node = Downcast<tir::BufferRealize>(tir::StmtExprMutator::VisitStmt_(op));
      auto* node_cow = node.CopyOnWrite();
      node_cow->buffer = AttemptRemap(node->buffer);
      return node;
    }

    tir::Stmt VisitStmt_(const tir::DeclBufferNode* op) final {
      auto node = Downcast<tir::DeclBuffer>(tir::StmtExprMutator::VisitStmt_(op));
      auto* node_cow = node.CopyOnWrite();
      node_cow->buffer = AttemptRemap(node->buffer);
      return node;
    }

    tir::Stmt VisitStmt_(const tir::BlockNode* op) final {
      auto node = Downcast<tir::Block>(tir::StmtExprMutator::VisitStmt_(op));
      auto* node_cow = node.CopyOnWrite();
      // need the lambdas because class methods are not first-class (how ironic)
      node_cow->alloc_buffers =
          node->alloc_buffers.Map([this](const tir::Buffer& b) { return AttemptRemap(b); });
      node_cow->reads =
          node->reads.Map([this](const tir::BufferRegion& br) { return VisitBufferRegion(br); });
      node_cow->writes =
          node->writes.Map([this](const tir::BufferRegion& br) { return VisitBufferRegion(br); });
      node_cow->match_buffers = node->match_buffers.Map(
          [this](const tir::MatchBufferRegion& mbr) { return VisitMatchBufferRegion(mbr); });
      return node;
    }

   private:
    tir::Buffer AttemptRemap(const tir::Buffer& buffer) {
      if (buffer_map_.count(buffer)) {
        return buffer_map_.at(buffer);
      }
      return buffer;
    }

    tir::BufferRegion VisitBufferRegion(tir::BufferRegion region) {
      auto* region_cow = region.CopyOnWrite();
      region_cow->buffer = AttemptRemap(region_cow->buffer);
      return region;
    }

    tir::MatchBufferRegion VisitMatchBufferRegion(tir::MatchBufferRegion region) {
      auto* region_cow = region.CopyOnWrite();
      region_cow->buffer = AttemptRemap(region_cow->buffer);
      return region;
    }

    const Map<tir::Buffer, tir::Buffer>& buffer_map_;
  };

  BufferMapper mapper(buffer_map);
  auto ret = mapper.Remap(stmt);
  return ret;
}

Call add_inplace_legalization(const BlockBuilder& bb, const Call& call,
                              const Array<Integer>& inplace_indices) {
  static const auto& legalize_map = Op::GetAttrMap<FLegalize>("FLegalize");
  static const auto& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");

  auto op = Downcast<Op>(call->op);
  auto legalized_call = Downcast<Call>(legalize_map[op](bb, call));
  auto* legalized_call_cow = legalized_call.CopyOnWrite();

  // The legalized call should be call_tir. We will replace it with call_tir_inplace
  // and replace the called PrimFunc with an inplace version
  auto legal_op = Downcast<GlobalVar>(legalized_call->args[0]);
  auto inline_legal_op_name = legal_op->name_hint + "_inplace";
  auto mod = bb->GetContextIRModule();

  auto legal_primfunc = Downcast<tir::PrimFunc>(mod->Lookup(legal_op));
  auto* legal_primfunc_cow = legal_primfunc.CopyOnWrite();
  size_t num_outs = inplace_indices.size();
  size_t num_params = legal_primfunc->params.size();

  // the replacement we must make:
  // 1. For each output var, replace its corresponding buffers with the corresponding inplace index
  //    var's buffers
  // 2. For each output var, replace its instances with the corresponding inplace index var
  // 3. Do the same for the *buffer vars* corresponding to the output vars
  // 4. Remove the output vars from the param list and buffer map
  Map<tir::Buffer, tir::Buffer> buffer_subst_map;
  Map<tir::Var, tir::Var> var_subst_map;
  for (size_t i = 0; i < num_outs; i++) {
    // we will substitute output i with the corresponding param indicated by inplace indices
    auto output_var = legal_primfunc->params[num_params - num_outs + i];
    auto inplace_var = legal_primfunc->params[inplace_indices[i].IntValue()];
    var_subst_map.Set(output_var, inplace_var);

    // also do the same with the buffer vars
    auto output_buffer = legal_primfunc->buffer_map.at(output_var);
    auto inplace_buffer = legal_primfunc->buffer_map.at(inplace_var);
    var_subst_map.Set(output_buffer->data, inplace_buffer->data);
    buffer_subst_map.Set(output_buffer, inplace_buffer);
  }

  // apply substitutions
  legal_primfunc_cow->body = remap_buffers(legal_primfunc->body, buffer_subst_map);
  legal_primfunc_cow->body = tir::Substitute(
      legal_primfunc->body, [&var_subst_map](const tir::Var& v) -> Optional<PrimExpr> {
        if (var_subst_map.count(v)) {
          return var_subst_map.at(v);
        }
        return Optional<PrimExpr>();
      });

  // remove the now-unused outputs from the buffer map
  auto buffer_map = legal_primfunc->buffer_map;
  for (size_t i = 0; i < num_outs; i++) {
    buffer_map.erase(legal_primfunc->params[num_params - num_outs + i]);
  }
  legal_primfunc_cow->buffer_map = buffer_map;

  // now get rid of the last num_outputs arguments
  // (couldn't do earlier or else it would have thrown off the indexing)
  legal_primfunc_cow->params = Array<tir::Var>(
      legal_primfunc->params.begin(), legal_primfunc->params.begin() + (num_params - num_outs));

  mod->Remove(legal_op);
  bb->AddFunction(legal_primfunc, inline_legal_op_name);
  // need to update the mod to get the new function
  mod = std::move(bb->GetContextIRModule());
  auto new_gv = mod->GetGlobalVar(inline_legal_op_name);

  // update the call (change the op, update the argument, change the attrs)
  legalized_call_cow->op = call_tir_inplace_op;

  Array<Expr> new_args(legalized_call->args.begin(), legalized_call->args.end());
  new_args.Set(0, new_gv);
  legalized_call_cow->args = new_args;

  ObjectPtr<CallTIRInplaceAttrs> attrs = make_object<CallTIRInplaceAttrs>();
  attrs->inplace_indices = inplace_indices;
  legalized_call_cow->attrs = Attrs(attrs);

  return legalized_call;
}

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

Array<ObjectRef> DataflowAliasAnalysis(const DataflowBlock& block, Array<Var> inputs) {
  AliasAnalyzer analyzer;
  auto res = analyzer.Analyze(block, inputs);
  auto alias_sets = res.first;
  auto tuple_map = res.second;
  Map<Var, Array<Integer>> new_alias_sets;
  Map<Integer, Array<Array<Integer>>> new_tuple_map;
  for (auto kv : alias_sets) {
    Array<Integer> aliases;
    for (auto alias : kv.second) {
      aliases.push_back(alias);
    }
    new_alias_sets.Set(kv.first, aliases);
  }
  for (auto kv : tuple_map) {
    Array<Array<Integer>> elem_aliases;
    for (auto alias_set : kv.second) {
      Array<Integer> dim_aliases;
      for (auto alias : alias_set) {
        dim_aliases.push_back(alias);
      }
      elem_aliases.push_back(dim_aliases);
    }
    new_tuple_map.Set(kv.first, elem_aliases);
  }
  return {new_alias_sets, new_tuple_map};
}

Array<Array<Array<Integer>>> DataflowInPlaceAnalysis(const DataflowBlock& block,
                                                     const Array<Var>& inputs) {
  auto index_lists = relax::find_inplace_opportunities(block, inputs);
  Array<Array<Integer>> size_match_array;
  for (auto indices : index_lists.first) {
    Array<Integer> index_array;
    for (auto index : indices) {
      index_array.push_back(Integer(index));
    }
    size_match_array.push_back(index_array);
  }
  Array<Array<Integer>> exact_match_array;
  for (auto indices : index_lists.second) {
    Array<Integer> index_array;
    for (auto index : indices) {
      index_array.push_back(Integer(index));
    }
    exact_match_array.push_back(index_array);
  }
  return {size_match_array, exact_match_array};
}

TVM_REGISTER_GLOBAL("relax.analysis.DataflowLivenessAnalysis")
    .set_body_typed(DataflowLivenessAnalysis);
TVM_REGISTER_GLOBAL("relax.analysis.DataflowAliasAnalysis").set_body_typed(DataflowAliasAnalysis);
TVM_REGISTER_GLOBAL("relax.analysis.DataflowInPlaceAnalysis")
    .set_body_typed(DataflowInPlaceAnalysis);

// really only for testing (not actually an analysis, will move)
TVM_REGISTER_GLOBAL("relax.analysis.SingleInplaceCall")
    .set_body_typed(relax::add_inplace_legalization);

}  // namespace transform
}  // namespace relax
}  // namespace tvm