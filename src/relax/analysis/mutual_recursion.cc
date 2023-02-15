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
 *
 * \file analysis.cc
 *
 * \brief Analysis functions for Relax.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
namespace relax {

/*
 * General approach to detecting mutual recursion:
 *   Suppose we have a dependency graph of global functions,
 *   where function A depends on function B if A contains a call to B
 *   (i.e., an edge A->B means A calls B).
 * 
 *   Note that the call can happen _anywhere_ in the function's body:
 *   All that is important for mutual recursion is that one function
 *   needs the other to be in scope (it needs to know about it) to define
 *   the body. This includes calls that happen inside local function definitions,
 *   branches that may not execute, etc.
 * 
 *   (Note: We will ignore simple recursion and not include the self-edges.)
 *
 *   Then detecting mutual recursion is a problem of cycle detection:
 *   Functions F1, F2, ..., Fn are mutually recursive if there exists a single
 *   directed path that contains all of them.
 *
 *   We aim to find the _largest_ directed cycles in the graph, as there can
 *   be smaller cycles within the larger ones, as in the following example:
 *
 *   A <-> B <-> C
 *   ^     |     ^
 *   |     v     |
 *   |     D     |
 *   |     |     |
 *   v     v     v
 *   E <-> F <-> G
 *
 *   We can detect this condition using DFS:
 *     1. Track node states as unprocessed (never visited),
 *        partially processed (visited but not all children have been searched),
 *        or completely processed (visited and all children have been searched).
 *     2. Pick a node that is unprocessed.
 *     3. Set up our search: During our search, we will keep the found set,
 *        which is a running set of nodes we have found so far that are mutually recursive,
 *        and we will keep track of the path we have taken so far.
 *     4. Start DFS with the node we chose:
 *        a. Set the node to partially processed and add it to the current path.
 *        b. Check the states of each child:
 *           i.   If the child is done, then there is no cycle and nothing further to do.
 *           ii.  If the child is unprocessed, then continue DFS
 *                (return to the start of step 4 with this node and recurse)
 *           iii. If the child is partially processed, then we have hit a cycle.
 *                Backtrack in the current path until you find the child.
 *                (Backtracking is needed in case only *part* of the path forms a cycle.)
 *                Insert each node visited during the backtracking into the found set.
 *        c. Set the current node to done and remove it from the path.
 *     5. The found set constitutes one group of functions that is mutually recursive.
 *        Return to step 2 and repeat until all nodes are marked as done.
 *     6. Return all groups of mutually recursive functions discovered.
 */

class DependencyGatherer : public ExprVisitor {
 public:
  DependencyGatherer(std::string own_name) : own_name_(own_name) {}

  std::unordered_set<std::string> Track(const Function& func) {
    this->VisitExpr(func);
    return deps_;
  }

  void VisitExpr_(const CallNode* call) override {
    auto* gv = call->op.as<GlobalVarNode>();
    if (gv && gv->name_hint != own_name_) {
      deps_.insert(gv->name_hint);
    }
    ExprVisitor::VisitExpr_(call);
  }

 private:
  std::string own_name_;
  std::unordered_set<std::string> deps_;
};

using adjacency_map = std::unordered_map<std::string, std::unordered_set<std::string>>;

adjacency_map GatherDependencyGraph(const IRModule& m) {
  adjacency_map ret;
  for (auto gv_func : m->functions) {
    const relax::FunctionNode* func = gv_func.second.as<relax::FunctionNode>();
    // disregard PrimFuncs and the like
    if (!func) {
      continue;
    }
    std::string name = gv_func.first->name_hint;
    auto deps = DependencyGatherer(name).Track(GetRef<relax::Function>(func));
    ret.insert({name, deps});
  }
  return ret;
}

enum NodeState { kUnprocessed, kPartial, kDone };
using state_map = std::unordered_map<std::string, NodeState>;

void DFSHelper(const adjacency_map& graph, state_map* node_states, const std::string& node,
               std::vector<std::string>* path, std::unordered_set<std::string>* found_so_far) {
  // now processing this node, so add to path and set it to partial
  node_states->extract(node);
  node_states->insert({node, kPartial});
  path->push_back(node);

  auto children = graph.find(node)->second;
  for (std::string child : children) {
    auto state = node_states->find(child)->second;
    if (state == kDone) {
      // not a cycle, move on to next child
    } else if (state == kUnprocessed) {
      // unprocessed child: continue our search
      DFSHelper(graph, node_states, child, path, found_so_far);
    } else {
      // partial -> we have hit a cycle, so backtrack until we get back to the child
      // those nodes into the found set
      for (size_t i = 0; i < path->size(); i++) {
        std::string last_node = path->at(path->size() - i - 1);
        found_so_far->insert(last_node);
        if (last_node == child) {
          break;
        }
      }
    }
  }

  // done with this node
  path->pop_back();
  node_states->extract(node);
  node_states->insert({node, kDone});
}

std::unordered_set<std::string> CheckForMutualRecursion(const adjacency_map& graph,
                                                        state_map* states,
                                                        const std::string& node) {
  std::vector<std::string> path;
  std::unordered_set<std::string> found_so_far;
  DFSHelper(graph, states, node, &path, &found_so_far);
  return found_so_far;
}

tvm::Array<tvm::Array<GlobalVar>> FindMutualRecursion(const IRModule& m) {
  auto graph = GatherDependencyGraph(m);

  state_map states;
  std::vector<std::string> remaining;
  for (auto kv : graph) {
    states[kv.first] = kUnprocessed;
    remaining.push_back(kv.first);
  }

  // search for mutual recursion until all nodes have been proceed
  std::vector<std::unordered_set<std::string>> groups;
  while (!remaining.empty()) {
    std::string current = remaining.back();
    remaining.pop_back();
    if (states[current] == kDone) {
      continue;
    }
    auto group = CheckForMutualRecursion(graph, &states, current);
    if (!group.empty()) {
      groups.push_back(group);
    }
  }

  // convert to expected representation
  tvm::Array<tvm::Array<GlobalVar>> ret;
  for (auto group : groups) {
    tvm::Array<GlobalVar> found;
    for (std::string node : group) {
      found.push_back(m->GetGlobalVar(node));
    }
    ret.push_back(found);
  }
  return ret;
}

TVM_REGISTER_GLOBAL("relax.analysis.find_mutual_recursion").set_body_typed(FindMutualRecursion);

}  // namespace relax
}  // namespace tvm