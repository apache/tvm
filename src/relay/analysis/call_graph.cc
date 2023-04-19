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
 * \file src/relay/analysis/call_graph.cc
 * \brief Implementation of APIs to handle the call graph of a Relay module.
 */

#include "call_graph.h"

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "../op/call/call.h"

namespace tvm {
namespace relay {

CallGraph::CallGraph(IRModule module) {
  auto n = make_object<CallGraphNode>();
  n->module = std::move(module);
  auto gvar_funcs = n->module->functions;
  for (const auto& it : gvar_funcs) {
    if (auto func = it.second.as<Function>()) {
      // Add the global function to gradually build up the call graph.
      n->AddToCallGraph(it.first, func.value());
    }
  }
  data_ = std::move(n);
}

void CallGraphNode::AddToCallGraph(const GlobalVar& gv, const Function& func) {
  ICHECK(func.defined() && gv.defined());
  // Add the current global function as an entry to the call grpah.
  CallGraphEntry* cg_node = LookupGlobalVar(gv);

  // Only GlobalVar nodes need to be handled in a function. It indicates that
  // the global function of a callee is called by the function that is being
  // processed. An edge will be added from the current global function, cg_node,
  // to the node that contains the found callee GlobalVarNode.
  //
  // This is the major overhead for constructing a call graph because the
  // post-order visitor will visit each AST node of the current function to
  // figure out the dependencies between functions.
  PostOrderVisit(func, [&](const Expr& expr) {
    // TODO(mbs): Cleanup shapes functions.
    if (const auto* call_node = expr.as<CallNode>()) {
      CallLoweredProps props = GetCallLoweredProps(call_node);
      if (props.lowered_func.defined() && props.attrs.metadata.count("prim_shape_fn_var")) {
        // We are implicitly calling the shape function *in addition to* the call target.
        CallGraphEntry* callee_cg_node =
            LookupGlobalVar(Downcast<GlobalVar>(props.attrs.metadata["prim_shape_fn_var"]));
        cg_node->AddCalledGlobal(callee_cg_node);
      }
    } else if (auto callee = expr.as<GlobalVar>()) {
      CallGraphEntry* callee_cg_node = LookupGlobalVar(callee.value());
      cg_node->AddCalledGlobal(callee_cg_node);
    }
  });
}

const CallGraphEntry* CallGraphNode::operator[](const GlobalVar& gv) const {
  const_iterator cit = call_graph_.find(gv);
  ICHECK(cit != call_graph_.end())
      << "GlobalVar " << gv->name_hint << " not found in the call graph!";
  return cit->second.get();
}

CallGraphEntry* CallGraphNode::operator[](const GlobalVar& gv) {
  const_iterator cit = call_graph_.find(gv);
  ICHECK(cit != call_graph_.end())
      << "GlobalVar " << gv->name_hint << " not found in the call graph!";
  return cit->second.get();
}

BaseFunc CallGraphNode::GetGlobalFunction(const GlobalVar& var) const {
  ICHECK(module->ContainGlobalVar(var->name_hint))
      << "GlobalVar " << var->name_hint << " not found in the current ir module";
  return module->Lookup(var->name_hint);
}

CallGraphEntry* CallGraphNode::LookupGlobalVar(const GlobalVar& gv) {
  ICHECK(gv.defined());

  // This inserts an element to the call graph if it is not there yet.
  auto& call_graph_node = call_graph_[gv];
  if (call_graph_node) return call_graph_node.get();

  // Create the node for the inserted entry.
  call_graph_node = std::make_unique<CallGraphEntry>(gv);
  return call_graph_node.get();
}

void CallGraphNode::Print(std::ostream& os) const {
  // Print the call graph in the topological order.
  std::vector<CallGraphEntry*> nodes = TopologicalOrder();
  for (const auto* cgn : nodes) {
    cgn->Print(os);
  }
}

GlobalVar CallGraphNode::RemoveGlobalVarFromModule(CallGraphEntry* cg_node,
                                                   bool update_call_graph) {
  ICHECK(cg_node->empty() || (cg_node->IsRecursive() && cg_node->size() == 1))
      << "Cannot remove global var " << cg_node->GetNameHint()
      << " from call graph, because it still calls " << cg_node->size()
      << " other global functions";

  if (update_call_graph) {
    // Update the call graph by removing all edges that point to the node
    // `cg_node`.
    for (auto& it : *this) {
      it.second->RemoveAllCallTo(cg_node);
    }
  }
  GlobalVar gv = cg_node->GetGlobalVar();
  call_graph_.erase(gv);
  // Update the IR module.
  module->Remove(gv);
  return gv;
}

std::vector<CallGraphEntry*> CallGraphNode::GetEntryGlobals() const {
  std::vector<CallGraphEntry*> ret;
  // An entry function in Relay is a function that never called by other
  // functions or only called by itself.
  for (const auto& it : *this) {
    if (it.second->GetRefCount() == 0 || it.second->IsRecursiveEntry()) {
      ret.push_back(it.second.get());
    }
  }
  return ret;
}

std::vector<CallGraphEntry*> CallGraphNode::TopologicalOrder() const {
  std::vector<CallGraphEntry*> ret;
  // Collect all entry nodes.
  std::vector<CallGraphEntry*> entries = GetEntryGlobals();
  CallGraphEntry::CallGraphEntrySet visited;

  for (const auto& it : entries) {
    // Keep tracking the nodes that have been visited.
    auto topo = it->TopologicalOrder(&visited);
    // Prepend the collected items. The intermediate nodes that are shared by
    // multiple entries are guaranteed to be collected when visiting the
    // previous entries. Therefore, topological order remains.
    ret.insert(ret.begin(), topo.begin(), topo.end());
  }

  // Find out the missing global functions if there are any to help debugging.
  if (ret.size() != module->functions.size()) {
    for (auto it : module->functions) {
      if (visited.find((*this)[it.first]) == visited.end()) {
        LOG(WARNING) << "Missing global:" << it.first->name_hint
                     << " with # refs = " << (*this)[it.first]->GetRefCount();
      }
    }
    LOG(FATAL) << "Expected " << module->functions.size() << " globals, but received "
               << ret.size();
  }

  return ret;
}

// BSF traversal is used to collect the nodes in a CallGraphEntry. The nodes
// that are visited by previous CallGraphEntry entries can be memoized. This
// helps us to make sure no entry will be visited multiple times when collecting
// the nodes for an entire call graph.
std::vector<CallGraphEntry*> CallGraphEntry::TopologicalOrder(CallGraphEntrySet* visited) const {
  std::vector<CallGraphEntry*> ret;
  std::vector<CallGraphEntry*> current_nodes;
  if (visited->find(this) == visited->end()) {
    visited->emplace(this);
    current_nodes.emplace_back(const_cast<CallGraphEntry*>(this));
  }

  std::vector<CallGraphEntry*> next_nodes;
  while (!current_nodes.empty()) {
    for (const auto& node : current_nodes) {
      ret.push_back(node);
      // Iterate through the called entries.
      for (auto git = node->begin(); git != node->end(); ++git) {
        if (visited->find(git->second) == visited->end()) {
          next_nodes.push_back(git->second);
          visited->emplace(git->second);
        }
      }
    }
    // Update the current level and clean the next level.
    current_nodes = next_nodes;
    next_nodes.clear();
  }
  return ret;
}

void CallGraphEntry::CleanCallGraphEntries() {
  while (!called_globals_.empty()) {
    // Decrement the reference counter
    called_globals_.back().second->DecRef();
    called_globals_.pop_back();
  }
}

inline void CallGraphEntry::AddCalledGlobal(CallGraphEntry* cg_node) {
  called_globals_.emplace_back(global_, cg_node);
  // Increment the reference to indicate that another call site is found for
  // the callee in `cg_node`.
  cg_node->IncRef();
  // Mark the global function as recursive if it calls itself.
  if (global_ == cg_node->GetGlobalVar()) {
    cg_node->is_recursive_ = true;
  }
}

// Remove an edge from the current global function to the callee.
void CallGraphEntry::RemoveCallTo(const GlobalVar& callee) {
  for (auto it = begin();; ++it) {
    ICHECK(it != end()) << "Cannot find global function " << callee->name_hint << " to remove!";
    if (it->second->GetGlobalVar() == callee) {
      // Only remove one occurrence of the call site.
      it->second->DecRef();
      *it = called_globals_.back();
      called_globals_.pop_back();
      return;
    }
  }
}

// Remove all edges from the current global function to the callee.
void CallGraphEntry::RemoveAllCallTo(CallGraphEntry* callee) {
  for (uint32_t i = 0, e = size(); i != e;) {
    if (called_globals_[i].second == callee) {
      callee->DecRef();
      called_globals_[i] = called_globals_.back();
      called_globals_.pop_back();
      --e;
    } else {
      ++i;
    }
  }
  // Make sure all references to the callee are removed.
  ICHECK_EQ(callee->GetRefCount(), 0U)
      << "All references to " << callee->GetNameHint() << " should have been removed";
}

void CallGraphEntry::Print(std::ostream& os) const {
  if (!global_.defined()) {
    os << "GlobalVar is not defined\n";
    return;
  }

  os << "Call graph node: " << global_->name_hint;
  os << " at: " << this << ",  #refs = " << GetRefCount() << "\n";

  for (const auto& it : *this) {
    os << "  call site: <" << it.first->name_hint << "> calls ";
    os << it.second->GetNameHint() << "\n";
  }
  os << "\n";
}

std::ostream& operator<<(std::ostream& os, const CallGraph& cg) {
  cg->Print(os);
  return os;
}

std::ostream& operator<<(std::ostream& os, const CallGraphEntry& cgn) {
  cgn.Print(os);
  return os;
}

TVM_REGISTER_NODE_TYPE(CallGraphNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CallGraphNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CallGraphNode*>(ref.get());
      ICHECK(node);
      p->stream << "CallGraph: \n" << GetRef<CallGraph>(node);
    });

TVM_REGISTER_GLOBAL("relay.analysis.CallGraph").set_body_typed([](IRModule module) {
  return CallGraph(module);
});

TVM_REGISTER_GLOBAL("relay.analysis.PrintCallGraph").set_body_typed([](CallGraph call_graph) {
  std::stringstream ss;
  ss << call_graph;
  return ss.str();
});

TVM_REGISTER_GLOBAL("relay.analysis.GetModule").set_body_typed([](CallGraph call_graph) {
  return call_graph->module;
});

TVM_REGISTER_GLOBAL("relay.analysis.PrintCallGraphGlobalVar")
    .set_body_typed([](CallGraph call_graph, GlobalVar var) {
      const auto* entry_node = call_graph[var];
      std::stringstream ss;
      ss << *entry_node;
      return ss.str();
    });

TVM_REGISTER_GLOBAL("relay.analysis.GetRefCountGlobalVar")
    .set_body_typed([](CallGraph call_graph, GlobalVar var) {
      const auto* entry_node = call_graph[var];
      return static_cast<int>(entry_node->GetRefCount());
    });

TVM_REGISTER_GLOBAL("relay.analysis.GetGlobalVarCallCount")
    .set_body_typed([](CallGraph call_graph, GlobalVar var) {
      const auto* entry_node = call_graph[var];
      return static_cast<int>(entry_node->size());
    });

TVM_REGISTER_GLOBAL("relay.analysis.IsRecursive")
    .set_body_typed([](CallGraph call_graph, GlobalVar var) {
      const auto* entry_node = call_graph[var];
      return entry_node->IsRecursive();
    });

}  // namespace relay
}  // namespace tvm
