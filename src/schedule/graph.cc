/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.cc
 * \brief Utilities to get information about schedule graph.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>
#include "./graph.h"

namespace tvm {
namespace schedule {

// construct a read graph that gives readers of each operation
// that the root depend on
ReadGraph CreateReadGraph(const Array<Operation>& roots) {
  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const Node*> visited;
  // initialize the roots
  for (Operation op : roots) {
    stack.push_back(op);
    visited.insert(op.get());
  }

  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    Array<Tensor> deps;
    if (op.as<ComputeOpNode>()) {
      auto fvisit = [&deps, &visited, &stack](const NodeRef& n) {
        auto *call = n.as<ir::Call>();
        if (call != nullptr && call->func.defined()) {
          Operation call_op(call->func.node_);
          deps.push_back(call_op.output(call->value_index));
        }
      };
      ir::PostOrderVisit(op.as<ComputeOpNode>()->body, fvisit);
    } else if (op.as<ScanOpNode>()) {
      const ScanOpNode* scan = op.as<ScanOpNode>();
      for (Tensor t : scan->init) {
        deps.push_back(t);
      }
      for (Tensor t : scan->update) {
        deps.push_back(t);
      }
    } else if (op.as<PlaceholderOpNode>()) {
    } else {
      LOG(FATAL) << "unknown Operation" << op->type_key();
    }
    rmap.Set(op, deps);
    for (Tensor t : deps) {
      if (t->op.defined() && visited.count(t->op.get()) == 0) {
        visited.insert(t->op.get());
        stack.push_back(t->op);
      }
    }
  }
  return rmap;
}


void PostDFSOrder(const Operation& op,
                  const ReadGraph& g,
                  std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order) {
  if (visited->count(op)) return;
  visited->insert(op);
  for (const auto& t : g.at(op)) {
    PostDFSOrder(t->op, g, visited, post_order);
  }
  post_order->push_back(op);
}

Array<Operation> PostDFSOrder(
    const Array<Operation>& roots,
    const ReadGraph& g) {
  std::unordered_set<Operation> visited;
  Array<Operation> post_order;
  for (Operation op : roots) {
    PostDFSOrder(op, g, &visited, &post_order);
  }
  return post_order;
}

}  // namespace schedule
}  // namespace tvm
