/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.cc
 * \brief Utilities to get information about schedule graph.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>
#include "./int_set.h"
#include "./graph.h"

namespace tvm {
namespace schedule {

// construct a read graph that gives readers of each operation
// that the root depend on
ReadGraph CreateReadGraph(const Operation& root) {
  ReadGraph rmap;
  std::vector<Operation> stack{root};
  std::unordered_set<const Node*> visited{root.get()};

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
          if (call_op.defined() && visited.count(call_op.get()) == 0) {
            visited.insert(call_op.get());
            stack.push_back(call_op);
          }
        }
      };
      ir::PostOrderVisit(op.as<ComputeOpNode>()->body, fvisit);
      rmap.Set(op, deps);
    } else {
      if (!op.as<PlaceholderOpNode>()) {
        LOG(FATAL) << "unknown Operation" << op->type_key();
      }
    }
  }
  return rmap;
}


void PostDFSOrder(const Operation& op,
                  const ReadGraph& g,
                  std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order) {
  visited->insert(op);
  for (const auto& t : g.at(op)) {
    if (!t->op.as<PlaceholderOpNode>() && !visited->count(t->op)) {
      PostDFSOrder(t->op, g, visited, post_order);
    }
  }
  post_order->push_back(op);
}

Array<Operation> PostDFSOrder(
    const Operation& root, const ReadGraph& g) {
  std::unordered_set<Operation> visited;
  Array<Operation> post_order;
  PostDFSOrder(root, g, &visited, &post_order);
  return post_order;
}

}  // namespace schedule
}  // namespace tvm
