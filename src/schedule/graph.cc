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
ReadGraph CreateReadGraph(Operation root) {
  std::unordered_map<Operation, std::vector<Tensor> > rmap;
  rmap[root] = {};
  std::vector<Operation> stack{root};
  while (!stack.empty()) {
    Operation r = stack.back();
    stack.pop_back();
    auto& vec = rmap.at(r);
    if (r.as<ComputeOpNode>()) {
      auto fvisit = [&vec, &rmap, &stack](const NodeRef& n) {
        auto *call = n.as<ir::Call>();
        if (call != nullptr && call->func.defined()) {
          Tensor t(call->func.node_);
          vec.push_back(t);
          if (t->op.defined() && rmap.count(t->op) == 0) {
            rmap[t->op] = {}; stack.push_back(t->op);
          }
        }
      };
      ir::PostOrderVisit(r.as<ComputeOpNode>()->body, fvisit);
    } else {
      LOG(FATAL) << "unknown operation mode";
    }
  }
  return rmap;
}


void PostDFSOrder(const Operation& op,
                    const ReadGraph& g,
                    std::unordered_set<Operation>* visited,
                    std::vector<Operation>* post_order) {
  visited->insert(op);
  for (const auto& t : g.at(op)) {
    if (t->op.defined() && !visited->count(t->op)) {
      PostDFSOrder(t->op, g, visited, post_order);
    }
  }
  post_order->push_back(op);
}

std::vector<Operation> PostDFSOrder(
    const Operation& root, const ReadGraph& g) {
  std::unordered_set<Operation> visited;
  std::vector<Operation> post_order;
  PostDFSOrder(root, g, &visited, &post_order);
  return post_order;
}

}  // namespace schedule
}  // namespace tvm
