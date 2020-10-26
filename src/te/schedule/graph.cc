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
 * \file graph.cc
 * \brief Utilities to get information about schedule graph.
 */
#include "graph.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace te {
// key to specific tensor dimension.
struct TensorDimKey {
  Operation op;
  int value_index;
  int dim;
  TensorDimKey() {}
  TensorDimKey(const Tensor& t, int dim) : op(t->op), value_index(t->value_index), dim(dim) {}
  TensorDimKey(const Tensor& t, size_t dim)
      : op(t->op), value_index(t->value_index), dim(static_cast<int>(dim)) {}
  inline bool operator==(const TensorDimKey& other) const {
    return op == other.op && value_index == other.value_index && dim == other.dim;
  }
  inline bool operator!=(const TensorDimKey& other) const { return !operator==(other); }
};
}  // namespace te
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::te::TensorDimKey> {
  std::size_t operator()(const ::tvm::te::TensorDimKey& k) const {
    size_t lhs = ::tvm::ObjectPtrHash()(k.op);
    size_t rhs = static_cast<size_t>(k.value_index) << 16UL | static_cast<size_t>(k.dim);
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std

namespace tvm {
namespace te {

// construct a read graph that gives readers of each operation
// that the root depend on
ReadGraph CreateReadGraph(const Array<Operation>& roots) {
  ReadGraph rmap;
  std::vector<Operation> stack;
  std::unordered_set<const Object*> visited;
  // initialize the roots
  for (Operation op : roots) {
    stack.push_back(op);
    visited.insert(op.get());
  }

  while (!stack.empty()) {
    Operation op = stack.back();
    stack.pop_back();
    Array<Tensor> deps = op->InputTensors();
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

// Do DFS visit to get the subgraph.
// Return if op is inside the subgraph.
bool GetSubGraphByPostDFS_(const Operation& op, const std::unordered_set<const Object*>& boundary,
                           bool include_bounary, std::unordered_map<const Object*, bool>* visited,
                           Array<Operation>* result) {
  if (visited->count(op.get())) {
    return visited->at(op.get());
  }
  if (boundary.count(op.get())) {
    (*visited)[op.get()] = true;
    if (include_bounary) {
      result->push_back(op);
    }
    return true;
  }
  // mark to avoid loop
  // Not necessary for DAG.
  (*visited)[op.get()] = false;
  // check if we can reach boundary.
  bool reach_boundary = false;
  for (Tensor t : op->InputTensors()) {
    if (GetSubGraphByPostDFS_(t->op, boundary, include_bounary, visited, result)) {
      reach_boundary = true;
    }
  }
  (*visited)[op.get()] = reach_boundary;
  if (reach_boundary) {
    result->push_back(op);
  }
  return reach_boundary;
}

Array<Operation> GetSubGraph(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs) {
  Array<Operation> result;
  std::unordered_set<const Object*> boundary;
  for (Tensor t : inputs) {
    boundary.insert(t->op.get());
  }
  std::unordered_map<const Object*, bool> visited;
  for (Tensor t : outputs) {
    GetSubGraphByPostDFS_(t->op, boundary, include_inputs, &visited, &result);
  }
  return result;
}

void PostDFSOrder(const Operation& op, const ReadGraph& g, std::unordered_set<Operation>* visited,
                  Array<Operation>* post_order) {
  if (visited->count(op)) return;
  visited->insert(op);
  for (const auto& t : g.at(op)) {
    PostDFSOrder(t->op, g, visited, post_order);
  }
  post_order->push_back(op);
}

Array<Operation> PostDFSOrder(const Array<Operation>& roots, const ReadGraph& g) {
  std::unordered_set<Operation> visited;
  Array<Operation> post_order;
  for (Operation op : roots) {
    PostDFSOrder(op, g, &visited, &post_order);
  }
  return post_order;
}

FeedGraph CreateFeedGraph(const ReadGraph& g) {
  FeedGraph fg;
  for (auto kv : g) {
    for (Tensor t : kv.second) {
      fg[t].push_back(kv.first);
    }
  }
  return fg;
}

AttachPath CreateAttachPath(Schedule sch) {
  AttachPath ret;
  for (Stage stage : sch->stages) {
    std::unordered_set<const Object*> visited;
    Array<IterVar> path;
    for (Stage s = stage; s.defined();) {
      ICHECK(!visited.count(s.get())) << "Find loop in compute_at attach group";
      visited.insert(s.get());
      Stage spec = s.GetAttachSpec();
      bool start_attach;
      IterVar attach_ivar;
      if (spec->attach_type == kScope) {
        attach_ivar = spec->attach_ivar;
        s = spec->attach_stage;
        start_attach = false;
        ICHECK(attach_ivar.defined());
      } else if (spec->attach_type == kScanUpdate) {
        s = spec->attach_stage;
        start_attach = true;
      } else {
        break;
      }
      ICHECK(s.defined());
      for (size_t i = s->leaf_iter_vars.size(); i != 0; --i) {
        IterVar iv = s->leaf_iter_vars[i - 1];
        if (!start_attach && iv.same_as(attach_ivar)) {
          start_attach = true;
        }
        if (start_attach) path.push_back(iv);
      }
      ICHECK(start_attach) << "Invalid Schedule: cannot find attach point " << attach_ivar
                           << " in the schedule of " << s->op;
    }
    if (!ret.count(stage->op)) {
      ret.Set(stage->op, path);
    }
  }
  return ret;
}

// graph of push reach relation of tensor dimensions
using ReachGraph = std::unordered_map<TensorDimKey, std::vector<TensorDimKey>>;

ReachGraph GetReachGraph(const Array<Operation>& ops) {
  ReachGraph reach;
  std::unordered_set<const Object*> bset;
  for (size_t i = 0; i < ops.size(); ++i) {
    bset.insert(ops[i].get());
  }

  for (Operation op : ops) {
    if (const auto* scan_op = op.as<ScanOpNode>()) {
      const auto& update = scan_op->update;
      const auto& init = scan_op->init;
      for (size_t i = 0; i < update.size(); ++i) {
        Tensor t = op.output(i);
        for (int k = 1; k < static_cast<int>(update[i]->shape.size()); ++k) {
          reach[TensorDimKey(t, k)].emplace_back(TensorDimKey(update[i], k));
          reach[TensorDimKey(t, k)].emplace_back(TensorDimKey(init[i], k));
        }
      }
    } else if (const auto* compute_op = op.as<ComputeOpNode>()) {
      std::unordered_map<const Object*, TensorDimKey> vmap;
      const auto& axis = compute_op->axis;
      Tensor t = op.output(0);
      for (size_t i = 0; i < axis.size(); ++i) {
        vmap[axis[i]->var.get()] = TensorDimKey(t, i);
        reach[TensorDimKey(t, i)] = {};
      }
      auto fvisit = [&vmap, &reach, &bset](const ObjectRef& n) {
        if (auto* pload = n.as<tir::ProducerLoadNode>()) {
          Tensor t = Downcast<Tensor>(pload->producer);
          if (!bset.count(t->op.get())) return;
          for (size_t i = 0; i < pload->indices.size(); ++i) {
            TensorDimKey dkey(t, static_cast<int>(i));
            auto fpush = [&dkey, &vmap, &reach](const ObjectRef& node) {
              const VarNode* v = node.as<VarNode>();
              auto it = vmap.find(v);
              if (it != vmap.end()) {
                reach[it->second].push_back(dkey);
              }
            };
            tir::PostOrderVisit(pload->indices[i], fpush);
          }
        }
      };
      for (auto& e : compute_op->body) {
        tir::PostOrderVisit(e, fvisit);
      }
    }
  }
  return reach;
}

Array<Operation> ScanGetBody(const Operation& scan_op) {
  const ScanOpNode* scan = scan_op.as<ScanOpNode>();
  // Get the body.
  Array<Tensor> inputs;
  for (Tensor t : scan->state_placeholder) {
    inputs.push_back(t);
  }
  for (Tensor t : scan->inputs) {
    inputs.push_back(t);
  }
  return GetSubGraph(scan->update, inputs, false);
}

Map<IterVar, PrimExpr> ScanFixPointAnalysis(const Operation& scan_op) {
  const ScanOpNode* scan = scan_op.as<ScanOpNode>();
  Array<Operation> body = ScanGetBody(scan_op);

  std::unordered_map<TensorDimKey, const Object*> exact_reach;
  std::unordered_set<const Object*> fail_set;

  for (size_t i = 0, sp_idx = 0; i < scan->update.size(); ++i) {
    for (size_t k = 1; k < scan->update[i]->shape.size(); ++k, ++sp_idx) {
      TensorDimKey key(scan->state_placeholder[i], k);
      exact_reach[key] = scan->spatial_axis_[sp_idx].get();
    }
  }
  // merge exact reach
  auto f_merge_key = [&exact_reach, &fail_set](const TensorDimKey& dst, const TensorDimKey& src) {
    auto sit = exact_reach.find(src);
    if (sit == exact_reach.end()) return;
    auto dit = exact_reach.find(dst);
    if (dit == exact_reach.end()) {
      exact_reach[dst] = sit->second;
    } else {
      if (dit->second != sit->second) {
        fail_set.insert(dit->second);
        fail_set.insert(sit->second);
      }
    }
  };
  // prop exact reach back.
  for (size_t i = 0; i < body.size(); ++i) {
    const Operation& op = body[i];
    if (const auto* scan_op = op.as<ScanOpNode>()) {
      const auto& update = scan_op->update;
      const auto& init = scan_op->init;
      for (size_t i = 0; i < update.size(); ++i) {
        Tensor t = op.output(i);
        for (size_t k = 1; k < update[i]->shape.size(); ++k) {
          f_merge_key(TensorDimKey(t, k), TensorDimKey(update[i], k));
          f_merge_key(TensorDimKey(t, k), TensorDimKey(init[i], k));
        }
      }
    } else if (const auto* compute_op = op.as<ComputeOpNode>()) {
      std::unordered_map<const Object*, std::vector<TensorDimKey>> vmap;
      const auto& axis = compute_op->axis;
      for (size_t i = 0; i < axis.size(); ++i) {
        std::vector<TensorDimKey> keys;
        for (int j = 0; j < op->num_outputs(); ++j) {
          keys.emplace_back(op.output(j), i);
        }
        vmap[axis[i]->var.get()] = std::move(keys);
      }
      auto fvisit = [&vmap, &f_merge_key, &exact_reach, &fail_set](const ObjectRef& n) {
        if (auto* pload = n.as<tir::ProducerLoadNode>()) {
          Tensor t = Downcast<Tensor>(pload->producer);
          for (size_t i = 0; i < pload->indices.size(); ++i) {
            auto it = vmap.find(pload->indices[i].get());
            TensorDimKey src(t, static_cast<int>(i));
            if (it != vmap.end()) {
              const std::vector<TensorDimKey>& keys = it->second;
              for (const auto& key : keys) {
                f_merge_key(key, src);
              }
            } else {
              if (exact_reach.count(src)) {
                fail_set.insert(exact_reach.at(src));
              }
            }
          }
        }
      };
      for (auto& e : compute_op->body) {
        tir::PostOrderVisit(e, fvisit);
      }
    }
  }
  ReachGraph reach;
  Map<IterVar, PrimExpr> ret;
  std::unordered_set<TensorDimKey> place_holder_ref;
  for (size_t i = 0; i < scan->state_placeholder.size(); ++i) {
    for (size_t k = 0; k < scan->state_placeholder[i]->shape.size(); ++k) {
      place_holder_ref.insert(TensorDimKey(scan->state_placeholder[i], k));
    }
  }

  for (size_t i = 0, sp_idx = 0; i < scan->update.size(); ++i) {
    for (size_t k = 1; k < scan->update[i]->shape.size(); ++k, ++sp_idx) {
      TensorDimKey key(scan->update[i], k);
      TensorDimKey target(scan->state_placeholder[i], k);
      IterVar sp_iv = scan->spatial_axis_[sp_idx];
      if (fail_set.count(sp_iv.get()) || !exact_reach.count(key) ||
          exact_reach.at(key) != sp_iv.get()) {
        ret.Set(sp_iv, make_const(DataType::Int(32), 0));
      } else {
        // now we proved exact match, need to prove no interference with other graph.
        if (reach.size() == 0) reach = GetReachGraph(body);
        // do a DFS
        std::unordered_set<TensorDimKey> visited;
        std::vector<TensorDimKey> stack{key};
        visited.insert(key);
        while (!stack.empty()) {
          TensorDimKey k = stack.back();
          if (k != target && place_holder_ref.count(k)) break;
          stack.pop_back();
          if (!reach.count(k)) {
            LOG(FATAL) << "cannot find reach of " << k.op << "-" << k.dim;
          }

          for (TensorDimKey kk : reach.at(k)) {
            if (visited.count(kk)) {
              continue;
            }
            visited.insert(kk);
            stack.push_back(kk);
          }
        }
        if (!stack.empty()) {
          // failed the prove.
          ret.Set(sp_iv, make_const(DataType::Int(32), 0));
        } else {
          ret.Set(sp_iv, make_const(DataType::Int(32), 1));
        }
      }
    }
  }
  return ret;
}

TVM_REGISTER_GLOBAL("schedule.CreateReadGraph").set_body_typed(CreateReadGraph);

TVM_REGISTER_GLOBAL("schedule.PostDFSOrder")
    .set_body_typed([](const Array<Operation>& roots, const ReadGraph& g) {
      return PostDFSOrder(roots, g);
    });

TVM_REGISTER_GLOBAL("schedule.CreateAttachPath").set_body_typed(CreateAttachPath);

TVM_REGISTER_GLOBAL("schedule.ScanGetBody").set_body_typed(ScanGetBody);

TVM_REGISTER_GLOBAL("schedule.ScanFixPointAnalysis").set_body_typed(ScanFixPointAnalysis);

}  // namespace te
}  // namespace tvm
