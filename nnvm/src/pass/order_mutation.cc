/*!
 *  Copyright (c) 2016 by Contributors
 * \file order_mutation.cc
 * \brief Add control flow dependencies between nodes
 *  To correctly order mutation and read to resolve
 *  write after read problem and read after write problems.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

template<typename T>
inline T get_with_default(const std::unordered_map<Node*, T> &map,
                          Node* key,
                          const T& def) {
  auto it = map.find(key);
  if (it != map.end()) return it->second;
  return def;
}

inline bool IsMutate(const std::vector<uint32_t>& mutate_inputs, uint32_t i) {
  return std::binary_search(mutate_inputs.begin(), mutate_inputs.end(), i);
}

Graph OrderMutation(const Graph& src) {
  std::unordered_map<Node*, std::vector<NodeEntry> > version_hist;
  DFSVisit(src.outputs, [&version_hist](const NodePtr& n) {
      for (const NodeEntry& e : n->inputs) {
        if (e.node->is_variable()) {
          if (e.version != 0 && version_hist.count(e.node.get()) == 0) {
            version_hist[e.node.get()] = std::vector<NodeEntry>{};
          }
        }
      }
    });
  // no mutation happens, everything if fine.
  if (version_hist.size() == 0) return src;
  // start preparing for remapping the nodes.
  std::unordered_map<Node*, NodePtr> old_new;
  auto prepare = [&version_hist, &old_new] (const NodePtr& n) {
    static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
    std::vector<uint32_t> mutate_inputs;
    if (!n->is_variable() && fmutate_inputs.count(n->op())) {
      mutate_inputs = fmutate_inputs[n->op()](n->attrs);
    }
    std::sort(mutate_inputs.begin(), mutate_inputs.end());

    bool need_repl = false;
    for (size_t i = 0; i < n->inputs.size(); ++i) {
      const NodeEntry& e = n->inputs[i];
      if (e.node->is_variable()) {
        if (e.version != 0) need_repl = true;
        auto it = version_hist.find(e.node.get());
        if (it != version_hist.end()) {
          std::vector<NodeEntry>& vec = it->second;
          vec.emplace_back(NodeEntry{n, IsMutate(mutate_inputs, i), e.version});
        }
      } else {
        if (old_new.count(e.node.get()) != 0) need_repl = true;
      }
    }
    for (const NodePtr& p : n->control_deps) {
      if (old_new.count(p.get()) != 0) need_repl = true;
    }
    if (need_repl) {
      NodePtr np = Node::Create();
      np->attrs = n->attrs;
      old_new[n.get()] = std::move(np);
    }
  };
  DFSVisit(src.outputs, prepare);
  // comparator of history entry
  auto comparator = [](const NodeEntry& a, const NodeEntry &b) {
    if (a.version < b.version) return true;
    if (a.version > b.version) return false;
    return a.index > b.index;
  };

  for (auto &kv : version_hist) {
    std::sort(kv.second.begin(), kv.second.end(), comparator);
  }
  // copy the nodes, as well as add control deps
  for (auto &kv : old_new) {
    // copy the nodes
    for (const NodeEntry& e : kv.first->inputs) {
      auto it = old_new.find(e.node.get());
      if (it != old_new.end()) {
        kv.second->inputs.emplace_back(NodeEntry{it->second, e.index, e.version});
      } else {
        kv.second->inputs.push_back(e);
      }
    }
    for (const NodePtr& p : kv.first->control_deps) {
      kv.second->control_deps.emplace_back(
          get_with_default(old_new, p.get(), p));
    }
    // add control deps
    static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
    std::vector<uint32_t> mutate_inputs;
    if (fmutate_inputs.count(kv.first->op())) {
      mutate_inputs = fmutate_inputs[kv.first->op()](kv.first->attrs);
    }
    std::sort(mutate_inputs.begin(), mutate_inputs.end());

    for (size_t i = 0; i < kv.first->inputs.size(); ++i) {
      const NodeEntry& e = kv.first->inputs[i];
      if (e.node->is_variable() && version_hist.count(e.node.get()) != 0) {
        std::vector<NodeEntry>& vec = version_hist.at(e.node.get());
        auto it = std::lower_bound(vec.begin(), vec.end(),
                                   NodeEntry{nullptr, 1, e.version},
                                   comparator);
        if (IsMutate(mutate_inputs, i)) {
          int read_dep = 0;
          while (it != vec.begin()) {
            --it;
            if (it->index != 0) break;
            ++read_dep;
            // depend on previous read
            kv.second->control_deps.push_back(
                get_with_default(old_new, it->node.get(), it->node));
          }
          if (read_dep == 0 && it->index != 0) {
            // depend on last write
            kv.second->control_deps.push_back(
                get_with_default(old_new, it->node.get(), it->node));
          }
        } else {
          // depend on last write
          if (it->index != 0) {
            kv.second->control_deps.push_back(
                get_with_default(old_new, it->node.get(), it->node));
          }
        }
      }
    }
  }
  Graph ret;
  for (const NodeEntry &e : src.outputs) {
    ret.outputs.emplace_back(NodeEntry{
        get_with_default(old_new, e.node.get(), e.node), e.index, e.version});
  }
  return ret;
}

NNVM_REGISTER_PASS(OrderMutation)
.describe("Return a new graph that adds control dependencies, "\
          "to order the mutation and reads if mutation exists.")
.set_body(OrderMutation)
.set_change_graph(true);

}  // namespace
}  // namespace pass
}  // namespace nnvm
