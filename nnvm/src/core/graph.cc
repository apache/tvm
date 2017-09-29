/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph_attr_types.cc
 * \brief Graph node data structure.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <limits>

namespace nnvm {

const IndexedGraph& Graph::indexed_graph() const {
  if (indexed_graph_ == nullptr) {
    indexed_graph_.reset(new IndexedGraph(*this));
  }
  return *indexed_graph_;
}

// implement constructor from graph
IndexedGraph::IndexedGraph(const Graph &g) {
  entry_rptr_.push_back(0);
  std::vector<size_t> inputs_rptr{0}, control_rptr{0};

  DFSVisit(g.outputs, [this, &inputs_rptr, &control_rptr]
             (const NodePtr& n) {
      CHECK_LT(nodes_.size(), std::numeric_limits<uint32_t>::max());
      uint32_t nid = static_cast<uint32_t>(nodes_.size());
      // nodes_
      IndexedGraph::Node new_node;
      new_node.source = n.get();
      new_node.weak_ref = n;
      nodes_.emplace_back(std::move(new_node));
      // arg_nodes_
      if (n->is_variable()) {
        input_nodes_.push_back(nid);
      }
      // node2index_
      node2index_[n.get()] = nid;
      // entry rptr
      entry_rptr_.push_back(entry_rptr_.back() + n->num_outputs());
      // input entries
      for (const auto& e : n->inputs) {
        auto it = node2index_.find(e.node.get());
        CHECK(it != node2index_.end() && it->first == e.node.get());
        input_entries_.emplace_back(NodeEntry{it->second, e.index, e.version});
      }
      inputs_rptr.push_back(input_entries_.size());
      // control deps
      for (const auto& nptr : n->control_deps) {
        auto it = node2index_.find(nptr.get());
        CHECK(it != node2index_.end() && it->first == nptr.get());
        control_deps_.push_back(it->second);
      }
      control_rptr.push_back(control_deps_.size());
  });

  for (const auto& e : g.outputs) {
    outputs_.emplace_back(NodeEntry{
        node2index_.at(e.node.get()), e.index, e.version});
  }

  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
  // setup array view
  // input_entries_ and control_rptr must not change after this step.
  const NodeEntry* iptr = dmlc::BeginPtr(input_entries_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].inputs = array_view<NodeEntry>(
        iptr + inputs_rptr[nid], iptr + inputs_rptr[nid + 1]);
    if (nodes_[nid].source->op() != nullptr &&
        fmutate_inputs.count(nodes_[nid].source->op())) {
      for (uint32_t i : fmutate_inputs[nodes_[nid].source->op()](nodes_[nid].source->attrs)) {
        mutable_input_nodes_.insert(nodes_[nid].inputs[i].node_id);
      }
    }
  }
  const uint32_t* cptr = dmlc::BeginPtr(control_deps_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].control_deps = array_view<uint32_t>(
        cptr + control_rptr[nid], cptr + control_rptr[nid + 1]);
  }
}

}  // namespace nnvm
