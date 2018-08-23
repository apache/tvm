/*!
 *  Copyright (c) 2018 by Contributors
 * \file alter_op_layout.cc
 * \brief Alter the operator layouts. Keep inferred layouts (if any) from previous stages.
 *        e.g., convolution may calculates faster with NCHW16c layout.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/layout.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/pass_functions.h>
#include <tvm/tvm.h>
#include <algorithm>
#include <functional>
#include "compile_engine.h"
#include "graph_transform.h"

namespace nnvm {
namespace compiler {
namespace {

tvm::Array<tvm::Tensor> GetTensorInfo(const IndexedGraph& idx_graph,
                                      const uint32_t nid,
                                      const ShapeVector& shape_vec,
                                      const DTypeVector& dtype_vec) {
  tvm::Array<tvm::Tensor> vec;
  for (uint32_t i = 0; i < idx_graph[nid].source->num_outputs(); ++i) {
    tvm::Array<tvm::Expr> shape;
    for (int64_t x : shape_vec[idx_graph.entry_id(nid, i)]) {
      CHECK_LE(x, static_cast<int64_t>(std::numeric_limits<int>::max()));
      shape.push_back(tvm::make_const(tvm::Int(32), x));
    }
    vec.push_back(tvm::placeholder(
      shape, GetTVMType(dtype_vec[idx_graph.entry_id(nid, i)])));
  }
  return vec;
}

Graph AlterOpLayout(const Graph& src) {
  static auto& falter_op_layout =
    Op::GetAttr<nnvm::compiler::FTVMAlterOpLayout >("FTVMAlterOpLayout");

  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = src.GetAttr<DTypeVector>("dtype");
  const IndexedGraph& idx_graph = src.indexed_graph();

  std::vector<std::vector<Layout> > in_layouts_of_node(idx_graph.num_nodes());
  std::vector<std::vector<Layout> > out_layouts_of_node(idx_graph.num_nodes());
  std::unordered_map<const Node*, uint32_t> new_nodes;

  if (src.HasAttr("layout")) {
    // record layouts so that LayoutTransform pass can fix layouts correctly,
    // e.g., conv2d can be replaced by some contrib implement
    // whose layout is different from the original one
    // (which was imported from a model file).
    const auto& layouts = src.GetAttr<std::vector<Layout> >("layout");
    for (uint32_t nid = 0; nid < idx_graph.num_nodes(); ++nid) {
      const auto &inode = idx_graph[nid];
      if (falter_op_layout.count(inode.source->op())) {
        // do not record input layouts of nodes that will be replaced.
        continue;
      }
      std::vector<Layout> in_layout;
      for (const auto& e : inode.inputs) {
        in_layout.emplace_back(layouts[idx_graph.entry_id(e)]);
      }
      in_layouts_of_node[nid] = in_layout;

      std::vector<Layout> out_layout;
      for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
        out_layout.emplace_back(layouts[idx_graph.entry_id(nid, i)]);
      }
      out_layouts_of_node[nid] = out_layout;
    }
  }

  auto transform = [&](uint32_t nid,
                       const NodePtr& n,
                       std::vector<NodeEntry>* ret) {
    nnvm::compiler::FTVMAlterOpLayout fn_alter_op_layout =
      falter_op_layout.get(n->op(), nullptr);
    if (fn_alter_op_layout == nullptr) {
      new_nodes[n.get()] = nid;
      return false;
    }

    // construct parameters for registered function
    std::vector<Symbol> op_inputs;
    tvm::Array<tvm::Tensor> tensor_infos;
    CHECK_EQ(n->num_inputs(), idx_graph[nid].inputs.size());
    for (uint32_t i = 0; i < n->num_inputs(); ++i) {
      const nnvm::NodeEntry& input = n->inputs[i];
      // input operator
      Symbol op_input;
      op_input.outputs.push_back(input);
      op_inputs.push_back(op_input);

      // input tinfo, extract from the original graph
      // because it was where infer_shape & infer_type applied.
      tvm::Array<tvm::Tensor> op_output_tinfos =
        GetTensorInfo(idx_graph, idx_graph[nid].inputs[i].node_id,
                      shape_vec, dtype_vec);
      tensor_infos.push_back(op_output_tinfos[input.index]);
    }
    // callback registered function to get a new operator.
    Symbol op;
    bool do_alter =
      fn_alter_op_layout(n->attrs, Symbol::CreateGroup(op_inputs), tensor_infos, &op);
    if (do_alter) *ret = op.outputs;
    return do_alter;
  };

  Graph ret = nnvm::compiler::GraphTransform(src, transform);

  if (src.HasAttr("layout")) {
    // restore the layouts to return graph
    const auto& ret_idx = ret.indexed_graph();
    std::vector<Layout> ret_layouts(ret_idx.num_node_entries(), Layout::Undef());
    for (uint32_t nid = 0; nid < ret_idx.num_nodes(); ++nid) {
      const auto& inode = ret_idx[nid];
      if (new_nodes.count(inode.source)) {
        const std::vector<Layout>& in_layouts =
          in_layouts_of_node[new_nodes[inode.source]];
        for (uint32_t i = 0; i < inode.inputs.size(); ++i) {
          const auto& e = inode.inputs[i];
          ret_layouts[ret_idx.entry_id(e)] = in_layouts[i];
        }
        const std::vector<Layout>& out_layouts =
          out_layouts_of_node[new_nodes[inode.source]];
        for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
          ret_layouts[ret_idx.entry_id(nid, i)] = out_layouts[i];
        }
      }
    }

    // cannot call indexed_graph() before return the origin Graph,
    // thus create a new one.
    nnvm::Graph new_ret;
    new_ret.outputs = ret.outputs;
    new_ret.attrs["layout"] = std::make_shared<any>(std::move(ret_layouts));
    return new_ret;
  }

  return ret;
}

// register pass
NNVM_REGISTER_PASS(AlterOpLayout)
.set_body(AlterOpLayout)
.set_change_graph(true);

}  // namespace
}  // namespace compiler
}  // namespace nnvm
