/*!
 *  Copyright (c) 2017 by Contributors
 * \file layout_transform.cc
 * \brief Transforms layout.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/contrib_op_param.h>

namespace nnvm {
namespace compiler {

const TLayoutInfo& GetDefaultLayout() {
  static TLayoutInfo default_layout = "default";
  return default_layout;
}

nnvm::NodePtr CreateLayoutTransformNode(const std::string& src,
                                        const std::string& dst) {
  static const nnvm::Op* trans_op = nnvm::Op::Get("layout_transform");
  static int count = 0;
  nnvm::NodePtr n = nnvm::Node::Create();
  n->attrs.op = trans_op;
  n->attrs.name = src + "_to_" + dst + std::to_string(count++);
  n->attrs.dict["src_layout"] = src;
  n->attrs.dict["dst_layout"] = dst;
  n->op()->attr_parser(&(n->attrs));
  return n;
}

/*!
 * \brief A simple layout transform pass that will
 *  insert layout transform nodes automatically.
 */
nnvm::Graph LayoutTransform(nnvm::Graph src) {
  static auto& op_layout_request =
    nnvm::Op::GetAttr<FTVMLayoutRequest>("FTVMLayoutRequest");
  static auto& op_vecop =
    nnvm::Op::GetAttr<FTVMVectorizedOp>("FTVMVectorizedOp");
  static auto& op_pattern = nnvm::Op::GetAttr<TOpPattern>("TOpPattern");

  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  const std::vector<TLayoutInfo>& input_layouts =
      src.GetAttr<std::vector<TLayoutInfo> >("layout_inputs");

  const IndexedGraph& idx = src.indexed_graph();
  std::vector<TLayoutInfo> produce_vec(idx.num_node_entries(), GetDefaultLayout());
  std::vector<nnvm::NodePtr> mirror_vec(idx.num_nodes(), nullptr);

  // use op pattern to decide whether an op is map
  auto is_map_op = [&](size_t nid) {
    TOpPattern pt = op_pattern.get(idx[nid].source->op(), kOpaque);
    bool is_map = (pt <= kBroadcast);
    if (pt == kBroadcast) {
      for (const auto& e : idx[nid].inputs) {
        if (shape_vec[idx.entry_id(nid, 0)] != shape_vec[idx.entry_id(e)]) {
          is_map = false;
          break;
        }
      }
    }
    return is_map;
  };

  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    nnvm::NodePtr new_node = nnvm::Node::Create();
    *new_node = *(inode.source);
    if (new_node->is_variable()) {
      auto input_iter = std::find(
        idx.input_nodes().cbegin(), idx.input_nodes().cend(), nid);
      CHECK(input_iter != idx.input_nodes().cend());
      size_t input_id = std::distance(idx.input_nodes().cbegin(), input_iter);
      produce_vec[idx.entry_id(nid, 0)] = input_layouts[input_id];
      mirror_vec[nid] = new_node;
      continue;
    }

    if (op_vecop.count(inode.source->op())) {
      new_node = op_vecop[inode.source->op()](inode.source);
      new_node->inputs.resize(new_node->num_inputs());
    }

    // set up output and input layouts
    std::vector<TLayoutInfo> request_ilayouts(new_node->num_inputs(), GetDefaultLayout());
    if (op_layout_request.count(new_node->op())) {
      std::vector<TLayoutInfo> produce_olayouts(new_node->num_outputs(), GetDefaultLayout());
      CHECK(op_layout_request[new_node->op()](
          new_node->attrs, &request_ilayouts, &produce_olayouts))
          << "Layout request fail";

      CHECK_EQ(request_ilayouts.size(), new_node->num_inputs());
      CHECK_EQ(produce_olayouts.size(), new_node->num_outputs());
      for (size_t i = 0; i < new_node->num_outputs(); ++i) {
        produce_vec[idx.entry_id(nid, i)] = produce_olayouts[i];
      }
    }

    bool map_layout = is_map_op(nid);
    if (map_layout) {
      const TLayoutInfo& layout = produce_vec[idx.entry_id(inode.inputs[0])];
      for (const auto& e : inode.inputs) {
        if (produce_vec[idx.entry_id(e)] != layout) {
          map_layout = false;
          break;
        }
      }
      if (map_layout) {
        for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
          produce_vec[idx.entry_id(nid, i)] = layout;
        }
      }
    }

    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      const auto& e = inode.inputs[i];
      const nnvm::NodePtr& in = mirror_vec[e.node_id];
      new_node->inputs[i] =
        nnvm::NodeEntry{in, e.index, e.version};

      TLayoutInfo produce = produce_vec[idx.entry_id(e)];
      TLayoutInfo request = request_ilayouts[i];
      if (!map_layout && (produce != request)) {
        nnvm::NodePtr tnode = CreateLayoutTransformNode(produce, request);
        tnode->attrs.name =
          idx[e.node_id].source->attrs.name + "_" + request;
        tnode->inputs.emplace_back(new_node->inputs[i]);
        new_node->inputs[i] = nnvm::NodeEntry{tnode, 0, 0};
      }
    }
    mirror_vec[nid] = new_node;
  }

  std::vector<nnvm::NodeEntry> outputs;
  for (const auto& e : idx.outputs()) {
    TLayoutInfo produce = produce_vec[idx.entry_id(e)];
    if (produce != GetDefaultLayout()) {
      nnvm::NodePtr tnode = CreateLayoutTransformNode(produce, GetDefaultLayout());
      tnode->attrs.name =
        idx[e.node_id].source->attrs.name + "_default";
      tnode->inputs.emplace_back(
        nnvm::NodeEntry{mirror_vec[e.node_id], e.index, e.version});
      outputs.emplace_back(nnvm::NodeEntry{tnode, 0, 0});
    } else {
      outputs.emplace_back(
        nnvm::NodeEntry{mirror_vec[e.node_id], e.index, e.version});
    }
  }

  nnvm::Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

}  // namespace compiler
}  // namespace nnvm
