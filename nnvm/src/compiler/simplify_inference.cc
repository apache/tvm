/*!
 * Copyright (c) 2017 by Contributors
 * \file simplify_inference.cc
 * \author Ziheng Jiang
*/
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "graph_transform.h"
#include "pattern_util.h"

namespace nnvm {
namespace compiler {

std::vector<NodeEntry>
BatchNormToInferUnpack(const nnvm::NodeAttrs& attrs,
                       nnvm::NodeEntry data,
                       nnvm::NodeEntry gamma,
                       nnvm::NodeEntry beta,
                       nnvm::NodeEntry moving_mean,
                       nnvm::NodeEntry moving_var,
                       TShape dshape,
                       TShape bshape) {
  CHECK_NE(dshape.ndim(), 0);
  CHECK(attrs.op);
  static const  Op* bn_op = Op::Get("batch_norm");
  CHECK(attrs.op == bn_op);
  const auto& param = nnvm::get<top::BatchNormParam>(attrs.parsed);
  std::string bn_name = attrs.name;

  // transform batch_norm(data) to scale * data + shift
  NodeEntry var_add_eps = MakeNode(
      "__add_scalar__", bn_name + "_add_eps",
      {moving_var}, {{"scalar", std::to_string(param.epsilon)}});

  NodeEntry sqrt = MakeNode(
      "sqrt", bn_name + "_sqrt", {var_add_eps});

  NodeEntry scale = MakeNode(
      "__rdiv_scalar__", bn_name + "_div",
      {sqrt}, {{"scalar", "1"}});

  if (param.scale) {
    scale = MakeNode(
        "elemwise_mul", bn_name + "_gamma_mul_div",
        {scale, gamma});
  }

  NodeEntry neg_mean = MakeNode(
      "negative", bn_name + "_neg_mean", {moving_mean});

  NodeEntry shift = MakeNode(
      "elemwise_mul", bn_name + "_neg_mean_mul_a",
      {neg_mean, scale});

  if (param.center) {
    shift = MakeNode(
        "elemwise_add", bn_name + "_add_beta", {shift, beta});
  }
  int axis = param.axis;
  scale = ExpandBiasToMatchAxis(scale, dshape.ndim()-bshape.ndim()+1, 1, axis);
  shift = ExpandBiasToMatchAxis(shift, dshape.ndim()-bshape.ndim()+1, 1, axis);

  NodeEntry out = MakeNode("broadcast_mul", bn_name + "_a_mul_data",
                           {data, scale});
  out = MakeNode("broadcast_add", bn_name + "_out",
                 {out, shift});
  // It is invalid to ref the other values of BN after inference transform.
  NodeEntry undef = MakeNode("__undef__", "undef", {});
  return {out, undef, undef};
}

Graph SimplifyInference(nnvm::Graph src) {
  // Get attributes from the graph
  const IndexedGraph& idx = src.indexed_graph();
  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;
    static const Op* bn_op = Op::Get("batch_norm");
    static const Op* dropout_op = Op::Get("dropout");
    if (n->op() == bn_op) {
      *ret = BatchNormToInferUnpack(
          n->attrs,
          n->inputs[0],
          n->inputs[1],
          n->inputs[2],
          n->inputs[3],
          n->inputs[4],
          shape_vec[idx.entry_id(nid, 0)],
          shape_vec[idx.entry_id(nid, 1)]);
      return true;
    } else if (n->op() == dropout_op) {
      NodeEntry undef = MakeNode("__undef__", "undef", {});
      *ret = {n->inputs[0], undef};
      return true;
    } else {
      return false;
    }
  };
  return GraphTransform(src, transform);
}

NNVM_REGISTER_PASS(SimplifyInference)
.set_body(SimplifyInference)
.set_change_graph(true);

}  // namespace compiler
}  // namespace nnvm
