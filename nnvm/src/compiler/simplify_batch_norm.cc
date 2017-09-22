/*!
 * Copyright (c) 2017 by Contributors
 * \file simplify_batch_norm.cc
 * \author Ziheng Jiang
*/
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/nn.h>
#include "./graph_transform.h"

namespace nnvm {
namespace compiler {

std::vector<NodeEntry>
BatchNormToInferUnpack(const nnvm::NodeAttrs& attrs,
                       nnvm::NodeEntry data,
                       nnvm::NodeEntry gamma,
                       nnvm::NodeEntry beta,
                       nnvm::NodeEntry moving_mean,
                       nnvm::NodeEntry moving_var,
                       TShape dshape) {
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
  // use broaodcast to reshape
  std::ostringstream oshape;
  for (dim_t i = 0; i < dshape.ndim(); ++i) {
    dshape[i] = (i != param.axis) ? 1 : -1;
  }
  oshape << dshape;
  scale = MakeNode("reshape", bn_name + "_sc_reshape",
                   {scale}, {{"shape", oshape.str()}});
  shift = MakeNode("reshape", bn_name + "_sh_reshape",
                   {shift}, {{"shape", oshape.str()}});
  NodeEntry out = MakeNode("broadcast_mul", bn_name + "_a_mul_data",
                           {data, scale});
  out = MakeNode("broadcast_add", bn_name + "_out",
                 {out, shift});
  // It is invalid to ref the other values of BN after infernece transform.
  NodeEntry undef = MakeNode("__undef__", "undef", {});
  return {out, undef, undef};
}

Graph SimplifyBatchNormInference(nnvm::Graph src) {
  // Get attributes from the graph
  const IndexedGraph& idx = src.indexed_graph();
  const ShapeVector& shape_vec = src.GetAttr<ShapeVector>("shape");
  auto transform = [&](uint32_t nid, const Node* n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;
    static const Op* bn_op = Op::Get("batch_norm");
    if (n->op() == bn_op) {
      *ret = BatchNormToInferUnpack(
          n->attrs,
          n->inputs[0],
          n->inputs[1],
          n->inputs[2],
          n->inputs[3],
          n->inputs[4],
          shape_vec[idx.entry_id(nid, 0)]);
      return true;
    } else {
      return false;
    }
  };
  return GraphTransform(src, transform);
}

NNVM_REGISTER_PASS(SimplifyBatchNormInference)
.set_body(SimplifyBatchNormInference);

}  // namespace compiler
}  // namespace nnvm
