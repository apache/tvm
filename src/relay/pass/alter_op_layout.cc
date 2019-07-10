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
 * Copyright (c) 2019 by Contributors
 * \file alter_op_layout.cc
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/operation.h>
#include <tuple>
#include <vector>
#include <functional>
#include <string>
#include <utility>
#include <unordered_map>

#include "alter_op_layout.h"

namespace tvm {
namespace relay {

namespace alter_op_layout {

// Make a transform CallNode
Expr TransformLayout(Expr raw, Layout src_layout, Layout dst_layout) {
  if (src_layout.Equals(dst_layout)) { return raw; }
  CHECK(src_layout.defined() && dst_layout.defined())
    << "Cannot insert layout transform because there are undefined layouts";
  CHECK(BijectiveLayoutNode::make(src_layout, dst_layout).defined())
    << "Cannot insert layout transform because there are inconvertible layouts: "
    << src_layout << " v.s. " << dst_layout;
  static auto &transform_op = Op::Get("layout_transform");
  NodePtr<LayoutTransformAttrs> attrs = make_node<LayoutTransformAttrs>();
  attrs->src_layout = src_layout.name();
  attrs->dst_layout = dst_layout.name();
  Call transform = CallNode::make(transform_op, {raw}, Attrs{attrs});
  return std::move(transform);
}

// Memorize layout transform so we can reuse internal transformed nodes
class TransformMemorizerNode : public Node {
 public:
  // map from (Expr, src_layout, dst_layout) to transformed Expr
  using TransformKey = std::tuple<const Node*, std::string, std::string>;
struct key_hash : public std::function<std::size_t(TransformKey)> {
    std::size_t operator()(const TransformKey& k) const {
      return dmlc::HashCombine<std::string>(dmlc::HashCombine<std::string>(
              std::hash<const Node*>()(std::get<0>(k)), std::get<1>(k)), (std::get<2>(k)));
    }
  };

  std::unordered_map<TransformKey, Expr, key_hash> memo;
  static constexpr const char *_type_key = "relay.alter_op_layout.TransformMemorizerNode";
  TVM_DECLARE_NODE_TYPE_INFO(TransformMemorizerNode, Node);
};

class TransformMemorizer : public NodeRef {
 public:
  TransformMemorizer() {}
  explicit TransformMemorizer(NodePtr<Node> n) : NodeRef(n) {}

  TransformMemorizerNode* operator->() {
    return static_cast<TransformMemorizerNode*>(node_.get());
  }

  // Transform layout with memorizer
  Expr Transform(Expr raw, const Layout& src_layout, const Layout& dst_layout) {
    if (src_layout.Equals(dst_layout)) { return raw; }

    std::tuple<const Node*, std::string, std::string> key =
        std::make_tuple<>(raw.get(), src_layout.name(), dst_layout.name());
    auto& memo = operator->()->memo;

    auto iter = memo.find(key);
    if (iter != memo.end()) {
      return iter->second;
    } else {
      Expr transform = TransformLayout(raw, src_layout, dst_layout);
      memo[key] = transform;
      return transform;
    }
  }

  using ContainerType = TransformMemorizerNode;
};


// TempExprNode during layout transform
// Instance of this expr will be Realized to normal expr ultimately
class LayoutAlternatedExprNode : public TempExprNode {
 public:
  Expr value;
  Layout old_layout;
  Layout new_layout;
  TransformMemorizer memorizer;

  Expr Realize() const final {
    // NOTE: use a copy to discard the "const" qualifier
    TransformMemorizer tmp_memorizer = memorizer;
    // fallback to old layout
    return tmp_memorizer.Transform(value, new_layout, old_layout);
  }

  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("value", &value);
    v->Visit("old_layout", &old_layout);
    v->Visit("new_layout", &new_layout);
  }

  static constexpr const char *_type_key = "relay.alter_op_layout.LayoutAlternatedExprNode";
  TVM_DECLARE_NODE_TYPE_INFO(LayoutAlternatedExprNode, TempExprNode);
};

RELAY_DEFINE_NODE_REF(LayoutAlternatedExpr, LayoutAlternatedExprNode, TempExpr);

// Call registered FInferCorrectLayout of an op.
// Parameters are the same as the parameters for FInferCorrectLayout
// Returns inferred_input_layout, inferred_output_layout, success
std::tuple<Array<Layout>, Array<Layout>, bool> CallInfer(
    const Call& call,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr> > &old_in_shapes) {
  static auto finfer_layout = Op::GetAttr<FInferCorrectLayout>("FInferCorrectLayout");

  Op op = Downcast<Op>(call->op);
  if (finfer_layout.count(op)) {
    Array<Array<Layout> > inferred_layouts;
    inferred_layouts = finfer_layout[op](call->attrs, new_in_layouts,
                                         old_in_layouts, old_in_shapes);
    CHECK_EQ(inferred_layouts.size(), 2)
      << "FInferCorrectLayout should return an array with size of 2";
    for (auto x : inferred_layouts) {
      for (auto y : x) {
        if (!y.defined()) {  // inference fails
          return std::make_tuple<>(Array<Layout>(nullptr), Array<Layout>(nullptr), false);
        }
      }
    }
    return std::make_tuple<>(inferred_layouts[0], inferred_layouts[1], true);
  } else {
    return std::make_tuple<>(Array<Layout>(nullptr), Array<Layout>(nullptr), false);
  }
}

// Call registered FTVMAlterOpLayout of an op
// Returns the altered expression
Call CallAlter(const Call& ref_call,
               const std::vector<Expr>& new_args) {
  static auto falter_layout = Op::GetAttr<FTVMAlterOpLayout>("FTVMAlterOpLayout");
  Op op = Downcast<Op>(ref_call->op);

  Expr new_e;
  bool modified = false;
  if (falter_layout.count(op)) {
    tvm::Array<tvm::Tensor> tinfos;
    for (auto expr : ref_call->args) {
      auto ttype = expr->type_as<TensorTypeNode>();
      tinfos.push_back(tvm::placeholder(ttype->shape, ttype->dtype));
    }
    Expr altered_value = falter_layout[op](ref_call->attrs, new_args, tinfos);
    if (altered_value.defined()) {
      new_e = altered_value;
      modified = true;
    }
  }
  if (!modified) {
    new_e = CallNode::make(ref_call->op, new_args,
                           ref_call->attrs);
  }

  const CallNode *new_call = new_e.as<CallNode>();
  CHECK(new_call) << "Can only replace the original operator with another call node";
  return GetRef<Call>(new_call);
}

Expr AlterOpLayoutRewrite(const Call &ref_call,
                          const Array<Expr> &new_args,
                          const NodeRef& ctx) {
  std::vector<LayoutAlternatedExpr> inputs;
  std::vector<Expr> normal_new_args;
  Array<Array<IndexExpr> > input_shapes;

  // NOTE: discard the "const" qualifier
  TransformMemorizer memorizer = Downcast<TransformMemorizer>(ctx);

  // fill incomplete state and flatten tuple
  auto push_back_one_arg = [&inputs, memorizer](Expr arg) {
    // We always expect LayoutAlternatedExpr.
    // This is used to convert the normal Expr to LayoutAlternatedExpr.
    if (const LayoutAlternatedExprNode *inp = arg.as<LayoutAlternatedExprNode>()) {
      inputs.push_back(GetRef<LayoutAlternatedExpr>(inp));
      return inp->value;
    } else {
      auto inode = make_node<LayoutAlternatedExprNode>();
      inode->value = arg;
      inode->memorizer = memorizer;
      inputs.push_back(LayoutAlternatedExpr(inode));
      return arg;
    }
  };

  for (auto new_arg : new_args) {
    // NOTE: do not support nested tuple
    if (new_arg->is_type<TupleNode>()) {
      Tuple tuple_new_arg = Downcast<Tuple>(new_arg);
      std::vector<Expr> fields;
      for (auto x : tuple_new_arg->fields) {
        Expr tmp = push_back_one_arg(x);
        fields.push_back(tmp);
      }
      normal_new_args.push_back(TupleNode::make(fields));
    } else {
      Expr tmp = push_back_one_arg(new_arg);
      normal_new_args.push_back(tmp);
    }
  }

  // old_in, new_in = state[inputs]
  Array<Layout> old_in, old_out, new_in, new_out, new_in2;
  for (auto inp : inputs) {
    old_in.push_back(inp->old_layout);
    new_in.push_back(inp->new_layout);
  }

  for (auto arg : ref_call->args) {
    if (arg->is_type<TupleNode>()) {  // flatten tuple
      Tuple tuple_arg = Downcast<Tuple>(arg);
      for (auto x : tuple_arg->fields) {
        input_shapes.push_back(x->type_as<TensorTypeNode>()->shape);
      }
    } else {
      input_shapes.push_back(arg->type_as<TensorTypeNode>()->shape);
    }
  }

  // old_in, old_out = op.infer(old_in)
  bool success = false;
  std::tie(old_in, old_out, success) = CallInfer(ref_call,
                                                 Array<Layout>(nullptr),
                                                 old_in, input_shapes);
  if (!success) { return Expr(nullptr); }
  CHECK_EQ(old_in.size(), new_in.size());

  // if new_in == 'undef':  new_in = old_in
  for (size_t i = 0; i < new_in.size(); ++i) {
    if (!new_in[i].defined()) {
      new_in.Set(i, old_in[i]);
    }
  }

  // new_op = alter(op)
  Call new_call = CallAlter(ref_call, normal_new_args);

  // new_in2, new_out = op.infer(new_in)
  if (new_call->op->is_type<OpNode>()) {
    success = false;
    std::tie(new_in2, new_out, success) = CallInfer(new_call, new_in, old_in, input_shapes);
    if (!success) { return Expr(nullptr); }
  } else {
    return Expr(nullptr);
  }

  CHECK_EQ(new_out.size(), old_out.size())
    << "The number of output nodes should keep the same during alter_op_layout";
  CHECK_EQ(new_in.size(), new_in2.size())
    << "The number of input nodes should keep the same during alter_op_layout";

  // if (new_in != new_in2): insert transform (new_in -> new_in2)
  Array<Expr> transformed_args;
  size_t pt = 0;
  for (auto arg : new_call->args) {
    if (arg->is_type<TupleNode>()) {  // unflatten tuple
      Tuple tuple_arg = Downcast<Tuple>(arg);
      std::vector<Expr> transformed_tuple_arg;
      for (auto arg_item : tuple_arg->fields) {
          transformed_tuple_arg.push_back(
                  memorizer.Transform(arg_item, new_in[pt], new_in2[pt]));
          pt++;
      }
      transformed_args.push_back(TupleNode::make(transformed_tuple_arg));
    } else {
      transformed_args.push_back(
              memorizer.Transform(arg, new_in[pt], new_in2[pt]));
      pt++;
    }
  }
  CHECK_EQ(pt, inputs.size());

  // state[node] = (old_out, new_out)
  // (handle tuple output)
  if (ref_call->checked_type()->is_type<TupleTypeNode>()) {
    Expr tuple_output = CallNode::make(new_call->op, transformed_args,
                                       new_call->attrs);
    Array<Expr> fields;
    for (size_t i = 0; i < new_out.size(); ++i) {
      auto rnode = make_node<LayoutAlternatedExprNode>();
      rnode->value = TupleGetItemNode::make(tuple_output, i);
      rnode->old_layout = old_out[i];
      rnode->new_layout = new_out[i];
      rnode->memorizer = memorizer;
      fields.push_back(Expr(rnode));
    }
    return TupleNode::make(fields);
  } else {
    auto rnode = make_node<LayoutAlternatedExprNode>();
    CHECK_EQ(new_out.size(), 1);
    rnode->value = CallNode::make(new_call->op, transformed_args,
                                  new_call->attrs);
    rnode->old_layout = old_out[0];
    rnode->new_layout = new_out[0];
    rnode->memorizer = memorizer;
    return Expr(rnode);
  }
}

// Limiations:
// 1. the altered op should have the same number of arguments as the previous one
// 2. do not support nested tuple arguments
Expr AlterOpLayout(const Expr& expr) {
  TransformMemorizer transformMemorizer(make_node<TransformMemorizerNode>());
  auto fcontext = [&](const Call& call) -> NodeRef{
    return transformMemorizer;
  };

  return ForwardRewrite(expr, AlterOpLayoutRewrite, fcontext);
}

}  // namespace alter_op_layout

namespace transform {

Pass AlterOpLayout() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(relay::alter_op_layout::AlterOpLayout(f));
  };
  return CreateFunctionPass(pass_func, 3, "AlterOpLayout",
                            {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.AlterOpLayout")
.set_body_typed(AlterOpLayout);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
