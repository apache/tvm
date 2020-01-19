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
 *
 * \file transform_layout.h
 * \brief Common infrastructure for transforming the layouts. This is used for AlterOpLayout and
 *        ConvertLayout pass. */

#ifndef TVM_RELAY_PASS_TRANSFORM_LAYOUT_H_
#define TVM_RELAY_PASS_TRANSFORM_LAYOUT_H_

#include <tvm/tir/data_layout.h>
#include <tvm/relay/expr.h>
#include <string>
#include <unordered_map>
#include <tuple>
#include <vector>
#include "pattern_util.h"
#include "infer_layout_util.h"

namespace tvm {
namespace relay {

/*!
 * \brief Memorizes layout transformations to reuse.
 */
class TransformMemorizerNode : public Object {
 public:
  /*! \brief The key for the memorizer map is (Expr, src_layout, dst_layout). */
  using TransformKey = std::tuple<const Object*, std::string, std::string>;

  struct key_hash : public std::function<std::size_t(TransformKey)> {
    std::size_t operator()(const TransformKey& k) const {
      return dmlc::HashCombine<std::string>(
          dmlc::HashCombine<std::string>(
              std::hash<const Object*>()(std::get<0>(k)), std::get<1>(k)),
          (std::get<2>(k)));
    }
  };

  /*! \brief The memorizer map. */
  std::unordered_map<TransformKey, Expr, key_hash> memo;

  static constexpr const char* _type_key = "relay.alter_op_layout.TransformMemorizerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(TransformMemorizerNode, Object);
};

/*!
 * \brief Container that transforms the layouts and memorizes them.
 */
class TransformMemorizer : public ObjectRef {
 public:
  TransformMemorizer() {}
  explicit TransformMemorizer(ObjectPtr<Object> n) : ObjectRef(n) {}

  TransformMemorizerNode* operator->() {
    return static_cast<TransformMemorizerNode*>(get_mutable());
  }

  /*
   * \brief Memorizes and transforms the layout.
   * \param expr The initial expr.
   * \param src_layout The source layout.
   * \param dst_layout The dest layout.
   * \return The new expr with the dst layout.
   */
  Expr Transform(Expr raw, const Layout& src_layout, const Layout& dst_layout) {
    if (src_layout.Equals(dst_layout)) {
      return raw;
    }

    std::tuple<const Object*, std::string, std::string> key =
        std::make_tuple<>(raw.get(), src_layout.name(), dst_layout.name());
    auto& memo = operator->()->memo;

    auto iter = memo.find(key);
    if (iter != memo.end()) {
      return iter->second;
    } else {
      Expr transform = TransformHelper(raw, src_layout, dst_layout);
      memo[key] = transform;
      return transform;
    }
  }

  /*
   * \brief Helper to transform the layouts.
   * \param expr The initial expr.
   * \param src_layout The source layout.
   * \param dst_layout The dest layout.
   * \return The new expr with the dst layout.
   * \note It performs following 2 operations
   *       1) If src_layout ndim is smaller then dst_layout, expand_dim is inserted to match the dim
   *          size. For example, src_layout = C, dst_layout = NCHW16c. The src is expanded to NHWC.
   *       2) Call layout transform with new src layout.
   */
  Expr TransformHelper(Expr raw, Layout src_layout, Layout dst_layout) {
    if (src_layout.Equals(dst_layout)) {
      return raw;
    }

    // 1) Check if the shape lengths are different. If yes, expand dims.
    Expr input_expr = raw;
    Layout new_src_layout = src_layout;
    if (src_layout.ndim_primal() < dst_layout.ndim_primal()) {
      // If scalar, then no need of layout transformation as scalar can be broadcasted easily even
      // if the other operand has a transformed layout.
      if (IsScalar(input_expr)) {
        return raw;
      }
      int num_new_axis = dst_layout.ndim_primal() - src_layout.ndim_primal();
      new_src_layout = src_layout.ExpandPrimal(dst_layout);
      input_expr = MakeExpandDims(input_expr, 0, num_new_axis);
      if (new_src_layout.Equals(dst_layout)) {
        return input_expr;
      }
    }

    // 2) Insert layout transform on the transformed src.
    CHECK(new_src_layout.defined() && dst_layout.defined())
        << "Cannot insert layout transform because there are undefined layouts";
    CHECK(BijectiveLayoutNode::make(new_src_layout, dst_layout).defined())
        << "Cannot insert layout transform because there are inconvertible layouts: "
        << new_src_layout << " v.s. " << dst_layout;
    return MakeLayoutTransform(input_expr, new_src_layout.name(), dst_layout.name());
  }

  /*!
   * \brief Defines the call transformation for derived passes. The new layouts are defined by
   * used for different targets using a packed func.
   * \param ref_call The original call.
   * \param new_args The traversed/recursed args to the call.
   * \return The new Call after calling the packed func.
   */
  virtual Call CallWithNewLayouts(const Call& ref_call, const std::vector<Expr>& new_args) = 0;
  using ContainerType = TransformMemorizerNode;
};

/*
 * \brief TempExprNode during layout transform. Instance of this expr will be Realized to normal
 *        expr ultimately.
 * \tparam TransformMemorizerT The derived TransformMemorizer type.
 */
template <class TransformMemorizerT>
class LayoutAlternatedExprNode : public TempExprNode {
 public:
  Expr value;
  Layout old_layout;
  Layout new_layout;
  TransformMemorizerT memorizer;

  Expr Realize() const final {
    // NOTE: use a copy to discard the "const" qualifier
    TransformMemorizerT tmp_memorizer = memorizer;
    // fallback to old layout
    return tmp_memorizer.Transform(value, new_layout, old_layout);
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("old_layout", &old_layout);
    v->Visit("new_layout", &new_layout);
  }

  static constexpr const char* _type_key = "relay.alter_op_layout.LayoutAlternatedExprNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayoutAlternatedExprNode, TempExprNode);
};

/*!
 * \brief Container for the layout alternated expr.
 * \tparam TransformMemorizerT The derived TransformMemorizer type.
 */
template <class TransformMemorizerT>
class LayoutAlternatedExpr : public ObjectRef {
 public:
  LayoutAlternatedExpr() {}
  explicit LayoutAlternatedExpr(ObjectPtr<Object> n) : ObjectRef(n) {}

  LayoutAlternatedExprNode<TransformMemorizerT>* operator->() {
    return static_cast<LayoutAlternatedExprNode<TransformMemorizerT>*>(get_mutable());
  }

  using ContainerType = LayoutAlternatedExprNode<TransformMemorizerT>;
};

/*
 * \brief Used with ForwardRewrite to transform the expr. The input args are same as
 *        FForwardRewrite.
 * \param ref_call The reference old call type to be rewritten.
 *                 We can make use of the op and type information.
 * \param new_args The new arguments (some of them could be TempExpr).
 * \param ctx  Optional context information about ref_call.
 * \tparam TransformMemorizerT The derived TransformMemorizer type.
 * \return The rewriten result call, can also return nullptr,
 *         which indicate the rewriter should use the default fallback
 *         rule that realizes all its input and compose the call.
 *
 * \note The ctx can be used to provide extra information during transformation. The ctx is
 *       templated to reuse across AlterOpLayout and ConvertLayout pass. The steps are
 *       - Extract the original layouts.
 *       - Use ctx transformation to get a Call with new layouts - CallWithNewLayouts.
 *       - Extract the new layouts from the returned Call.
 *       - Transform the original call to reuse the new layouts using TransformMemorizer.
 */
template <class TransformMemorizerT>
Expr LayoutRewriter(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  std::vector<LayoutAlternatedExpr<TransformMemorizerT>> inputs;
  std::vector<Expr> normal_new_args;
  Array<Array<IndexExpr>> input_shapes;

  // NOTE: discard the "const" qualifier
  // TransformMemorizer memorizer = Downcast<TransformMemorizer>(ctx);
  // TransformMemorizerT* ctx_transformer =
  // static_cast<TransformMemorizerT*>(memorizer.operator->());
  TransformMemorizerT memorizer = Downcast<TransformMemorizerT>(ctx);

  // fill incomplete state and flatten tuple
  auto push_back_one_arg = [&inputs, memorizer](Expr arg) {
    // We always expect LayoutAlternatedExpr<TransformMemorizerT>.
    // This is used to convert the normal Expr to LayoutAlternatedExpr<TransformMemorizerT>.
    if (const LayoutAlternatedExprNode<TransformMemorizerT>* inp =
            arg.as<LayoutAlternatedExprNode<TransformMemorizerT>>()) {
      inputs.push_back(GetRef<LayoutAlternatedExpr<TransformMemorizerT>>(inp));
      return inp->value;
    } else {
      auto inode = make_object<LayoutAlternatedExprNode<TransformMemorizerT>>();
      inode->value = arg;
      inode->memorizer = memorizer;
      inputs.push_back(LayoutAlternatedExpr<TransformMemorizerT>(inode));
      return arg;
    }
  };

  for (auto new_arg : new_args) {
    // NOTE: do not support nested tuple
    if (new_arg->IsInstance<TupleNode>()) {
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
    if (arg->IsInstance<TupleNode>()) {  // flatten tuple
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
  std::tie(old_in, old_out, success) =
      InferCorrectLayouts(ref_call, Array<Layout>(nullptr), old_in, input_shapes);
  if (!success) {
    return Expr(nullptr);
  }
  CHECK_EQ(old_in.size(), new_in.size());

  // if new_in == 'undef':  new_in = old_in
  for (size_t i = 0; i < new_in.size(); ++i) {
    if (!new_in[i].defined()) {
      new_in.Set(i, old_in[i]);
    }
  }

  // new_op = alter(op)
  Call new_call = memorizer.CallWithNewLayouts(ref_call, normal_new_args);

  // new_in2, new_out = op.infer(new_in)
  if (new_call->op->IsInstance<OpNode>()) {
    success = false;
    std::tie(new_in2, new_out, success) =
        InferCorrectLayouts(new_call, new_in, old_in, input_shapes);
    if (!success) {
      return Expr(nullptr);
    }
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
    if (arg->IsInstance<TupleNode>()) {  // unflatten tuple
      Tuple tuple_arg = Downcast<Tuple>(arg);
      std::vector<Expr> transformed_tuple_arg;
      for (auto arg_item : tuple_arg->fields) {
        transformed_tuple_arg.push_back(memorizer.Transform(arg_item, new_in[pt], new_in2[pt]));
        pt++;
      }
      transformed_args.push_back(TupleNode::make(transformed_tuple_arg));
    } else {
      transformed_args.push_back(memorizer.Transform(arg, new_in[pt], new_in2[pt]));
      pt++;
    }
  }
  CHECK_EQ(pt, inputs.size());

  // state[node] = (old_out, new_out)
  // (handle tuple output)
  if (ref_call->checked_type()->IsInstance<TupleTypeNode>()) {
    Expr tuple_output = CallNode::make(new_call->op, transformed_args, new_call->attrs);
    Array<Expr> fields;
    for (size_t i = 0; i < new_out.size(); ++i) {
      auto rnode = make_object<LayoutAlternatedExprNode<TransformMemorizerT>>();
      rnode->value = TupleGetItemNode::make(tuple_output, i);
      rnode->old_layout = old_out[i];
      rnode->new_layout = new_out[i];
      rnode->memorizer = memorizer;
      fields.push_back(Expr(rnode));
    }
    return TupleNode::make(fields);
  } else {
    auto rnode = make_object<LayoutAlternatedExprNode<TransformMemorizerT>>();
    CHECK_EQ(new_out.size(), 1);
    rnode->value = CallNode::make(new_call->op, transformed_args, new_call->attrs);
    rnode->old_layout = old_out[0];
    rnode->new_layout = new_out[0];
    rnode->memorizer = memorizer;
    return Expr(rnode);
  }
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_PASS_TRANSFORM_LAYOUT_H_
