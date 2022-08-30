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

#ifndef TVM_RELAY_TRANSFORMS_TRANSFORM_LAYOUT_H_
#define TVM_RELAY_TRANSFORMS_TRANSFORM_LAYOUT_H_

#include <tvm/relay/expr.h>
#include <tvm/tir/data_layout.h>

#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "infer_layout_utils.h"
#include "pattern_utils.h"

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
          dmlc::HashCombine<std::string>(std::hash<const Object*>()(std::get<0>(k)),
                                         std::get<1>(k)),
          (std::get<2>(k)));
    }
  };

  /*!
   * \brief Defines the call transformation for derived passes. The new layouts are defined by
   * used for different targets using a packed func.
   * \param ref_call The original call.
   * \param new_attrs Updated attributes consistent with new layouts.
   * \param new_args The traversed/recursed args to the call.
   * \return The new Call after calling the packed func.
   */
  virtual Call CallWithNewLayouts(const Call& ref_call, Attrs new_attrs,
                                  const std::vector<Expr>& new_args) = 0;

  virtual Call CallWithNewLayouts(const Call& ref_call, const std::vector<Expr>& new_args) {
    return CallWithNewLayouts(ref_call, ref_call->attrs, new_args);
  }

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
  TransformMemorizer() = default;
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
      if (input_expr->checked_type_.defined() && IsScalar(input_expr)) {
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
    ICHECK(new_src_layout.defined() && dst_layout.defined())
        << "Cannot insert layout transform because there are undefined layouts";
    ICHECK(tir::BijectiveLayout(new_src_layout, dst_layout).defined())
        << "Cannot insert layout transform because there are inconvertible layouts: "
        << new_src_layout << " v.s. " << dst_layout;
    return MakeLayoutTransform(input_expr, new_src_layout.name(), dst_layout.name());
  }

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

/*!
 * Call registered FInferCorrectLayout of an op.
 * Parameters are the same as the parameters for FInferCorrectLayout
 * Returns inferred_input_layout, inferred_output_layout, updated attributes, and a flag
 * indicating whether or not layout conversion is successful.
 */
static inline std::tuple<InferCorrectLayoutOutput, bool> InferCorrectLayouts(
    const Call& call, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  static auto finfer_layout = Op::GetAttrMap<FInferCorrectLayout>("FInferCorrectLayout");
  auto null_res = std::make_tuple(
      InferCorrectLayoutOutput(Array<Layout>(nullptr), Array<Layout>(nullptr), Attrs(nullptr)),
      false);
  if (!call->op.as<OpNode>()) {
    return null_res;
  }

  Op op = Downcast<Op>(call->op);
  if (finfer_layout.count(op)) {
    auto out = finfer_layout[op](call->attrs, new_in_layouts, old_in_layouts, old_in_types);
    for (auto inferred_layouts : {out->input_layouts, out->output_layouts}) {
      for (auto layout : inferred_layouts) {
        if (!layout.defined()) {  // inference fails
          return null_res;
        }
      }
    }
    return std::make_tuple(out, true);
  } else {
    return null_res;
  }
}

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
      Array<Expr> fields;
      fields.reserve(tuple_new_arg->fields.size());
      for (auto x : tuple_new_arg->fields) {
        Expr tmp = push_back_one_arg(x);
        fields.push_back(tmp);
      }
      normal_new_args.push_back(WithFields(tuple_new_arg, fields));
    } else {
      Expr tmp = push_back_one_arg(new_arg);
      normal_new_args.push_back(tmp);
    }
  }

  // If there is no FInferCorrectLayout for the type, then we just assume the layout is correct.
  static auto finfer_layout = Op::GetAttrMap<FInferCorrectLayout>("FInferCorrectLayout");
  if (Op::HasAttrMap("FTVMAlterOpLayout")) {
    static auto falter_layout = Op::GetAttrMap<FTVMAlterOpLayout>("FTVMAlterOpLayout");
    if (ref_call->op.as<OpNode>()) {
      Op op = Downcast<Op>(ref_call->op);
      if (falter_layout.count(op) && !finfer_layout.count(op)) {
        return memorizer->CallWithNewLayouts(ref_call, normal_new_args);
      }
    }
  }

  // old_prd, new_prd = state[inputs]
  // different ops can view a tensor with different layouts, e.g. conv_1->transpose(H, W)->conv_2
  // transpose view its output having NCWH layout, but conv_2 still views it as NCHW to operate
  // old_prd, new_prd: the input layouts from the perspective of the producer (transpose)
  // old_cur, new_cur: the input layouts from the perspective of the current node (conv_2)
  // old_prd->new_prd tells how producer changed the layout
  // old_cur->new_cur tells what change the current node wants to see
  // No layout transforms are needed when they mean the same (NCHW->NCHW4c == NCWH->NCWH4c)

  // The workflow:
  // 1. Run InferCorrectLayouts(NULL, old_prd) to get old_cur
  // 2. Run InferCorrectLayouts(new_prd, old_prd) to get new_cur and rewrite the current op

  Array<Layout> old_prd, old_cur, old_out, new_prd, new_out, new_cur;
  for (auto inp : inputs) {
    old_prd.push_back(inp->old_layout);
    new_prd.push_back(inp->new_layout);
  }

  // Collect input types to pass on to Infer Correct Layout.
  tvm::Array<tvm::relay::Type> types;
  for (auto arg : ref_call->args) {
    types.push_back(arg->checked_type());
  }

  bool success = false;
  InferCorrectLayoutOutput infer_out;
  std::tie(infer_out, success) =
      InferCorrectLayouts(ref_call, Array<Layout>(nullptr), old_prd, types);
  old_cur = infer_out->input_layouts;
  old_out = infer_out->output_layouts;
  if (!success) {
    return Expr(nullptr);
  }
  ICHECK_EQ(old_cur.size(), new_prd.size());

  // for backward compatibility of InferCorrectLayouts
  Array<Layout> new_prd_inferred = new_prd;
  // if new_prd_inferred == 'undef':  new_prd_inferred = old_cur
  for (size_t i = 0; i < new_prd_inferred.size(); ++i) {
    if (!new_prd_inferred[i].defined()) {
      new_prd_inferred.Set(i, old_cur[i]);
    }
  }
  Array<Layout> old_prd_inferred = old_prd;
  // if old_prd_inferred == 'undef':  old_prd_inferred = old_cur
  for (size_t i = 0; i < old_prd_inferred.size(); ++i) {
    if (!old_prd_inferred[i].defined()) {
      old_prd_inferred.Set(i, old_cur[i]);
    }
  }

  // new_op = alter(op)
  Call new_call = memorizer->CallWithNewLayouts(ref_call, infer_out->new_attrs, normal_new_args);

  // new_cur, new_out = op.infer(new_prd)
  if (new_call->op->IsInstance<OpNode>()) {
    success = false;
    std::tie(infer_out, success) =
        InferCorrectLayouts(new_call, new_prd_inferred, old_prd_inferred, types);
    new_cur = infer_out->input_layouts;
    new_out = infer_out->output_layouts;
    if (!success) {
      return Expr(nullptr);
    }
  } else {
    return Expr(nullptr);
  }

  ICHECK_EQ(new_out.size(), old_out.size())
      << "The number of output nodes should keep the same during alter_op_layout";
  ICHECK_EQ(new_prd.size(), new_cur.size())
      << "The number of input nodes should keep the same during alter_op_layout";

  auto transform_layout = [&memorizer](Expr arg_item, const Layout& old_prd, const Layout& old_cur,
                                       const Layout& new_prd, const Layout& new_cur) {
    if (old_cur.Equals(old_prd)) {  // the two transforms can be fused to one
      arg_item = memorizer.Transform(arg_item, new_prd, new_cur);
    } else {
      if (old_prd.defined()) arg_item = memorizer.Transform(arg_item, new_prd, old_prd);
      arg_item = memorizer.Transform(arg_item, old_cur, new_cur);
    }
    return arg_item;
  };

  DLOG(INFO) << "Transforming layout for `" << ref_call->op << "`";
  DLOG(INFO) << " old_prd=" << old_prd;
  DLOG(INFO) << " new_prd=" << new_prd;
  DLOG(INFO) << " old_cur=" << old_cur;
  DLOG(INFO) << " new_cur=" << new_cur;

  // if (new_prd != new_cur): insert transform (new_prd -> new_cur)
  Array<Expr> transformed_args;
  size_t pt = 0;
  for (auto arg : new_call->args) {
    if (arg->IsInstance<TupleNode>()) {  // unflatten tuple
      Tuple tuple_arg = Downcast<Tuple>(arg);
      Array<Expr> transformed_tuple_arg;
      transformed_tuple_arg.reserve(tuple_arg->fields.size());
      for (auto arg_item : tuple_arg->fields) {
        transformed_tuple_arg.push_back(
            transform_layout(arg_item, old_prd[pt], old_cur[pt], new_prd[pt], new_cur[pt]));
        pt++;
      }
      transformed_args.push_back(WithFields(tuple_arg, transformed_tuple_arg));
    } else {
      transformed_args.push_back(
          transform_layout(arg, old_prd[pt], old_cur[pt], new_prd[pt], new_cur[pt]));
      pt++;
    }
  }
  ICHECK_EQ(pt, inputs.size());

  // state[node] = (old_out, new_out)
  // (handle tuple output)
  if (ref_call->checked_type()->IsInstance<TupleTypeNode>()) {
    Expr tuple_output = Call(new_call->op, transformed_args, infer_out->new_attrs);
    Array<Expr> fields;
    for (size_t i = 0; i < new_out.size(); ++i) {
      auto rnode = make_object<LayoutAlternatedExprNode<TransformMemorizerT>>();
      rnode->value = TupleGetItem(tuple_output, i);
      rnode->old_layout = old_out[i];
      rnode->new_layout = new_out[i];
      rnode->memorizer = memorizer;
      fields.push_back(Expr(rnode));
    }
    return Tuple(fields);
  } else {
    auto rnode = make_object<LayoutAlternatedExprNode<TransformMemorizerT>>();
    ICHECK_EQ(new_out.size(), 1);
    rnode->value = Call(new_call->op, transformed_args, infer_out->new_attrs, {}, ref_call->span);
    rnode->old_layout = old_out[0];
    rnode->new_layout = new_out[0];
    rnode->memorizer = memorizer;
    return Expr(rnode);
  }
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_TRANSFORM_LAYOUT_H_
