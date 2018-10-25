/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file fold_scale_axis.cc
 *
 * \brief Fold axis scaling into weights of
 *  conv/dense operators.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include "pattern_util.h"
#include "../op/nn/layout.h"

namespace tvm {
namespace relay {
/*!
 * \brief namespace of fold scale axis
 *
 * Use namespace to reduce potential naming conflict.
 */
namespace fold_scale_axis {

using runtime::TypedPackedFunc;


// FoldScaleAxisFoward algorithm:
//
// The general idea is that we transform Expr to tuple of
// (value, axes, scale), where the final result satiesfies:
//
// result = value
// for i, k in enumerate(axes):
//    k-ith dimension of result *= i-th dimension of scale
//
// Then we can propagate this signal along and fold the scale if necessary.
// However, it is possible that certain scale may never be consumed
// if there is no dense/conv2d that follows multiplication.
//
// In order to make sure all the scale we sent out can be consumed eventually,
// we run a backward "preparation phase", which propagates the demand
// of the potential axes scaling back to its input.
//
// The folding process is done in two steps:
// - Prepare phase: backward propagation of demand.
// - Transform phase: forward transformation,

/*!
 * \brief sorted axis, can also be nullptr.
 *
 *  nullptr means no scaling request can be done.
 */
using AxesSet = Array<Integer>;

/*!
 * \brief Merge two axis set together by taking
 *  intersection.
 *
 * \param lhs The left axis.
 * \param rhs The right axis.
 * \return The result of the inersection.
 */
AxesSet Intersect(const AxesSet& lhs, const AxesSet& rhs) {
  if (!lhs.defined()) return lhs;
  if (!rhs.defined()) return rhs;
  AxesSet ret;
  size_t i = 0, j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i]->value < rhs[j]->value) {
      ++i;
    } else if (lhs[i]->value > rhs[j]->value) {
      ++j;
    } else {
      ret.push_back(lhs[i]);
      ++i; ++j;
    }
  }
  return ret;
}

/*!
 * \param Get function from op_map.
 * \param op_map The OpMap.
 * \param op The operator being called.
 * \tparam ValueType the content value type.
 * \return The result value map.
 */
template<typename ValueType>
ValueType GetFunc(const OpMap<ValueType>& op_map,
                  const Expr& op) {
  if (const OpNode* opnode = op.as<OpNode>()) {
    return op_map.get(GetRef<Op>(opnode), ValueType());
  } else {
    return ValueType();
  }
}

/*!
 * \brief Preparation function for for pass scale forward.
 * \param call The call node.
 * \param out_scale_axes Possible scaling on axes of the output.
 * \return The result scaling on axes of the input.
 */
using FForwardPrep = runtime::TypedPackedFunc<
  Array<AxesSet> (const Call& call, const AxesSet& out_scale_axes)>;

/*! \brief Axis scale tuple.  */
class STupleNode : public Node {
 public:
  /*! \brief The value */
  Expr value;
  /*! \brief The axes to scale, can be nullptr(means no-scaling) */
  AxesSet axes = NullValue<AxesSet>();
  /*! \brief The scaling factor */
  Expr scale = NullValue<Expr>();

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("value", &value);
    v->Visit("axes", &axes);
    v->Visit("scale", &scale);
  }

  static constexpr const char* _type_key = "relay.fold_scale_axis.STupleNode";
  TVM_DECLARE_NODE_TYPE_INFO(STupleNode, Node);
};

RELAY_DEFINE_NODE_REF(STuple, STupleNode, NodeRef);

/*!
 * \brief The transform function, transform an old call to
 *  new one given the new args.
 * \param ref_call Reference call node that represent the op and the types.
 * \param expected_out_axes The scale axes allowed in the output.
 * \param sargs The input arguments.
 */
using FForwardTransform = TypedPackedFunc<
  STuple(const Call& ref_call,
         const AxesSet& expected_out_axes,
         const Array<STuple>& sargs)>;

//----------------------------------------------
// Generic Visitors for FScaleAxisForward
//----------------------------------------------
class FScaleAxisForwardPrep : private ExprVisitor {
 public:
  std::unordered_map<const Node*, AxesSet>
  Prepare(const Expr& body) {
    this->Update(body, NullValue<AxesSet>());
    this->VisitExpr(body);
    // flist is added in the Post-DFS order
    // which is a special case of topological order.
    // We reversely traverse the list to invoke the lazy functions.
    // This act like a backprop of valid scale axis messages
    for (auto it = flist_.rbegin(); it != flist_.rend(); ++it) {
      (*it)();
    }
    // return the created message;
    return std::move(message_);
  }

 private:
  // The invoke list
  std::vector<std::function<void()> > flist_;
  // The message on each node.
  std::unordered_map<const Node*, AxesSet> message_;
  // Update the message stored at node.
  void Update(const Expr& node, const AxesSet& axes) {
    // We run interection of messages:
    //
    // %y = multiply(%x, %scale)
    // %z1 = conv2d(%y, %w)
    // %z2 = exp(%y)
    //
    // Consider the above code example,
    // because %z2 will propagate null to %y,
    // the AxesSet on %y is also null,
    // and the forward folding won't be triggered.
    const Node* key = node.get();
    if (message_.count(key)) {
      message_[key] = Intersect(message_[key], axes);
    } else {
      message_[key] = axes;
    }
  }
  // Visitor pattern override.
  void VisitExpr_(const LetNode* call) {
    LOG(FATAL) << "FoldScaleAxis only accept dataflow-form";
  }

  void VisitExpr_(const FunctionNode* op) {
    ExprVisitor::VisitExpr_(op);
    auto flazy = [this, op] {
      this->Update(op->body, NullValue<AxesSet>());
    };
    flist_.push_back(flazy);
  }

  void VisitExpr_(const TupleGetItemNode* op) {
    // pass, do nothing
  }

  void VisitExpr_(const VarNode* op) {
    // pass, do nothing.
  }

  void VisitExpr_(const CallNode* call) {
    ExprVisitor::VisitExpr_(call);
    // function to be lazily invoked
    auto flazy = [this, call]() {
      static const auto& fprep =
        Op::GetAttr<FForwardPrep>("FScaleAxisForwardPrep");
      // find the message send to this node.
      auto it = message_.find(call);
      AxesSet out_axes;
      if (it != message_.end()) {
        out_axes = it->second;
      } else {
        out_axes = NullValue<AxesSet>();
      }
      // pass the message back to all the children it references.
      auto f = GetFunc(fprep, call->op);
      if (f != nullptr) {
        Array<AxesSet> in_axes = f(GetRef<Call>(call), out_axes);
        CHECK_EQ(in_axes.size(), call->args.size());
        for (size_t i = 0; i < call->args.size(); ++i) {
          this->Update(call->args[i], in_axes[i]);
        }
      } else {
        for (size_t i = 0; i < call->args.size(); ++i) {
          this->Update(call->args[i], NullValue<AxesSet>());
        }
      }
    };
    flist_.push_back(flazy);
  }

  void VisitExpr_(const TupleNode* op) {
    ExprVisitor::VisitExpr_(op);
    // do not support pass scale through tuple for now.
    auto flazy = [this, op]() {
      for (const Expr& field : op->fields) {
        this->Update(field, NullValue<AxesSet>());
      }
    };
    flist_.push_back(flazy);
  }

  void VisitExpr_(const IfNode* op) {
    ExprVisitor::VisitExpr_(op);
    // do pass through condition.
    // by assigning NullValue<AxesSet>
    // it means fuse signal cannot pass
    // through into these subexpressions.
    auto flazy = [this, op]() {
      this->Update(op->cond, NullValue<AxesSet>());
      this->Update(op->true_branch, NullValue<AxesSet>());
      this->Update(op->false_branch, NullValue<AxesSet>());
    };
    flist_.push_back(flazy);
  }
};

class FScaleAxisForwardTransform : private ExprMutator {
 public:
  // Transform expression.
  Expr Transform(Expr expr) {
    expected_scale_axes_ =
        FScaleAxisForwardPrep().Prepare(expr);
    return this->Mutate(expr);
  }

 private:
  // Valid axes on each node.
  std::unordered_map<const Node*, AxesSet> expected_scale_axes_;
  std::unordered_map<const Node*, STuple> scale_memo_;
  // If user simply call mutate,
  // then only Expr is returned and we cannot
  // accept outstanding scales.
  Expr VisitExpr(const Expr& expr) final {
    Expr res = ExprMutator::VisitExpr(expr);
    CHECK(!scale_memo_.count(expr.get()))
        << "Outstanding scale";
    return res;
  }

  STuple GetSTuple(const Expr& expr) {
    Expr res = ExprMutator::VisitExpr(expr);
    auto it = scale_memo_.find(expr.get());
    if (it != scale_memo_.end()) {
      CHECK(it->second->value.same_as(res));
      return it->second;
    } else {
      auto node = make_node<STupleNode>();
      node->value = res;
      return STuple(node);
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    static const auto& ftransform =
        Op::GetAttr<FForwardTransform>("FScaleAxisForwardTransform");
    auto new_op = this->Mutate(call_node->op);
    bool has_scale = false;
    bool unchanged = call_node->op.same_as(new_op);

    Array<STuple> call_sargs;
    Array<Expr> call_args;
    for (auto arg : call_node->args) {
      STuple new_sarg = this->GetSTuple(arg);
      unchanged &= new_sarg->value.same_as(arg);
      if (new_sarg->axes.defined()) has_scale = true;
      call_sargs.push_back(new_sarg);
      call_args.push_back(new_sarg->value);
    }

    // get expected scale axes.
    AxesSet expected_out_axes;
    auto axis_it = expected_scale_axes_.find(call_node);
    if (axis_it != expected_scale_axes_.end()) {
      expected_out_axes = axis_it->second;
    }
    // propagation function
    auto f = GetFunc(ftransform, call_node->op);
    if (f != nullptr) {
      STuple sret = f(GetRef<Call>(call_node), expected_out_axes, call_sargs);
      if (sret.defined()) {
        if (sret->axes.defined()) {
          scale_memo_[call_node] = sret;
        }
        return sret->value;
      }
    }
    // normal path
    CHECK(!has_scale) << "Outstanding scale, on op=" << call_node->op;
    if (unchanged) {
      return GetRef<Expr>(call_node);
    } else {
      return CallNode::make(
          new_op, call_args, call_node->attrs, call_node->type_args);
    }
  }
};

//----------------------------------------------
// Per operator defs for FScaleAxisForward
//----------------------------------------------

// Intermediate operators
Array<AxesSet> ReluForwardPrep(const Call& call, AxesSet out) {
  return {out};
}

STuple ReluForwardTransform(const Call& ref_call,
                              const AxesSet& expected_axes,
                              const Array<STuple>& sargs) {
  if (!sargs[0]->axes.defined()) return STuple();
  // return transformed conv2d
  auto rnode = make_node<STupleNode>();
  rnode->value = CallNode::make(
      ref_call->op, {sargs[0]->value}, ref_call->attrs, {});
  rnode->scale = sargs[0]->scale;
  rnode->axes = sargs[0]->axes;
  return STuple(rnode);
}

RELAY_REGISTER_OP("nn.relu")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", ReluForwardPrep);

RELAY_REGISTER_OP("nn.relu")
.set_attr<FForwardTransform>("FScaleAxisForwardTransform", ReluForwardTransform);

RELAY_REGISTER_OP("nn.leaky_relu")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", ReluForwardPrep);

RELAY_REGISTER_OP("nn.leaky_relu")
.set_attr<FForwardTransform>("FScaleAxisForwardTransform", ReluForwardTransform);

// AddSub
Array<AxesSet> AddSubForwardPrep(const Call& call, AxesSet out_axes) {
  const auto* tlhs = call->args[0]->checked_type().as<TensorTypeNode>();
  const auto* trhs = call->args[1]->checked_type().as<TensorTypeNode>();
  auto none = NullValue<AxesSet>();
  if (MatchBroadcastToLeftAxes(tlhs, trhs, out_axes)) {
    return {out_axes, none};
  } else if (MatchBroadcastToLeftAxes(trhs, tlhs, out_axes)) {
    return {none, out_axes};
  } else {
    return {none, none};
  }
}

STuple AddSubForwardTransform(const Call& ref_call,
                              const AxesSet& expected_out_axes,
                              const Array<STuple>& sargs) {
  if (!sargs[0]->axes.defined() && !sargs[1]->axes.defined()) {
    return STuple();
  }
  const auto* tlhs = ref_call->args[0]->checked_type().as<TensorTypeNode>();
  const auto* trhs = ref_call->args[1]->checked_type().as<TensorTypeNode>();

  auto rnode = make_node<STupleNode>();

  if (sargs[0]->axes.defined()) {
    CHECK(!sargs[1]->axes.defined());
    CHECK(MatchBroadcastToLeftAxes(tlhs, trhs, sargs[0]->axes));
    Expr scale = ExpandBiasToMatchAxis(
        sargs[0]->scale, tlhs->shape.size(), sargs[0]->axes);
    Expr rhs = Divide(sargs[1]->value, scale);
    rnode->value = CallNode::make(ref_call->op, {sargs[0]->value, rhs},
                                  ref_call->attrs, ref_call->type_args);
    rnode->scale = sargs[0]->scale;
    rnode->axes = sargs[0]->axes;
  } else {
    CHECK(sargs[1]->axes.defined());
    CHECK(sargs[0]->axes.defined());
    CHECK(MatchBroadcastToLeftAxes(trhs, tlhs, sargs[1]->axes));
    Expr scale = ExpandBiasToMatchAxis(
        sargs[1]->scale, trhs->shape.size(), sargs[1]->axes);
    Expr lhs = Divide(sargs[0]->value, scale);
    rnode->value = CallNode::make(ref_call->op, {lhs, sargs[1]->value},
                                  ref_call->attrs, ref_call->type_args);
    rnode->scale = sargs[1]->scale;
    rnode->axes = sargs[1]->axes;
  }
  return STuple(rnode);
}

RELAY_REGISTER_OP("add")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", AddSubForwardPrep);

RELAY_REGISTER_OP("add")
.set_attr<FForwardTransform>("FScaleAxisForwardTransform", AddSubForwardTransform);

RELAY_REGISTER_OP("subtract")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", AddSubForwardPrep);

RELAY_REGISTER_OP("subtract")
.set_attr<FForwardTransform>("FScaleAxisForwardTransform", AddSubForwardTransform);

// Producer operators
// Multiply produces the scale-axis pair.
STuple MultiplyForwardTransform(const Call& ref_call,
                                const AxesSet& expected_out_axes,
                                const Array<STuple>& sargs) {
  if (!expected_out_axes.defined()) return STuple();
  // TODO(tvm-team) allow same axes accumulation
  // not as important because it is less common in nn.
  CHECK(!sargs[0]->axes.defined());
  CHECK(!sargs[1]->axes.defined());
  const auto* tlhs = ref_call->args[0]->checked_type().as<TensorTypeNode>();
  const auto* trhs = ref_call->args[1]->checked_type().as<TensorTypeNode>();

  Expr lhs = sargs[0]->value;
  Expr rhs = sargs[1]->value;
  auto rnode = make_node<STupleNode>();
  if (MatchBroadcastToLeftAxes(tlhs, trhs, expected_out_axes, &rhs)) {
    rnode->value = lhs;
    rnode->scale = rhs;
    rnode->axes = expected_out_axes;
  } else if (MatchBroadcastToLeftAxes(trhs, tlhs, expected_out_axes, &lhs)) {
    rnode->value = rhs;
    rnode->scale = lhs;
    rnode->axes = expected_out_axes;
  }
  return STuple(rnode);
}

RELAY_REGISTER_OP("multiply")
.set_attr<FForwardTransform>("FScaleAxisForwardTransform", MultiplyForwardTransform);

// Consumer operators
// Conv2D send out requirement of axis folding.
Array<AxesSet> Conv2DForwardPrep(const Call& call, AxesSet out) {
  // TODO(tvm-team) support general data layout
  // by transforming weight
  const auto* param = call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout weight_layout(param->weight_layout);
  int c_big_axis = data_layout.indexof('C');
  int c_small_axis = data_layout.indexof('c');
  const auto* tdata = call->args[0]->checked_type().as<TensorTypeNode>();
  CHECK(tdata) << "require checked type";

  CHECK_GE(c_big_axis, 0);
  AxesSet data_axes = NullValue<AxesSet>();
  // For now, we only support simple pattern (no folded weight/data)
  // More general layout can be supported under the current framework.
  // By using a unified layout transformation.
  // We only need to change the Prep and Mutate function.
  //
  // only handle depthwise or full conv2d.
  // TODO(tvm-team) handle grouped conv by reshape + bcast
  bool is_depthwise_conv2d =
      is_const_int(tdata->shape[c_big_axis], param->groups);
  if (weight_layout.indexof('i') < 0 &&
      c_small_axis < 0 &&
      (param->groups == 1 || is_depthwise_conv2d)) {
    data_axes = {c_big_axis};
  }
  return {data_axes, NullValue<AxesSet>()};
}

// Conv2D consumes the scale axis during transformation.
STuple Conv2DForwardTransform(const Call& ref_call,
                                const AxesSet& expected_axes,
                                const Array<STuple>& sargs) {
  // if data do not have scale, normal transform path.
  STuple sdata = sargs[0];
  if (!sdata->scale.defined()) return STuple();
  CHECK(sdata->axes.defined());
  const auto* param = ref_call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout weight_layout(param->weight_layout);
  int c_big_axis = data_layout.indexof('C');
  CHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // TODO(tvm-team) support general data layout
  CHECK_EQ(weight_layout.indexof('i'), -1);
  CHECK(sdata->axes.size() == 1 &&
        c_big_axis == sdata->axes[0]->value);
  int big_ic_axis = weight_layout.indexof('I');
  const auto* tdata = ref_call->args[0]->checked_type().as<TensorTypeNode>();
  CHECK(tdata) << "require checked type";
  // Check it must be depthwise or full conv2d.
  bool is_depthwise_conv2d =
      is_const_int(tdata->shape[c_big_axis], param->groups);
  CHECK(param->groups == 1 || is_depthwise_conv2d);

  // match the ic_axis
  Expr scale = ExpandBiasToMatchAxis(
      sdata->scale, weight_layout.ndim(), {big_ic_axis});
  Expr weight = Multiply(sargs[1]->value, scale);
  // return transformed conv2d
  auto rnode = make_node<STupleNode>();
  rnode->value = CallNode::make(
      ref_call->op, {sdata->value, weight}, ref_call->attrs, ref_call->type_args);
  return STuple(rnode);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", Conv2DForwardPrep);

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardTransform>("FScaleAxisForwardTransform", Conv2DForwardTransform);


Expr ForwardFoldScaleAxis(Expr data) {
  return FScaleAxisForwardTransform().Transform(data);
}

// Expose the FoldScaleAxisFoward
TVM_REGISTER_API("relay._ir_pass.forward_fold_scale_axis")
.set_body_typed<Expr(Expr)>(ForwardFoldScaleAxis);

}  // namespace fold_scale_axis
}  // namespace relay
}  // namespace tvm
