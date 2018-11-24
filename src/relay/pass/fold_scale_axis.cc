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
#include "pass_util.h"
#include "../op/layout.h"


namespace tvm {
namespace relay {
/*!
 * \brief namespace of fold scale axis
 *
 * Use namespace to reduce potential naming conflict.
 */
namespace fold_scale_axis {

using runtime::TypedPackedFunc;


// FoldScaleAxis algorithm:
//
// The general idea is to transform Expr to tuple of
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
// Forward folding process is done in two steps:
// - Prepare phase: backward propagation of demand.
// - Transform phase: forward transformation,
//
// Similarly, backward folding process is done in two steps:
// - Prepare phase: forward propagation of demand.
// - Transform phase: transformation by push down the axes scale signal to inputs.
//

/*!
 * \brief sorted array axis, can also be nullptr.
 *
 *  nullptr means no scaling request can be done.
 */
using AxesSet = Array<Integer>;

/*!
 * \brief Merge two axis set together by taking
 *  intersection.
 *
 * \note The axes in a AxesSet should be sorted.
 *
 * \param lhs The left axis.
 * \param rhs The right axis.
 * \return The result of the inersection.
 */
AxesSet Intersect(const AxesSet& lhs, const AxesSet& rhs) {
  if (!lhs.defined()) return lhs;
  if (!rhs.defined()) return rhs;
  // This code relies on axes in a AxesSet to be sorted.
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
 * \brief Preparation function for pass scale forward.
 * \param call The call node.
 * \param out_scale_axes Possible scaling on axes of the output.
 * \return The result scaling on axes of the input.
 */
using FForwardPrep = runtime::TypedPackedFunc<
  Array<AxesSet> (const Call& call, const AxesSet& out_scale_axes)>;

/*! \brief Axis scale tuple.  */
class ScaledExprNode : public TempExprNode {
 public:
  /*! \brief The value */
  Expr value;
  /*! \brief The axes to scale, can be nullptr(means no-scaling) */
  AxesSet axes = NullValue<AxesSet>();
  /*! \brief The scaling factor */
  Expr scale = NullValue<Expr>();

  Expr Realize() const final {
    CHECK(!axes.defined())
        << "outstanding scale";
    return value;
  }

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("value", &value);
    v->Visit("axes", &axes);
    v->Visit("scale", &scale);
  }

  static constexpr const char* _type_key = "relay.fold_scale_axis.ScaledExpr";
  TVM_DECLARE_NODE_TYPE_INFO(ScaledExprNode, TempExprNode);
};

using FForwardRewrite = TypedPackedFunc<
  Expr(const Call& ref_call,
       const Array<Expr>& new_args,
       const AxesSet& expeced_out_axes)>;

//----------------------------------------------
// Generic Visitors for FScaleAxisForward
//----------------------------------------------
class ForwardPrep : private ExprVisitor {
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
    // We run intersection of messages:
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
      auto f = fprep.get(call->op, nullptr);
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
    // do pass through condition
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

//----------------------------------------------
// Per operator defs for FScaleAxisForward
//----------------------------------------------

// Intermediate operators
Array<AxesSet> ReluForwardPrep(const Call& call, AxesSet out) {
  return {out};
}

Expr ReluForwardRewrite(const Call& ref_call,
                        const Array<Expr>& new_args,
                        const AxesSet& expected_axes) {
  const auto* input = new_args[0].as<ScaledExprNode>();
  if (input == nullptr) return Expr(nullptr);
  // return transformed conv2d
  auto rnode = make_node<ScaledExprNode>();
  rnode->value = CallNode::make(
      ref_call->op, {input->value}, ref_call->attrs, ref_call->type_args);
  rnode->scale = input->scale;
  rnode->axes = input->axes;
  return Expr(rnode);
}

RELAY_REGISTER_OP("nn.relu")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", ReluForwardPrep);

RELAY_REGISTER_OP("nn.relu")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", ReluForwardRewrite);

RELAY_REGISTER_OP("nn.leaky_relu")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", ReluForwardPrep);

RELAY_REGISTER_OP("nn.leaky_relu")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", ReluForwardRewrite);

// AddSub
Array<AxesSet> AddSubForwardPrep(const Call& call, AxesSet out_axes) {
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();

  auto none = NullValue<AxesSet>();
  if (MatchBroadcastToLeftAxes(tlhs, trhs, out_axes)) {
    return {out_axes, none};
  } else if (MatchBroadcastToLeftAxes(trhs, tlhs, out_axes)) {
    return {none, out_axes};
  } else {
    return {none, none};
  }
}

Expr AddSubForwardRewrite(const Call& ref_call,
                          const Array<Expr>& new_args,
                          const AxesSet& expected_out_axes) {
  const auto* slhs = new_args[0].as<ScaledExprNode>();
  const auto* srhs = new_args[1].as<ScaledExprNode>();
  if (!slhs && !srhs) return Expr();
  const auto* tlhs = ref_call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = ref_call->args[1]->type_as<TensorTypeNode>();
  auto rnode = make_node<ScaledExprNode>();

  if (slhs != nullptr) {
    CHECK(srhs == nullptr);
    CHECK(MatchBroadcastToLeftAxes(tlhs, trhs, slhs->axes));
    Expr scale = ExpandBiasToMatchAxis(
        slhs->scale, tlhs->shape.size(), slhs->axes);
    Expr rhs = Divide(new_args[1], scale);
    rnode->value = CallNode::make(ref_call->op, {slhs->value, rhs},
                                  ref_call->attrs, ref_call->type_args);
    rnode->scale = slhs->scale;
    rnode->axes = slhs->axes;
  } else {
    CHECK(slhs != nullptr);
    CHECK(MatchBroadcastToLeftAxes(trhs, tlhs, srhs->axes));
    Expr scale = ExpandBiasToMatchAxis(
        srhs->scale, trhs->shape.size(), srhs->axes);
    Expr lhs = Divide(new_args[0], scale);
    rnode->value = CallNode::make(ref_call->op, {lhs, srhs->value},
                                  ref_call->attrs, ref_call->type_args);
    rnode->scale = srhs->scale;
    rnode->axes = srhs->axes;
  }
  return Expr(rnode);
}

RELAY_REGISTER_OP("add")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", AddSubForwardPrep);

RELAY_REGISTER_OP("add")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", AddSubForwardRewrite);

RELAY_REGISTER_OP("subtract")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", AddSubForwardPrep);

RELAY_REGISTER_OP("subtract")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", AddSubForwardRewrite);

// Producer operators
// Multiply produces the scale-axis pair.
Expr MultiplyForwardRewrite(const Call& ref_call,
                            const Array<Expr>& new_args,
                            const AxesSet& expected_out_axes) {
  if (!expected_out_axes.defined()) return Expr();
  if (expected_out_axes.size() == 0) return Expr();
  // TODO(tvm-team) allow same axes accumulation
  // not as important because it is less common in nn.
  const auto* slhs = new_args[0].as<ScaledExprNode>();
  const auto* srhs = new_args[1].as<ScaledExprNode>();
  CHECK(!slhs && !srhs);

  const auto* tlhs = ref_call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = ref_call->args[1]->type_as<TensorTypeNode>();
  Expr lhs = new_args[0];
  Expr rhs = new_args[1];
  auto rnode = make_node<ScaledExprNode>();
  if (MatchBroadcastToLeftAxes(tlhs, trhs, expected_out_axes, &rhs)) {
    rnode->value = lhs;
    rnode->scale = rhs;
    rnode->axes = expected_out_axes;
  } else if (MatchBroadcastToLeftAxes(trhs, tlhs, expected_out_axes, &lhs)) {
    rnode->value = rhs;
    rnode->scale = lhs;
    rnode->axes = expected_out_axes;
  }
  return Expr(rnode);
}

RELAY_REGISTER_OP("multiply")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", MultiplyForwardRewrite);

// Consumer operators
// Conv2D send out requirement of axis folding.
Array<AxesSet> Conv2DForwardPrep(const Call& call, AxesSet out) {
  // TODO(tvm-team) support general data layout
  // by transforming weight
  const auto* param = call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout weight_layout(param->weight_layout);
  int c_big_axis = data_layout.Indexof('C');
  int c_small_axis = data_layout.Indexof('c');

  CHECK_GE(c_big_axis, 0);
  AxesSet data_axes = NullValue<AxesSet>();
  // For now, we only support simple pattern (no folded weight/data)
  // More general layout can be supported under the current framework.
  // By using a unified layout transformation.
  // We only need to change the Prep and Mutate function.
  //
  // only handle depthwise or full conv2d.
  // TODO(tvm-team) handle grouped conv by reshape + bcast
  bool is_depthwise_conv2d = IsDepthwiseConv2D(call, param, weight_layout);
  if (weight_layout.Indexof('i') < 0 &&
      c_small_axis < 0 &&
      (param->groups == 1 || is_depthwise_conv2d)) {
    data_axes = {c_big_axis};
  }
  return {data_axes, NullValue<AxesSet>()};
}

// Conv2D consumes the scale axis during transformation.
Expr Conv2DForwardRewrite(const Call& ref_call,
                          const Array<Expr>& new_args,
                          const AxesSet& expected_axes) {
  // if data do not have scale, normal transform path.
  const auto* sdata = new_args[0].as<ScaledExprNode>();
  const auto* sweight = new_args[1].as<ScaledExprNode>();
  if (sdata == nullptr) return Expr();
  if (sweight != nullptr) return Expr();
  const auto* param = ref_call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout weight_layout(param->weight_layout);
  int c_big_axis = data_layout.Indexof('C');
  CHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // TODO(tvm-team) support general data layout
  CHECK_EQ(weight_layout.Indexof('i'), -1);
  CHECK(sdata->axes.size() == 1 &&
        c_big_axis == sdata->axes[0]->value);
  int big_oc_axis = weight_layout.Indexof('O');
  int big_ic_axis = weight_layout.Indexof('I');

  // Check it must be depthwise or full conv2d.
  bool is_depthwise_conv2d = IsDepthwiseConv2D(ref_call, param, weight_layout);
  CHECK(param->groups == 1 || is_depthwise_conv2d);

  Expr weight = new_args[1];

  // match the ic_axis
  if (is_depthwise_conv2d) {
    Expr scale = ExpandBiasToMatchAxis(
        sdata->scale, weight_layout.ndim(), {big_oc_axis});
    weight = Multiply(weight, scale);
  } else {
    Expr scale = ExpandBiasToMatchAxis(
        sdata->scale, weight_layout.ndim(), {big_ic_axis});
    weight = Multiply(weight, scale);
  }
  // return transformed conv2d
  return CallNode::make(
      ref_call->op, {sdata->value, weight}, ref_call->attrs, ref_call->type_args);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", Conv2DForwardPrep);

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", Conv2DForwardRewrite);


Expr ForwardFoldScaleAxis(Expr data) {
  auto expected_scale_axes =
      ForwardPrep().Prepare(data);
  auto fcontext = [&](const Call& call) -> NodeRef{
    auto it = expected_scale_axes.find(call.get());
    if (it != expected_scale_axes.end()) {
      return it->second;
    } else {
      return NodeRef(nullptr);
    }
  };
  return ForwardRewrite(
      data, "FScaleAxisForwardRewrite", fcontext);
}

// Expose the FoldScaleAxisFoward
TVM_REGISTER_API("relay._ir_pass.forward_fold_scale_axis")
.set_body_typed<Expr(Expr)>(ForwardFoldScaleAxis);

//----------------------------------------
// Implement backward transformations.
//----------------------------------------
class BackwardTransformer;

/*!
 * \brief Preparation function for for pass scale backward.
 * \param call The call node.
 * \param in_scale_axes Allowed input scaling.
 * \return The result scaling on axes of the input.
 */
using FBackwardPrep = TypedPackedFunc<
  AxesSet(const Call& call, const Array<AxesSet>& in_scale_axes)>;

using FBackwardTransform = TypedPackedFunc<
  Expr(const Call& call,
       const AxesSet& axes,
       const Expr& scale,
       const BackwardTransformer& transformer)>;

//----------------------------------------------
// Generic Visitors for FScaleAxisBackward
//----------------------------------------------

class BackwardPrep : private ExprVisitor {
 public:
  // The message on each node.
  std::unordered_map<const Node*, AxesSet>
  Prepare(const Expr& body) {
    ref_counter_ = GetExprRefCount(body);
    this->VisitExpr(body);
    return std::move(message_);
  }

 private:
  // The message on each node.
  std::unordered_map<const Node*, AxesSet> message_;
  // reference counter of an internal expr
  std::unordered_map<const Node*, size_t> ref_counter_;
  // Visit the expression.
  void VisitExpr_(const CallNode* call) {
    ExprVisitor::VisitExpr_(call);
    static const auto& fprep =
        Op::GetAttr<FBackwardPrep>("FScaleAxisBackwardPrep");
    auto f = fprep.get(call->op, nullptr);
    if (f == nullptr) return;
    auto rit = ref_counter_.find(call);
    CHECK(rit != ref_counter_.end());
    // We only allow propagation of scale backward
    // if the expression is only referred by a single parent.
    if (rit->second != 1) return;
    Array<AxesSet> in_axes;
    for (Expr arg : call->args) {
      auto it = message_.find(arg.get());
      if (it != message_.end()) {
        in_axes.push_back(it->second);
      } else {
        in_axes.push_back(NullValue<AxesSet>());
      }
    }
    AxesSet out_axes = f(GetRef<Call>(call), in_axes);
    if (out_axes.defined()) {
      message_[call] = out_axes;
    }
  }
};

class BackwardTransformerNode :
      public Node,
      private ExprMutator {
 public:
  // Run forward transform.
  Expr Fold(Expr expr) {
    expected_scale_axes_ = BackwardPrep().Prepare(expr);
    return this->Mutate(expr);
  }
  /*!
   * \brief Transform the expr to consider the scaling.
   *
   * \param expr The input expression.
   * \param axes The axes to scale.
   * \param scale The scale applied to the axes.
   * \return The result of transformation.
   */
  Expr Transform(const Expr& expr, AxesSet axes, Expr scale) {
    // NOTE: the result of Transform is not memoized.
    // However, in the current rule, Transform will
    // only be called to expr that is referred once.
    if (const CallNode* call_node = expr.as<CallNode>()) {
      return Transform(call_node, axes, scale);
    } else {
      CHECK(!axes.defined()) << "outstanding scale";
      return ExprMutator::VisitExpr(expr);
    }
  }
  /*!
   * \brief Normal way of mutating call node.
   * \param call_node The call node to be mutated.
   * \return the result of the call Mutation.
   */
  Expr NormalCallTransform(const CallNode* call_node) {
    return ExprMutator::VisitExpr_(call_node);
  }
  /*!
   * \brief Get the expected axes on expr.
   * \param expr The expresison.
   * \return The expected axes.
   */
  AxesSet GetExpectedAxes(const Expr& expr) const {
    auto it = expected_scale_axes_.find(expr.get());
    if (it != expected_scale_axes_.end()) return it->second;
    return NullValue<AxesSet>();
  }

  // solver is not serializable.
  void VisitAttrs(tvm::AttrVisitor* v) final {}

  static constexpr const char* _type_key = "relay.fold_scale_axis.FBackwardTransformer";
  TVM_DECLARE_NODE_TYPE_INFO(BackwardTransformerNode, Node);

 private:
  // Valid axes on each node.
  std::unordered_map<const Node*, AxesSet> expected_scale_axes_;
  // Override mutation of call.
  Expr VisitExpr_(const CallNode* call_node) final {
    return Transform(call_node, NullValue<AxesSet>(), NullValue<Expr>());
  }
  // Transform of CallNode.
  Expr Transform(const CallNode* call_node, AxesSet axes, Expr scale);
};

class BackwardTransformer : public NodeRef {
 public:
  BackwardTransformer() {}
  explicit BackwardTransformer(
      ::tvm::NodePtr<::tvm::Node> n) : NodeRef(n) {
  }
  BackwardTransformerNode* operator->() const {
    return static_cast<BackwardTransformerNode*>(node_.get());
  }
  using ContainerType = BackwardTransformerNode;
};

Expr BackwardTransformerNode::Transform(
    const CallNode* call_node, AxesSet axes, Expr scale) {
  static const auto& ftransform =
      Op::GetAttr<FBackwardTransform>("FScaleAxisBackwardTransform");
  auto f = ftransform.get(call_node->op, nullptr);
  if (f != nullptr) {
    return f(GetRef<Call>(call_node),
             axes,
             scale,
             GetRef<BackwardTransformer>(this));
  } else {
    CHECK(!axes.defined()) << "outstanding scale";
    return NormalCallTransform(call_node);
  }
}


//----------------------------------------------
// Per operator defs for FScaleAxisForward
//----------------------------------------------

// Intermediate operators
AxesSet ReluBackwardPrep(const Call& call, const Array<AxesSet>& in_axes) {
  return in_axes[0];
}

Expr ReluBackwardTransform(const Call& call,
                           const AxesSet& axes,
                           const Expr& scale,
                           const BackwardTransformer& transformer) {
  if (!axes.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  Expr input = transformer->Transform(
      call->args[0], axes, scale);
  return CallNode::make(call->op, {input}, call->attrs, call->type_args);
}

RELAY_REGISTER_OP("nn.relu")
.set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", ReluBackwardPrep);

RELAY_REGISTER_OP("nn.relu")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", ReluBackwardTransform);

RELAY_REGISTER_OP("nn.leaky_relu")
.set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", ReluBackwardPrep);

RELAY_REGISTER_OP("nn.leaky_relu")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", ReluBackwardTransform);

// AddSub
AxesSet AddSubBackwardPrep(const Call& call, const Array<AxesSet>& in_axes) {
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  AttrsEqual equal;
  if (in_axes[0].defined() &&
      MatchBroadcastToLeftAxes(tlhs, trhs, in_axes[0])) {
    return in_axes[0];
  } else if (in_axes[1].defined() &&
             MatchBroadcastToLeftAxes(trhs, tlhs, in_axes[1])) {
    return in_axes[1];
  } else if (in_axes[0].defined() &&
             in_axes[1].defined() &&
             equal(in_axes[0], in_axes[1]) &&
             equal(tlhs->shape, trhs->shape)) {
    // add of two elements.
    return in_axes[0];
  } else {
    auto res = NullValue<AxesSet>();
    CHECK(!res.defined());
    return res;
  }
}

Expr AddSubBackwardTransform(const Call& call,
                             const AxesSet& axes,
                             const Expr& scale,
                             const BackwardTransformer& transformer) {
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  if (!axes.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  AxesSet lhs_axes = transformer->GetExpectedAxes(call->args[0]);
  AxesSet rhs_axes = transformer->GetExpectedAxes(call->args[1]);
  AttrsEqual equal;

  if (lhs_axes.defined() && rhs_axes.defined()) {
    CHECK(equal(lhs_axes, rhs_axes));
    CHECK(equal(axes, lhs_axes));
    Expr lhs = transformer->Transform(call->args[0], axes, scale);
    Expr rhs = transformer->Transform(call->args[1], axes, scale);
    return CallNode::make(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else if (lhs_axes.defined()) {
    CHECK(equal(axes, lhs_axes));
    Expr lhs = transformer->Transform(call->args[0], axes, scale);
    Expr rhs = transformer->Transform(
        call->args[1], NullValue<AxesSet>(), NullValue<Expr>());
    Expr rhs_scale = ExpandBiasToMatchAxis(
        scale, tlhs->shape.size(), axes);
    rhs = Multiply(rhs, rhs_scale);
    return CallNode::make(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else if (rhs_axes.defined()) {
    CHECK(equal(axes, rhs_axes));
    Expr lhs = transformer->Transform(
        call->args[0], NullValue<AxesSet>(), NullValue<Expr>());
    Expr rhs = transformer->Transform(call->args[1], axes, scale);
    Expr lhs_scale = ExpandBiasToMatchAxis(
        scale, trhs->shape.size(), axes);
    lhs = Multiply(lhs, lhs_scale);
    return CallNode::make(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else {
    LOG(FATAL) << "outstanding scale";
    return Expr();
  }
}

RELAY_REGISTER_OP("add")
.set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", AddSubBackwardPrep);

RELAY_REGISTER_OP("add")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", AddSubBackwardTransform);

RELAY_REGISTER_OP("subtract")
.set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", AddSubBackwardPrep);

RELAY_REGISTER_OP("subtract")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", AddSubBackwardTransform);

// Producer operators
// Multiply produces the scale-axis pair.
Expr MultiplyBackwardTransform(const Call& call,
                               const AxesSet& axes,
                               const Expr& scale,
                               const BackwardTransformer& transformer) {
  CHECK(!axes.defined()) << "outstanding scale";
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  AxesSet lhs_axes = transformer->GetExpectedAxes(call->args[0]);
  AxesSet rhs_axes = transformer->GetExpectedAxes(call->args[1]);
  if (lhs_axes.defined() && lhs_axes.size() != 0) {
    // NOTE we won't recursively call mutating on scale part.
    // since there  won't be scale chance within scale part.
    Expr rhs = call->args[1];
    if (MatchBroadcastToLeftAxes(tlhs, trhs, lhs_axes, &rhs)) {
      return transformer->Transform(call->args[0], lhs_axes, rhs);
    }
  } else if (rhs_axes.defined() && rhs_axes.size() != 0) {
    Expr lhs = call->args[0];
    if (MatchBroadcastToLeftAxes(trhs, tlhs, rhs_axes, &lhs)) {
      return transformer->Transform(call->args[1], rhs_axes, lhs);
    }
  }
  return transformer->NormalCallTransform(call.operator->());
}

RELAY_REGISTER_OP("multiply")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", MultiplyBackwardTransform);

// Consumer operators
// Conv2D send out requirement of axis folding.
AxesSet Conv2DBackwardPrep(const Call& call, const Array<AxesSet>& in_axes) {
  const auto* param = call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout out_layout(param->out_layout);
  if (!out_layout.defined()) {
    out_layout = Layout(param->data_layout);
  }
  Layout weight_layout(param->weight_layout);
  int c_big_axis = out_layout.Indexof('C');
  int c_small_axis = out_layout.Indexof('c');

  CHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // More general layout can be supported under the current framework.
  // By using a unified layout transformation.
  // We only need to change the Prep and Mutate function.
  //
  // only handle depthwise or full conv2d.
  // TODO(tvm-team) handle grouped conv by reshape + bcast
  bool is_depthwise_conv2d = IsDepthwiseConv2D(call, param, weight_layout);
  if (weight_layout.Indexof('o') < 0 &&
      weight_layout.Indexof('i') < 0 &&
      c_small_axis < 0 &&
      (param->groups == 1 || is_depthwise_conv2d)) {
    return {c_big_axis};
  } else {
    return NullValue<AxesSet>();
  }
}

// Conv2D consumes the scale axis during transformation.
Expr Conv2DBackwardTransform(const Call& call,
                             const AxesSet& axes,
                             const Expr& scale,
                             const BackwardTransformer& transformer) {
  if (!axes.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  const auto* param = call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout out_layout(param->out_layout);
  if (!out_layout.defined()) {
    out_layout = Layout(param->data_layout);
  }
  Layout weight_layout(param->weight_layout);
  int c_big_axis = out_layout.Indexof('C');
  CHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // TODO(tvm-team) support general data layout
  CHECK_EQ(weight_layout.Indexof('o'), -1);
  CHECK_EQ(weight_layout.Indexof('i'), -1);
  CHECK(axes.size() == 1 &&
        c_big_axis == axes[0]->value);

  int big_oc_axis = weight_layout.Indexof('O');
  // Check it must be depthwise or full conv2d.
  bool is_depthwise_conv2d = IsDepthwiseConv2D(call, param, weight_layout);
  CHECK(param->groups == 1 || is_depthwise_conv2d);

  Expr data = transformer->Transform(
      call->args[0], NullValue<AxesSet>(), NullValue<Expr>());
  Expr weight = transformer->Transform(
      call->args[1], NullValue<AxesSet>(), NullValue<Expr>());
  // scale on input for deptwise.
  Expr wscale = ExpandBiasToMatchAxis(
      scale, weight_layout.ndim(), {big_oc_axis});
  weight = Multiply(weight, wscale);
  return CallNode::make(
      call->op, {data, weight}, call->attrs, call->type_args);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", Conv2DBackwardPrep);

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", Conv2DBackwardTransform);

Expr BackwardFoldScaleAxis(Expr data) {
  return make_node<BackwardTransformerNode>()->Fold(data);
}

TVM_REGISTER_API("relay._ir_pass.backward_fold_scale_axis")
.set_body_typed<Expr(Expr)>(BackwardFoldScaleAxis);

}  // namespace fold_scale_axis
}  // namespace relay
}  // namespace tvm
