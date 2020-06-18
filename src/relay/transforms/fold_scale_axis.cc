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
 * \file fold_scale_axis.cc
 *
 * \brief Fold axis scaling into weights of
 *  conv/dense operators.
 */
#include <tvm/tir/data_layout.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include "pattern_util.h"
#include "pass_util.h"


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
// (value, axes, scale), where the final result satisfies:
//
// result = value
// for i, k in enumerate(axes):
//    k-th dimension of result *= i-th dimension of scale
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

class Message;

/*!
 * \brief Message propogated during the prepare phase.
 */
class MessageNode : public RelayNode {
 public:
  /*! \brief Axes for scaling */
  AxesSet axes;
  /*!
   * \brief Whether folding requires the scale to be positive constant. This is necessary if some
   *  operators (e.g. Relu) is present.
   */
  bool require_positive;

  static constexpr const char* _type_key = "relay.pass.fold_scale_axis.Message";
  TVM_DECLARE_FINAL_OBJECT_INFO(MessageNode, RelayNode);
};

class Message : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param axes Axes for scaling
   * \param require_positive If folding requires the scales to be positive
   *        values.
   */
  Message(const AxesSet& axes, bool require_positive);

  TVM_DEFINE_OBJECT_REF_METHODS(Message, ObjectRef, MessageNode);
};

Message::Message(const AxesSet& axes, bool require_positive)  {
  auto n = make_object<MessageNode>();
  n->axes = axes;
  n->require_positive = require_positive;
  data_ = std::move(n);
}

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
 * \brief Merge two messages together by taking intersection.
 *
 * \param lhs The lhs message.
 * \param rhs The rhs message.
 * \return The result of intersection.
 */
Message Intersect(const Message& lhs, const Message& rhs) {
  if (!lhs.defined()) return lhs;
  if (!rhs.defined()) return rhs;
  auto axes = Intersect(lhs->axes, rhs->axes);
  return Message(axes, lhs->require_positive || rhs->require_positive);
}

/*!
 * \brief Preparation function for pass scale forward.
 * \param call The call node.
 * \param out_message Message from the output containing possible scaling on axes and whether
 *        positive scale is required.
 * \return The message containing the result scaling on axes of the input.
 */
using FForwardPrep = runtime::TypedPackedFunc<
  Array<Message> (const Call& call, const Message& out_message)>;

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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("axes", &axes);
    v->Visit("scale", &scale);
  }

  static constexpr const char* _type_key = "relay.fold_scale_axis.ScaledExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScaledExprNode, TempExprNode);
};

using FForwardRewrite = TypedPackedFunc<
  Expr(const Call& ref_call,
       const Array<Expr>& new_args,
       const Message& message)>;

//----------------------------------------------
// Generic Visitors for FScaleAxisForward
//----------------------------------------------
class ForwardPrep : private ExprVisitor {
 public:
  std::unordered_map<const Object*, Message>
  Prepare(const Expr& body) {
    this->Update(body, NullValue<Message>());
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
  std::unordered_map<const Object*, Message> message_;
  // Update the message stored at node.
  void Update(const Expr& node, const Message& message) {
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
    const Object* key = node.get();
    if (message_.count(key)) {
      message_[key] = Intersect(message_[key], message);
    } else {
      message_[key] = message;
    }
  }
  // Visitor pattern override.
  void VisitExpr_(const LetNode* call) {
    LOG(FATAL) << "FoldScaleAxis only accept dataflow-form";
  }

  void VisitExpr_(const FunctionNode* op) {
    ExprVisitor::VisitExpr_(op);
    auto flazy = [this, op] {
      this->Update(op->body, NullValue<Message>());
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
      Message out_message;
      if (it != message_.end()) {
        out_message = it->second;
      } else {
        out_message = NullValue<Message>();
      }
      // pass the message back to all the children it references.
      auto f = fprep.get(call->op, nullptr);
      if (f != nullptr) {
        Array<Message> in_messages = f(GetRef<Call>(call), out_message);
        CHECK_EQ(in_messages.size(), call->args.size());
        for (size_t i = 0; i < call->args.size(); ++i) {
          this->Update(call->args[i], in_messages[i]);
        }
      } else {
        for (size_t i = 0; i < call->args.size(); ++i) {
          this->Update(call->args[i], NullValue<Message>());
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
        this->Update(field, NullValue<Message>());
      }
    };
    flist_.push_back(flazy);
  }

  void VisitExpr_(const IfNode* op) {
    ExprVisitor::VisitExpr_(op);
    // do pass through condition
    // by assigning NullValue<Message>
    // it means fuse signal cannot pass
    // through into these subexpressions.
    auto flazy = [this, op]() {
      this->Update(op->cond, NullValue<Message>());
      this->Update(op->true_branch, NullValue<Message>());
      this->Update(op->false_branch, NullValue<Message>());
    };
    flist_.push_back(flazy);
  }
};

//----------------------------------------------
// Per operator defs for FScaleAxisForward
//----------------------------------------------

// Intermediate operators
Array<Message> ReluForwardPrep(const Call& call, const Message& out_message) {
  if (out_message.defined()) {
    return {Message(out_message->axes, true)};
  }
  return {out_message};
}

Expr ReluForwardRewrite(const Call& ref_call,
                        const Array<Expr>& new_args,
                        const Message& message) {
  const auto* input = new_args[0].as<ScaledExprNode>();
  if (input == nullptr) return Expr(nullptr);
  // return transformed conv2d
  auto rnode = make_object<ScaledExprNode>();
  rnode->value = Call(
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
Array<Message> AddSubForwardPrep(const Call& call, const Message& out_message) {
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  auto none = NullValue<Message>();
  if (out_message.defined()) {
    if (MatchBroadcastToLeftAxes(tlhs, trhs, out_message->axes)) {
      return {out_message, none};
    } else if (MatchBroadcastToLeftAxes(trhs, tlhs, out_message->axes)) {
      return {none, out_message};
    }
  }
  return {none, none};
}

Expr AddSubForwardRewrite(const Call& ref_call,
                          const Array<Expr>& new_args,
                          const Message& message) {
  const auto* slhs = new_args[0].as<ScaledExprNode>();
  const auto* srhs = new_args[1].as<ScaledExprNode>();
  if (!slhs && !srhs) return Expr();
  const auto* tlhs = ref_call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = ref_call->args[1]->type_as<TensorTypeNode>();
  auto rnode = make_object<ScaledExprNode>();

  if (slhs != nullptr) {
    CHECK(srhs == nullptr);
    CHECK(MatchBroadcastToLeftAxes(tlhs, trhs, slhs->axes));
    Expr scale = ExpandBiasToMatchAxis(
        slhs->scale, tlhs->shape.size(), slhs->axes);
    Expr rhs = Divide(new_args[1], scale);
    rnode->value = Call(ref_call->op, {slhs->value, rhs},
                                  ref_call->attrs, ref_call->type_args);
    rnode->scale = slhs->scale;
    rnode->axes = slhs->axes;
  } else {
    CHECK(srhs != nullptr);
    CHECK(MatchBroadcastToLeftAxes(trhs, tlhs, srhs->axes));
    Expr scale = ExpandBiasToMatchAxis(
        srhs->scale, trhs->shape.size(), srhs->axes);
    Expr lhs = Divide(new_args[0], scale);
    rnode->value = Call(ref_call->op, {lhs, srhs->value},
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
                            const Message& message) {
  if (!message.defined()) return Expr();
  const auto& expected_out_axes = message->axes;
  CHECK(expected_out_axes.defined() && expected_out_axes.size());
  // TODO(tvm-team) allow same axes accumulation
  // not as important because it is less common in nn.
  const auto* slhs = new_args[0].as<ScaledExprNode>();
  const auto* srhs = new_args[1].as<ScaledExprNode>();
  CHECK(!slhs && !srhs);

  const auto* tlhs = ref_call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = ref_call->args[1]->type_as<TensorTypeNode>();
  Expr lhs = new_args[0];
  Expr rhs = new_args[1];
  auto rnode = make_object<ScaledExprNode>();

  if (MatchBroadcastToLeftAxes(tlhs, trhs, expected_out_axes, &rhs) &&
      (!message->require_positive || IsAllPositiveConstant(rhs))) {
    rnode->value = lhs;
    rnode->scale = rhs;
    rnode->axes = expected_out_axes;
    return Expr(rnode);
  } else if (MatchBroadcastToLeftAxes(trhs, tlhs, expected_out_axes, &lhs) &&
             (!message->require_positive || IsAllPositiveConstant(lhs))) {
    rnode->value = rhs;
    rnode->scale = lhs;
    rnode->axes = expected_out_axes;
    return Expr(rnode);
  } else {
    return Expr();
  }
}

RELAY_REGISTER_OP("multiply")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", MultiplyForwardRewrite);

// Consumer operators
// Conv2D send out requirement of axis folding.
Array<Message> Conv2DForwardPrep(const Call& call, const Message& out_message) {
  // TODO(tvm-team) support general data layout
  // by transforming weight
  const auto* param = call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout kernel_layout(param->kernel_layout);
  int c_big_axis = data_layout.IndexOf(LayoutAxis::Get('C'));
  int c_small_axis = data_layout.IndexOf(LayoutAxis::Get('c'));

  CHECK_GE(c_big_axis, 0);
  Message none = NullValue<Message>();
  AxesSet data_axes = NullValue<AxesSet>();
  // For now, we only support simple pattern (no folded weight/data)
  // More general layout can be supported under the current framework.
  // By using a unified layout transformation.
  // We only need to change the Prep and Mutate function.
  //
  // only handle depthwise or full conv2d.
  // TODO(tvm-team) handle grouped conv by reshape + bcast
  bool is_depthwise_conv2d = IsDepthwiseConv2D(call, param, kernel_layout);
  if (kernel_layout.IndexOf(LayoutAxis::Get('i')) < 0 &&
      c_small_axis < 0 &&
      (param->groups == 1 || is_depthwise_conv2d)) {
    data_axes = {c_big_axis};
  }
  if (data_axes.defined()) {
    return {Message(data_axes, false), none};
  }
  return {none, none};
}

// Conv2D consumes the scale axis during transformation.
Expr Conv2DForwardRewrite(const Call& ref_call,
                          const Array<Expr>& new_args,
                          const Message& message) {
  // if data do not have scale, normal transform path.
  const auto* sdata = new_args[0].as<ScaledExprNode>();
  const auto* sweight = new_args[1].as<ScaledExprNode>();
  if (sdata == nullptr) return Expr();
  if (sweight != nullptr) return Expr();
  const auto* param = ref_call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout kernel_layout(param->kernel_layout);
  int c_big_axis = data_layout.IndexOf(LayoutAxis::Get('C'));
  CHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // TODO(tvm-team) support general data layout
  CHECK_EQ(kernel_layout.IndexOf(LayoutAxis::Get('i')), -1);
  CHECK(sdata->axes.size() == 1 &&
        c_big_axis == sdata->axes[0]->value);
  int big_oc_axis = kernel_layout.IndexOf(LayoutAxis::Get('O'));
  int big_ic_axis = kernel_layout.IndexOf(LayoutAxis::Get('I'));

  // Check it must be depthwise or full conv2d.
  bool is_depthwise_conv2d = IsDepthwiseConv2D(ref_call, param, kernel_layout);
  CHECK(param->groups == 1 || is_depthwise_conv2d);

  Expr weight = new_args[1];

  // match the ic_axis
  if (is_depthwise_conv2d) {
    Expr scale = ExpandBiasToMatchAxis(
        sdata->scale, kernel_layout.ndim(), {big_oc_axis});
    weight = Multiply(weight, scale);
  } else {
    Expr scale = ExpandBiasToMatchAxis(
        sdata->scale, kernel_layout.ndim(), {big_ic_axis});
    weight = Multiply(weight, scale);
  }
  // return transformed conv2d
  return Call(
      ref_call->op, {sdata->value, weight}, ref_call->attrs, ref_call->type_args);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardPrep>("FScaleAxisForwardPrep", Conv2DForwardPrep);

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", Conv2DForwardRewrite);


Expr ForwardFoldScaleAxis(const Expr& data) {
  auto message = ForwardPrep().Prepare(data);
  auto fcontext = [&](const Call& call) -> ObjectRef{
    auto it = message.find(call.get());
    if (it != message.end()) {
      return it->second;
    } else {
      return ObjectRef(nullptr);
    }
  };
  return ForwardRewrite(
      data, "FScaleAxisForwardRewrite", fcontext);
}

//----------------------------------------
// Implement backward transformations.
//----------------------------------------
class BackwardTransformer;

/*!
 * \brief Preparation function for for pass scale backward.
 * \param call The call node.
 * \param in_messages Messages from the input containing allowed input scaling and whether
 *        positive scale is required.
 * \return Message containing the result scaling on axes of the input.
 */
using FBackwardPrep = TypedPackedFunc<
  Message(const Call& call, const Array<Message>& in_messages)>;

using FBackwardTransform = TypedPackedFunc<
  Expr(const Call& call,
       const Message& message,
       const Expr& scale,
       const BackwardTransformer& transformer)>;

//----------------------------------------------
// Generic Visitors for FScaleAxisBackward
//----------------------------------------------

class BackwardPrep : private ExprVisitor {
 public:
  // The message on each node.
  std::unordered_map<const Object*, Message>
  Prepare(const Expr& body) {
    ref_counter_ = GetExprRefCount(body);
    this->VisitExpr(body);
    return std::move(message_);
  }

 private:
  // The message on each node.
  std::unordered_map<const Object*, Message> message_;
  // reference counter of an internal expr
  std::unordered_map<const Object*, size_t> ref_counter_;
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
    Array<Message> in_messages;
    for (Expr arg : call->args) {
      auto it = message_.find(arg.get());
      if (it != message_.end()) {
        in_messages.push_back(it->second);
      } else {
        in_messages.push_back(NullValue<Message>());
      }
    }
    Message out_message = f(GetRef<Call>(call), in_messages);
    if (out_message.defined()) {
      message_[call] = out_message;
    }
  }
};

class BackwardTransformerNode :
      public Object,
      private ExprMutator {
 public:
  // Run forward transform.
  Expr Fold(Expr expr) {
    message_ = BackwardPrep().Prepare(expr);
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
  Expr Transform(const Expr& expr, Message message, Expr scale) {
    // NOTE: the result of Transform is memoized.
    if (const CallNode* call_node = expr.as<CallNode>()) {
      return Transform(call_node, message, scale);
    } else {
      CHECK(!message.defined()) << "outstanding scale";
      return ExprMutator::VisitExpr(expr);
    }
  }
  /*!
   * \brief Normal way of mutating call node.
   * \param call_node The call node to be mutated.
   * \return the result of the call Mutation.
   */
  Expr NormalCallTransform(const CallNode* call_node) {
    const Call call = GetRef<Call>(call_node);
    const auto it = memo_.find(call);
    if (it != memo_.end()) {
      return it->second;
    }
    Expr new_expr = ExprMutator::VisitExpr_(call_node);
    memo_[call] = new_expr;
    return new_expr;
  }
  /*!
   * \brief Get the message propogated to the expr.
   * \param expr The expresison.
   * \return The message containing the expected axes and whether positive scale is required.
   */
  Message GetMessage(const Expr& expr) const {
    auto it = message_.find(expr.get());
    if (it != message_.end()) return it->second;
    return NullValue<Message>();
  }

  // solver is not serializable.
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relay.fold_scale_axis.FBackwardTransformer";
  TVM_DECLARE_FINAL_OBJECT_INFO(BackwardTransformerNode, Object);

 private:
  // Valid axes on each node.
  std::unordered_map<const Object*, Message> message_;
  // Override mutation of call.
  Expr VisitExpr_(const CallNode* call_node) final {
    return Transform(call_node, NullValue<Message>(), NullValue<Expr>());
  }
  // Transform of CallNode.
  Expr Transform(const CallNode* call_node, Message message, Expr scale);
};

class BackwardTransformer : public ObjectRef {
 public:
  BackwardTransformer() {}
  explicit BackwardTransformer(
      ::tvm::ObjectPtr<::tvm::Object> n) : ObjectRef(n) {
  }
  BackwardTransformerNode* operator->() const {
    return static_cast<BackwardTransformerNode*>(get_mutable());
  }
  using ContainerType = BackwardTransformerNode;
};

Expr BackwardTransformerNode::Transform(
    const CallNode* call_node, Message message, Expr scale) {
  static const auto& ftransform =
      Op::GetAttr<FBackwardTransform>("FScaleAxisBackwardTransform");
  auto f = ftransform.get(call_node->op, nullptr);
  if (f != nullptr) {
    const Call call = GetRef<Call>(call_node);
    const auto it = memo_.find(call);
    if (it != memo_.end()) {
      return it->second;
    }
    Expr new_expr = f(GetRef<Call>(call_node),
                      message,
                      scale,
                      GetRef<BackwardTransformer>(this));
    memo_[call] = new_expr;
    return new_expr;
  } else {
    CHECK(!message.defined()) << "outstanding scale";
    return NormalCallTransform(call_node);
  }
}


//----------------------------------------------
// Per operator defs for FScaleAxisForward
//----------------------------------------------

// Intermediate operators
Message ReluBackwardPrep(const Call& call, const Array<Message>& in_messages) {
  if (in_messages[0].defined()) {
    return Message(in_messages[0]->axes, true);
  }
  return in_messages[0];
}

Expr ReluBackwardTransform(const Call& call,
                           const Message& message,
                           const Expr& scale,
                           const BackwardTransformer& transformer) {
  if (!message.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  Expr input = transformer->Transform(
      call->args[0], message, scale);
  return Call(call->op, {input}, call->attrs, call->type_args);
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
Message AddSubBackwardPrep(const Call& call, const Array<Message>& in_messages) {
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  StructuralEqual equal;
  if (in_messages[0].defined() &&
      MatchBroadcastToLeftAxes(tlhs, trhs, in_messages[0]->axes)) {
    return in_messages[0];
  } else if (in_messages[1].defined() &&
             MatchBroadcastToLeftAxes(trhs, tlhs, in_messages[1]->axes)) {
    return in_messages[1];
  } else if (in_messages[0].defined() &&
             in_messages[1].defined() &&
             equal(in_messages[0]->axes, in_messages[1]->axes) &&
             equal(tlhs->shape, trhs->shape)) {
    // add of two elements.
    return in_messages[0];
  } else {
    auto res = NullValue<Message>();
    return res;
  }
}

Expr AddSubBackwardTransform(const Call& call,
                             const Message& message,
                             const Expr& scale,
                             const BackwardTransformer& transformer) {
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  if (!message.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  Message lhs_message = transformer->GetMessage(call->args[0]);
  Message rhs_message = transformer->GetMessage(call->args[1]);
  StructuralEqual equal;

  if (lhs_message.defined() && rhs_message.defined()) {
    CHECK(equal(lhs_message->axes, rhs_message->axes));
    CHECK(equal(message->axes, lhs_message->axes));
    Expr lhs = transformer->Transform(call->args[0], message, scale);
    Expr rhs = transformer->Transform(call->args[1], message, scale);
    return Call(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else if (lhs_message.defined()) {
    CHECK(equal(message->axes, lhs_message->axes));
    Expr lhs = transformer->Transform(call->args[0], message, scale);
    Expr rhs = transformer->Transform(
        call->args[1], NullValue<Message>(), NullValue<Expr>());
    Expr rhs_scale = ExpandBiasToMatchAxis(
        scale, tlhs->shape.size(), message->axes);
    rhs = Multiply(rhs, rhs_scale);
    return Call(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else if (rhs_message.defined()) {
    CHECK(equal(message->axes, rhs_message->axes));
    Expr lhs = transformer->Transform(
        call->args[0], NullValue<Message>(), NullValue<Expr>());
    Expr rhs = transformer->Transform(call->args[1], message, scale);
    Expr lhs_scale = ExpandBiasToMatchAxis(
        scale, trhs->shape.size(), message->axes);
    lhs = Multiply(lhs, lhs_scale);
    return Call(call->op, {lhs, rhs}, call->attrs, call->type_args);
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
                               const Message& message,
                               const Expr& scale,
                               const BackwardTransformer& transformer) {
  CHECK(!message.defined()) << "outstanding scale";
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  Message lhs_message = transformer->GetMessage(call->args[0]);
  Message rhs_message = transformer->GetMessage(call->args[1]);
  if (lhs_message.defined()) {
    CHECK(lhs_message->axes.defined() && lhs_message->axes.size());
    // NOTE we won't recursively call mutating on scale part.
    // since there  won't be scale chance within scale part.
    Expr rhs = call->args[1];
    if (MatchBroadcastToLeftAxes(tlhs, trhs, lhs_message->axes, &rhs) &&
        (!lhs_message->require_positive || IsAllPositiveConstant(rhs))) {
      return transformer->Transform(call->args[0], lhs_message, rhs);
    }
  } else if (rhs_message.defined()) {
    CHECK(rhs_message->axes.defined() && rhs_message->axes.size());
    Expr lhs = call->args[0];
    if (MatchBroadcastToLeftAxes(trhs, tlhs, rhs_message->axes, &lhs) &&
        (!rhs_message->require_positive || IsAllPositiveConstant(lhs))) {
      return transformer->Transform(call->args[1], rhs_message, lhs);
    }
  }
  return transformer->NormalCallTransform(call.operator->());
}

RELAY_REGISTER_OP("multiply")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", MultiplyBackwardTransform);

// Consumer operators
// Conv2D send out requirement of axis folding.
Message Conv2DBackwardPrep(const Call& call, const Array<Message>& in_messages) {
  const auto* param = call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout kernel_layout(param->kernel_layout);
  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  int c_big_axis = out_layout.IndexOf(LayoutAxis::Get('C'));
  int c_small_axis = out_layout.IndexOf(LayoutAxis::Get('c'));

  CHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // More general layout can be supported under the current framework.
  // By using a unified layout transformation.
  // We only need to change the Prep and Mutate function.
  //
  // only handle depthwise or full conv2d.
  // TODO(tvm-team) handle grouped conv by reshape + bcast
  bool is_depthwise_conv2d = IsDepthwiseConv2D(call, param, kernel_layout);
  if (kernel_layout.IndexOf(LayoutAxis::Get('o')) < 0 &&
  kernel_layout.IndexOf(LayoutAxis::Get('i')) < 0 &&
      c_small_axis < 0 &&
      (param->groups == 1 || is_depthwise_conv2d)) {
    return Message({c_big_axis}, false);
  } else {
    return NullValue<Message>();
  }
}

// Conv2D consumes the scale axis during transformation.
Expr Conv2DBackwardTransform(const Call& call,
                             const Message& message,
                             const Expr& scale,
                             const BackwardTransformer& transformer) {
  if (!message.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  const auto* param = call->attrs.as<Conv2DAttrs>();
  CHECK(param != nullptr);
  Layout kernel_layout(param->kernel_layout);
  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  int c_big_axis = out_layout.IndexOf(LayoutAxis::Get('C'));
  CHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // TODO(tvm-team) support general data layout
  CHECK_EQ(kernel_layout.IndexOf(LayoutAxis::Get('o')), -1);
  CHECK_EQ(kernel_layout.IndexOf(LayoutAxis::Get('i')), -1);
  CHECK(message->axes.size() == 1 &&
        c_big_axis == message->axes[0]->value);

  int big_oc_axis = kernel_layout.IndexOf(LayoutAxis::Get('O'));
  // Check it must be depthwise or full conv2d.
  bool is_depthwise_conv2d = IsDepthwiseConv2D(call, param, kernel_layout);
  CHECK(param->groups == 1 || is_depthwise_conv2d);

  Expr data = transformer->Transform(
      call->args[0], NullValue<Message>(), NullValue<Expr>());
  Expr weight = transformer->Transform(
      call->args[1], NullValue<Message>(), NullValue<Expr>());
  // scale on input for deptwise.
  Expr wscale = ExpandBiasToMatchAxis(
      scale, kernel_layout.ndim(), {big_oc_axis});
  weight = Multiply(weight, wscale);
  return Call(
      call->op, {data, weight}, call->attrs, call->type_args);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", Conv2DBackwardPrep);

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", Conv2DBackwardTransform);

Expr BackwardFoldScaleAxis(const Expr& data) {
  return make_object<BackwardTransformerNode>()->Fold(data);
}

}  // namespace fold_scale_axis

namespace transform {

Pass ForwardFoldScaleAxis() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(
          relay::fold_scale_axis::ForwardFoldScaleAxis(f));
  };
  return CreateFunctionPass(pass_func, 3, "ForwardFoldScaleAxis", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.ForwardFoldScaleAxis")
.set_body_typed(ForwardFoldScaleAxis);

Pass BackwardFoldScaleAxis() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(
          relay::fold_scale_axis::BackwardFoldScaleAxis(f));
    };
  return CreateFunctionPass(pass_func, 3, "BackwardFoldScaleAxis", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.BackwardFoldScaleAxis")
.set_body_typed(BackwardFoldScaleAxis);

Pass FoldScaleAxis() {
  // FoldScaleAxis pass contains the following three passes. Therefore, we can
  // register it as a sequential pass.
  Pass pass = Sequential(
      {BackwardFoldScaleAxis(), ForwardFoldScaleAxis(), FoldConstant()},
      "FoldScaleAxis");
  return pass;
}

TVM_REGISTER_GLOBAL("relay._transform.FoldScaleAxis")
.set_body_typed(FoldScaleAxis);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
