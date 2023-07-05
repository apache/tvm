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
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/data_layout.h>

#include "../backend/utils.h"
#include "../op/tensor/transform.h"
#include "pass_utils.h"
#include "pattern_utils.h"

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

Message::Message(const AxesSet& axes, bool require_positive) {
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
      ++i;
      ++j;
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
using FForwardPrep =
    runtime::TypedPackedFunc<Array<Message>(const Call& call, const Message& out_message)>;

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
    ICHECK(!axes.defined()) << "outstanding scale";
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

using FForwardRewrite = TypedPackedFunc<Expr(const Call& ref_call, const Array<Expr>& new_args,
                                             const Message& message)>;

//----------------------------------------------
// Generic Visitors for FScaleAxisForward
//----------------------------------------------
class ForwardPrep : private MixedModeVisitor {
 public:
  std::unordered_map<const Object*, Message> Prepare(const Expr& body) {
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
  std::vector<std::function<void()>> flist_;
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

  // We intended the following overrides on implementations from ExprVisitor.
  using MixedModeVisitor::VisitExpr_;

  // Visitor pattern override.
  void VisitExpr_(const TupleGetItemNode* op) final { MixedModeVisitor::VisitExpr_(op); }

  void VisitExpr_(const LetNode* op) final {
    ExprVisitor::VisitExpr_(op);
    // do pass through condition
    // by assigning NullValue<Message>
    // it means fuse signal cannot pass
    // through into these subexpressions.
    auto flazy = [this, op]() {
      this->Update(op->value, NullValue<Message>());
      this->Update(op->body, NullValue<Message>());
    };
    flist_.push_back(flazy);
  }

  void VisitExpr_(const FunctionNode* op) final {
    ExprVisitor::VisitExpr_(op);
    auto flazy = [this, op] { this->Update(op->body, NullValue<Message>()); };
    flist_.push_back(flazy);
  }

  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    // function to be lazily invoked
    auto flazy = [this, call]() {
      static const auto& fprep = Op::GetAttrMap<FForwardPrep>("FScaleAxisForwardPrep");
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
        ICHECK_EQ(in_messages.size(), call->args.size());
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

  void VisitExpr_(const TupleNode* op) final {
    ExprVisitor::VisitExpr_(op);
    // do not support pass scale through tuple for now.
    auto flazy = [this, op]() {
      for (const Expr& field : op->fields) {
        this->Update(field, NullValue<Message>());
      }
    };
    flist_.push_back(flazy);
  }

  void VisitExpr_(const IfNode* op) final {
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

static bool IsIntInArray(const Array<Integer>& axis, int v) {
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] == v) return true;
  }
  return false;
}

static Expr ReshapeToMatchAxis(Expr scale, const Array<PrimExpr>& shape,
                               const Array<Integer>& axis) {
  Array<Integer> arr;
  for (size_t i = 0; i < shape.size(); i++) {
    if (IsIntInArray(axis, i)) {
      auto node = shape[i].as<IntImmNode>();
      if (!node) {
        // if the shape is not a constant, use normal transform
        return Expr();
      }
      arr.push_back(node->value);
    } else {
      arr.push_back(1);
    }
  }
  return MakeReshape(scale, std::move(arr));
}

// if only one axis, use expand dim. Else, use reshape
static Expr ReshapeOrExpandToMatchAxis(Expr scale, const Array<PrimExpr>& shape,
                                       const Array<Integer>& axis) {
  if (axis.size() > 1) {
    return ReshapeToMatchAxis(scale, shape, axis);
  } else {
    return ExpandBiasToMatchAxis(scale, shape.size(), axis);
  }
}

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

Expr ReluForwardRewrite(const Call& ref_call, const Array<Expr>& new_args, const Message& message) {
  const auto* input = new_args[0].as<ScaledExprNode>();
  if (input == nullptr) return Expr(nullptr);
  // return transformed conv2d
  auto rnode = make_object<ScaledExprNode>();
  rnode->value = Call(ref_call->op, {input->value}, ref_call->attrs, ref_call->type_args);
  rnode->scale = input->scale;
  rnode->axes = input->axes;
  return Expr(rnode);
}

RELAY_REGISTER_OP("nn.relu").set_attr<FForwardPrep>("FScaleAxisForwardPrep", ReluForwardPrep);

RELAY_REGISTER_OP("nn.relu").set_attr<FForwardRewrite>("FScaleAxisForwardRewrite",
                                                       ReluForwardRewrite);

RELAY_REGISTER_OP("nn.leaky_relu").set_attr<FForwardPrep>("FScaleAxisForwardPrep", ReluForwardPrep);

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

Expr AddSubForwardRewrite(const Call& ref_call, const Array<Expr>& new_args,
                          const Message& message) {
  const auto* slhs = new_args[0].as<ScaledExprNode>();
  const auto* srhs = new_args[1].as<ScaledExprNode>();
  if (!slhs && !srhs) return Expr();
  const auto* tlhs = ref_call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = ref_call->args[1]->type_as<TensorTypeNode>();
  auto rnode = make_object<ScaledExprNode>();

  if (slhs != nullptr) {
    ICHECK(srhs == nullptr);
    ICHECK(MatchBroadcastToLeftAxes(tlhs, trhs, slhs->axes));
    Expr scale = ReshapeOrExpandToMatchAxis(slhs->scale, tlhs->shape, slhs->axes);
    if (!scale.defined()) {
      return Expr();
    }
    Expr rhs = Divide(new_args[1], scale);
    rnode->value = Call(ref_call->op, {slhs->value, rhs}, ref_call->attrs, ref_call->type_args);
    rnode->scale = slhs->scale;
    rnode->axes = slhs->axes;
  } else {
    ICHECK(srhs != nullptr);
    ICHECK(MatchBroadcastToLeftAxes(trhs, tlhs, srhs->axes));
    Expr scale = ReshapeOrExpandToMatchAxis(srhs->scale, trhs->shape, srhs->axes);
    if (!scale.defined()) {
      return Expr();
    }
    Expr lhs = Divide(new_args[0], scale);
    rnode->value = Call(ref_call->op, {lhs, srhs->value}, ref_call->attrs, ref_call->type_args);
    rnode->scale = srhs->scale;
    rnode->axes = srhs->axes;
  }
  return Expr(rnode);
}

RELAY_REGISTER_OP("add").set_attr<FForwardPrep>("FScaleAxisForwardPrep", AddSubForwardPrep);

RELAY_REGISTER_OP("add").set_attr<FForwardRewrite>("FScaleAxisForwardRewrite",
                                                   AddSubForwardRewrite);

RELAY_REGISTER_OP("subtract").set_attr<FForwardPrep>("FScaleAxisForwardPrep", AddSubForwardPrep);

RELAY_REGISTER_OP("subtract")
    .set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", AddSubForwardRewrite);

// Producer operators
// Multiply produces the scale-axis pair.
Expr MultiplyForwardRewrite(const Call& ref_call, const Array<Expr>& new_args,
                            const Message& message) {
  if (!message.defined()) return Expr();
  const auto& expected_out_axes = message->axes;
  ICHECK(expected_out_axes.defined() && expected_out_axes.size());
  // TODO(tvm-team) allow same axes accumulation
  // not as important because it is less common in nn.
  const auto* slhs = new_args[0].as<ScaledExprNode>();
  const auto* srhs = new_args[1].as<ScaledExprNode>();
  ICHECK(!slhs && !srhs);

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
// Conv send out requirement of axis folding.
template <typename ATTRS>
Array<Message> ConvForwardPrep(const Call& call, const ATTRS* param, const Message& out_message) {
  // TODO(tvm-team) support general data layout
  // by transforming weight
  ICHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout kernel_layout(param->kernel_layout);
  int c_big_axis = data_layout.IndexOf(LayoutAxis::Get('C'));
  int c_small_axis = data_layout.IndexOf(LayoutAxis::Get('c'));

  ICHECK_GE(c_big_axis, 0);
  Message none = NullValue<Message>();
  // For now, we only support simple pattern (no folded weight/data)
  // More general layout can be supported under the current framework.
  // By using a unified layout transformation.
  // We only need to change the Prep and Mutate function.
  //
  // only handle depthwise or full conv2d.
  // TODO(tvm-team) handle grouped conv by reshape + bcast
  bool is_depthwise_conv = IsDepthwiseConv(call, param, kernel_layout);
  if (param->groups == 1 || is_depthwise_conv) {
    auto ko_small_axis = kernel_layout.IndexOf(LayoutAxis::Get('o'));
    auto ki_small_axis = kernel_layout.IndexOf(LayoutAxis::Get('i'));
    if ((ko_small_axis < 0 && ki_small_axis < 0 && c_small_axis < 0) ||     // simple layout
        (ko_small_axis >= 0 && ki_small_axis >= 0 && c_small_axis >= 0)) {  // blocked layout
      Array<Integer> arr{c_big_axis};
      if (c_small_axis >= 0) {
        arr.push_back(c_small_axis);
      }
      return {Message(arr, false), none};
    }
  }
  return {none, none};
}

// Conv2D consumes the scale axis during transformation.
template <typename ATTRS>
Expr ConvForwardRewrite(const Call& ref_call, const ATTRS* param, const Array<Expr>& new_args,
                        const Message& message) {
  // if data do not have scale, normal transform path.
  const auto* sdata = new_args[0].as<ScaledExprNode>();
  const auto* sweight = new_args[1].as<ScaledExprNode>();
  if (sdata == nullptr) return Expr();
  if (sweight != nullptr) return Expr();
  ICHECK(param != nullptr);
  Layout data_layout(param->data_layout);
  Layout kernel_layout(param->kernel_layout);
  int c_big_axis = data_layout.IndexOf(LayoutAxis::Get('C'));
  ICHECK_GE(c_big_axis, 0);
  int small_ko_axis = kernel_layout.IndexOf(LayoutAxis::Get('o'));
  int small_ki_axis = kernel_layout.IndexOf(LayoutAxis::Get('i'));
  int big_ki_axis = kernel_layout.IndexOf(LayoutAxis::Get('I'));
  int big_ko_axis = kernel_layout.IndexOf(LayoutAxis::Get('O'));

  bool is_simple = (small_ko_axis < 0 && small_ki_axis < 0 && big_ki_axis >= 0);
  bool is_blocking = (small_ko_axis >= 0 && small_ki_axis >= 0 && big_ki_axis >= 0);
  ICHECK(is_simple || is_blocking);

  // Check it must be depthwise or full conv2d.
  bool is_depthwise_conv = IsDepthwiseConv(ref_call, param, kernel_layout);
  ICHECK(param->groups == 1 || is_depthwise_conv);

  Expr weight = new_args[1];

  // match the ic_axis
  if (is_depthwise_conv) {
    if (is_simple) {
      Expr scale = ExpandBiasToMatchAxis(sdata->scale, kernel_layout.ndim(), {big_ko_axis});
      weight = Multiply(weight, scale);
    } else {
      weight = Multiply(weight,
                        ReshapeToMatchAxis(sdata->scale, weight->type_as<TensorTypeNode>()->shape,
                                           {big_ko_axis, small_ko_axis}));
      if (!weight.defined()) return Expr();
    }

  } else {
    if (is_simple) {
      Expr scale = ExpandBiasToMatchAxis(sdata->scale, kernel_layout.ndim(), {big_ki_axis});
      weight = Multiply(weight, scale);
    } else {
      weight = Multiply(weight,
                        ReshapeToMatchAxis(sdata->scale, weight->type_as<TensorTypeNode>()->shape,
                                           {big_ki_axis, small_ki_axis}));
      if (!weight.defined()) return Expr();
    }
  }
  // return transformed conv
  return Call(ref_call->op, {sdata->value, weight}, ref_call->attrs, ref_call->type_args);
}

Array<Message> PreConvForwardPrep(const Call& call, const Message& out_message) {
  if (backend::IsOp(call.as<CallNode>(), "nn.conv2d")) {
    const auto* param = call->attrs.as<Conv2DAttrs>();
    ICHECK(param != nullptr);
    return ConvForwardPrep(call, param, out_message);
  }
  const auto* param = call->attrs.as<Conv3DAttrs>();
  ICHECK(param != nullptr);
  return ConvForwardPrep(call, param, out_message);
}

Expr PreConvForwardRewrite(const Call& ref_call, const Array<Expr>& new_args,
                           const Message& message) {
  if (backend::IsOp(ref_call.as<CallNode>(), "nn.conv2d")) {
    const auto* param = ref_call->attrs.as<Conv2DAttrs>();
    ICHECK(param != nullptr);
    return ConvForwardRewrite(ref_call, param, new_args, message);
  }
  const auto* param = ref_call->attrs.as<Conv3DAttrs>();
  ICHECK(param != nullptr);
  return ConvForwardRewrite(ref_call, param, new_args, message);
}

RELAY_REGISTER_OP("nn.conv2d").set_attr<FForwardPrep>("FScaleAxisForwardPrep", PreConvForwardPrep);

RELAY_REGISTER_OP("nn.conv2d")
    .set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", PreConvForwardRewrite);

RELAY_REGISTER_OP("nn.conv3d").set_attr<FForwardPrep>("FScaleAxisForwardPrep", PreConvForwardPrep);

RELAY_REGISTER_OP("nn.conv3d")
    .set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", PreConvForwardRewrite);

// Dense send out requirement of axis folding.
Array<Message> DenseForwardPrep(const Call& call, const Message& out_message) {
  return {Message({1}, false), NullValue<Message>()};
}

// Dense consumes the scale axis during transformation.
Expr DenseForwardRewrite(const Call& ref_call, const Array<Expr>& new_args,
                         const Message& message) {
  const auto* sdata = new_args[0].as<ScaledExprNode>();
  const auto* sweight = new_args[1].as<ScaledExprNode>();
  if (sdata == nullptr) return Expr();
  if (sweight != nullptr) return Expr();

  Expr weight = Multiply(new_args[1], sdata->scale);
  return Call(ref_call->op, {sdata->value, weight}, ref_call->attrs, ref_call->type_args);
}

RELAY_REGISTER_OP("nn.dense").set_attr<FForwardPrep>("FScaleAxisForwardPrep", DenseForwardPrep);

RELAY_REGISTER_OP("nn.dense")
    .set_attr<FForwardRewrite>("FScaleAxisForwardRewrite", DenseForwardRewrite);

Expr ForwardFoldScaleAxis(const Expr& data) {
  auto message = ForwardPrep().Prepare(data);
  for (const auto& m : message) {
    if (m.second.defined()) {
      // run optimization
      auto fcontext = [&](const Call& call) -> ObjectRef {
        auto it = message.find(call.get());
        if (it != message.end()) {
          return it->second;
        } else {
          return ObjectRef(nullptr);
        }
      };
      return ForwardRewrite(data, "FScaleAxisForwardRewrite", fcontext);
    }
  }
  // no messages - no optimization
  return data;
}

//----------------------------------------
// Implement backward transformations.
//----------------------------------------
class BackwardTransformer;

/*!
 * \brief Preparation function for pass scale backward.
 * \param call The call node.
 * \param in_messages Messages from the input containing allowed input scaling and whether
 *        positive scale is required.
 * \return Message containing the result scaling on axes of the input.
 */
using FBackwardPrep = TypedPackedFunc<Message(const Call& call, const Array<Message>& in_messages)>;

using FBackwardTransform =
    TypedPackedFunc<Expr(const Call& call, const Message& message, const Expr& scale,
                         const BackwardTransformer& transformer)>;

//----------------------------------------------
// Generic Visitors for FScaleAxisBackward
//----------------------------------------------

class BackwardPrep : private MixedModeVisitor {
 public:
  // The message on each node.
  std::unordered_map<const Object*, Message> Prepare(const Expr& body) {
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
    static const auto& fprep = Op::GetAttrMap<FBackwardPrep>("FScaleAxisBackwardPrep");
    auto f = fprep.get(call->op, nullptr);
    if (f == nullptr) return;
    auto rit = ref_counter_.find(call);
    ICHECK(rit != ref_counter_.end());
    // We only allow propagation of scale backward
    // if the expression is only referred by a single parent.
    if (rit->second != 1) return;
    Array<Message> in_messages = GetInMessages(call);
    Message out_message = f(GetRef<Call>(call), in_messages);
    if (out_message.defined()) {
      message_[call] = out_message;
    }
  }

  Array<Message> GetInMessages(const CallNode* call) {
    Array<Message> in_messages;
    for (Expr arg : call->args) {
      auto it = message_.find(arg.get());
      if (it != message_.end()) {
        in_messages.push_back(it->second);
      } else {
        in_messages.push_back(NullValue<Message>());
      }
    }
    return in_messages;
  }
};

/*
 * Hybrid apporach is used with the transformation
 * itself is recursive but the traversal is non-recursive
 */
class BackwardTransformerNode : public Object, private MixedModeMutator {
 public:
  using MixedModeMutator::Mutate;
  // Run forward transform.
  Expr Fold(Expr expr) {
    message_ = BackwardPrep().Prepare(expr);
    for (const auto& m : message_) {
      if (m.second.defined()) {
        // run optimization
        return this->Mutate(expr);
      }
    }
    // no messages - no optimization
    return expr;
  }

  /*!
   * \brief Transform the expr to consider the scaling.
   */
  Expr Transform(const Expr& expr, Message message, Expr scale);
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
  Expr Rewrite_(const CallNode* call_node, const Expr& post) final {
    return Transform(GetRef<Call>(call_node), NullValue<Message>(), NullValue<Expr>());
  }

 public:
  Expr NormalCallTransform(const CallNode* call_node) { return ExprMutator::VisitExpr_(call_node); }
};

class BackwardTransformer : public ObjectRef {
 public:
  BackwardTransformer() {}
  explicit BackwardTransformer(::tvm::ObjectPtr<::tvm::Object> n) : ObjectRef(n) {}
  BackwardTransformerNode* operator->() const {
    return static_cast<BackwardTransformerNode*>(get_mutable());
  }
  using ContainerType = BackwardTransformerNode;
};

/*!
 * \brief Transform the expr to consider the scaling.
 *
 * \param expr The input expression.
 * \param message The axes to scale.
 * \param scale The scale applied to the axes.
 * \return The result of transformation.
 */
Expr BackwardTransformerNode::Transform(const Expr& expr, Message message, Expr scale) {
  if (const CallNode* call_node = expr.as<CallNode>()) {
    static const auto& ftransform =
        Op::GetAttrMap<FBackwardTransform>("FScaleAxisBackwardTransform");
    auto f = ftransform.get(call_node->op, nullptr);
    const Call call = GetRef<Call>(call_node);
    // ignore if there is a message
    if (!message.defined()) {
      const auto it = memo_.find(call);
      if (it != memo_.end()) {
        return it->second;
      }
    }
    Expr new_expr = NullValue<Expr>();
    if (f != nullptr) {
      new_expr = f(call, message, scale, GetRef<BackwardTransformer>(this));
    } else {
      ICHECK(!message.defined()) << "outstanding scale";
      new_expr = NormalCallTransform(call.operator->());
    }
    memo_[call] = new_expr;
    return new_expr;
  } else {
    ICHECK(!message.defined()) << "outstanding scale";
    return this->Mutate(expr);
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

Expr ReluBackwardTransform(const Call& call, const Message& message, const Expr& scale,
                           const BackwardTransformer& transformer) {
  if (!message.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  Expr input = transformer->Transform(call->args[0], message, scale);
  return Call(call->op, {input}, call->attrs, call->type_args);
}

RELAY_REGISTER_OP("nn.relu").set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", ReluBackwardPrep);

RELAY_REGISTER_OP("nn.relu").set_attr<FBackwardTransform>("FScaleAxisBackwardTransform",
                                                          ReluBackwardTransform);

RELAY_REGISTER_OP("nn.leaky_relu")
    .set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", ReluBackwardPrep);

RELAY_REGISTER_OP("nn.leaky_relu")
    .set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", ReluBackwardTransform);

// AddSub
Message AddSubBackwardPrep(const Call& call, const Array<Message>& in_messages) {
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  StructuralEqual equal;
  if (in_messages[0].defined() && MatchBroadcastToLeftAxes(tlhs, trhs, in_messages[0]->axes)) {
    return in_messages[0];
  } else if (in_messages[1].defined() &&
             MatchBroadcastToLeftAxes(trhs, tlhs, in_messages[1]->axes)) {
    return in_messages[1];
  } else if (in_messages[0].defined() && in_messages[1].defined() &&
             equal(in_messages[0]->axes, in_messages[1]->axes) && equal(tlhs->shape, trhs->shape)) {
    // add of two elements.
    return in_messages[0];
  } else {
    auto res = NullValue<Message>();
    return res;
  }
}

Expr AddSubBackwardTransform(const Call& call, const Message& message, const Expr& scale,
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
    ICHECK(equal(lhs_message->axes, rhs_message->axes));
    ICHECK(equal(message->axes, lhs_message->axes));
    Expr lhs = transformer->Transform(call->args[0], message, scale);
    Expr rhs = transformer->Transform(call->args[1], message, scale);
    return Call(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else if (lhs_message.defined()) {
    ICHECK(equal(message->axes, lhs_message->axes));
    Expr lhs = transformer->Transform(call->args[0], message, scale);
    Expr rhs = transformer->Transform(call->args[1], NullValue<Message>(), NullValue<Expr>());
    Expr rhs_scale = ReshapeOrExpandToMatchAxis(scale, tlhs->shape, message->axes);
    if (!rhs_scale.defined()) {
      return transformer->NormalCallTransform(call.operator->());
    }
    rhs = Multiply(rhs, rhs_scale);
    return Call(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else if (rhs_message.defined()) {
    ICHECK(equal(message->axes, rhs_message->axes));
    Expr lhs = transformer->Transform(call->args[0], NullValue<Message>(), NullValue<Expr>());
    Expr rhs = transformer->Transform(call->args[1], message, scale);
    Expr lhs_scale = ReshapeOrExpandToMatchAxis(scale, trhs->shape, message->axes);
    if (!lhs_scale.defined()) {
      return transformer->NormalCallTransform(call.operator->());
    }
    lhs = Multiply(lhs, lhs_scale);
    return Call(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else {
    LOG(FATAL) << "outstanding scale";
  }
}

RELAY_REGISTER_OP("add").set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", AddSubBackwardPrep);

RELAY_REGISTER_OP("add").set_attr<FBackwardTransform>("FScaleAxisBackwardTransform",
                                                      AddSubBackwardTransform);

RELAY_REGISTER_OP("subtract").set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", AddSubBackwardPrep);

RELAY_REGISTER_OP("subtract")
    .set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", AddSubBackwardTransform);

// Producer operators
// Multiply produces the scale-axis pair.
Expr MultiplyBackwardTransform(const Call& call, const Message& message, const Expr& scale,
                               const BackwardTransformer& transformer) {
  ICHECK(!message.defined()) << "outstanding scale";
  const auto* tlhs = call->args[0]->type_as<TensorTypeNode>();
  const auto* trhs = call->args[1]->type_as<TensorTypeNode>();
  Message lhs_message = transformer->GetMessage(call->args[0]);
  Message rhs_message = transformer->GetMessage(call->args[1]);
  if (lhs_message.defined()) {
    ICHECK(lhs_message->axes.defined() && lhs_message->axes.size());
    // NOTE we won't recursively call mutating on scale part.
    // since there  won't be scale chance within scale part.
    Expr rhs = call->args[1];
    if (MatchBroadcastToLeftAxes(tlhs, trhs, lhs_message->axes, &rhs) &&
        (!lhs_message->require_positive || IsAllPositiveConstant(rhs))) {
      return transformer->Transform(call->args[0], lhs_message, rhs);
    }
  } else if (rhs_message.defined()) {
    ICHECK(rhs_message->axes.defined() && rhs_message->axes.size());
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
// Conv send out requirement of axis folding.
template <typename ATTRS>
Message ConvBackwardPrep(const Call& call, const ATTRS* param, const Array<Message>& in_messages) {
  ICHECK(param != nullptr);
  Layout kernel_layout(param->kernel_layout);
  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  int c_big_axis = out_layout.IndexOf(LayoutAxis::Get('C'));
  int c_small_axis = out_layout.IndexOf(LayoutAxis::Get('c'));

  ICHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // More general layout can be supported under the current framework.
  // By using a unified layout transformation.
  // We only need to change the Prep and Mutate function.
  //
  // only handle depthwise or full conv.
  // TODO(tvm-team) handle grouped conv by reshape + bcast
  bool is_depthwise_conv = IsDepthwiseConv(call, param, kernel_layout);
  if (param->groups == 1 || is_depthwise_conv) {
    auto ko_small_axis = kernel_layout.IndexOf(LayoutAxis::Get('o'));
    auto ki_small_axis = kernel_layout.IndexOf(LayoutAxis::Get('i'));
    if ((ko_small_axis < 0 && ki_small_axis < 0 && c_small_axis < 0) ||     // simple layout
        (ko_small_axis >= 0 && ki_small_axis >= 0 && c_small_axis >= 0)) {  // blocked layout
      Array<Integer> arr{c_big_axis};
      if (c_small_axis >= 0) {
        arr.push_back(c_small_axis);
      }
      return Message(arr, false);
    }
  }
  return NullValue<Message>();
}

// Conv consumes the scale axis during transformation.
template <typename ATTRS>
Expr ConvBackwardTransform(const Call& call, const ATTRS* param, const Message& message,
                           const Expr& scale, const BackwardTransformer& transformer) {
  if (!message.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  ICHECK(param != nullptr);
  Layout kernel_layout(param->kernel_layout);
  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  int c_big_axis = out_layout.IndexOf(LayoutAxis::Get('C'));
  ICHECK_GE(c_big_axis, 0);
  // For now, we only support simple pattern (no folded weight/data)
  // TODO(tvm-team) support general data layout
  int small_ko_axis = kernel_layout.IndexOf(LayoutAxis::Get('o'));
  int small_ki_axis = kernel_layout.IndexOf(LayoutAxis::Get('i'));
  int big_ki_axis = kernel_layout.IndexOf(LayoutAxis::Get('I'));
  int big_ko_axis = kernel_layout.IndexOf(LayoutAxis::Get('O'));
  // Check it must be depthwise or full conv.
  bool is_depthwise_conv = IsDepthwiseConv(call, param, kernel_layout);
  ICHECK(param->groups == 1 || is_depthwise_conv);
  bool is_simple = (small_ko_axis < 0 && small_ki_axis < 0 && big_ki_axis >= 0);
  bool is_blocking = (small_ko_axis >= 0 && small_ki_axis >= 0 && big_ki_axis >= 0);
  ICHECK(is_simple || is_blocking);

  Expr data = transformer->Transform(call->args[0], NullValue<Message>(), NullValue<Expr>());
  Expr weight = transformer->Transform(call->args[1], NullValue<Message>(), NullValue<Expr>());
  // scale on input for deptwise.
  Expr wscale;
  if (is_simple) {
    wscale = ExpandBiasToMatchAxis(scale, kernel_layout.ndim(), {big_ko_axis});
  } else {
    wscale = ReshapeToMatchAxis(scale, weight->type_as<TensorTypeNode>()->shape,
                                {big_ko_axis, small_ko_axis});
    if (!wscale.defined()) {
      return transformer->NormalCallTransform(call.operator->());
    }
  }
  weight = Multiply(weight, wscale);
  return Call(call->op, {data, weight}, call->attrs, call->type_args);
}

Message PreConvBackwardPrep(const Call& call, const Array<Message>& in_messages) {
  if (backend::IsOp(call.as<CallNode>(), "nn.conv2d")) {
    const auto* param = call->attrs.as<Conv2DAttrs>();
    ICHECK(param != nullptr);
    return ConvBackwardPrep(call, param, in_messages);
  }
  const auto* param = call->attrs.as<Conv3DAttrs>();
  ICHECK(param != nullptr);
  return ConvBackwardPrep(call, param, in_messages);
}

Expr PreConvBackwardTransform(const Call& call, const Message& message, const Expr& scale,
                              const BackwardTransformer& transformer) {
  if (backend::IsOp(call.as<CallNode>(), "nn.conv2d")) {
    const auto* param = call->attrs.as<Conv2DAttrs>();
    ICHECK(param != nullptr);
    return ConvBackwardTransform(call, param, message, scale, transformer);
  }
  const auto* param = call->attrs.as<Conv3DAttrs>();
  ICHECK(param != nullptr);
  return ConvBackwardTransform(call, param, message, scale, transformer);
}

RELAY_REGISTER_OP("nn.conv2d")
    .set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", PreConvBackwardPrep);

RELAY_REGISTER_OP("nn.conv2d")
    .set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", PreConvBackwardTransform);

RELAY_REGISTER_OP("nn.conv3d")
    .set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", PreConvBackwardPrep);

RELAY_REGISTER_OP("nn.conv3d")
    .set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", PreConvBackwardTransform);

Message BiasAddBackwardPrep(const Call& call, const Array<Message>& in_messages) {
  const BiasAddAttrs* attrs = call->attrs.as<BiasAddAttrs>();
  ICHECK(attrs);
  if (in_messages[0].defined() && in_messages[0]->axes.size() == 1 &&
      attrs->axis == static_cast<int>(in_messages[0]->axes[0]->value)) {
    return in_messages[0];
  } else {
    return NullValue<Message>();
  }
}

Expr BiasAddBackwardTransform(const Call& call, const Message& message, const Expr& scale,
                              const BackwardTransformer& transformer) {
  if (!message.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  Message lhs_message = transformer->GetMessage(call->args[0]);
  Message rhs_message = transformer->GetMessage(call->args[1]);
  StructuralEqual equal;

  if (lhs_message.defined()) {
    ICHECK(equal(message->axes, lhs_message->axes));
    Expr lhs = transformer->Transform(call->args[0], message, scale);
    Expr rhs = transformer->Transform(call->args[1], NullValue<Message>(), NullValue<Expr>());
    rhs = Multiply(rhs, scale);
    return Call(call->op, {lhs, rhs}, call->attrs, call->type_args);
  } else {
    LOG(FATAL) << "outstanding scale";
  }
}

RELAY_REGISTER_OP("nn.bias_add")
    .set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", BiasAddBackwardPrep);

RELAY_REGISTER_OP("nn.bias_add")
    .set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", BiasAddBackwardTransform);

// Dense send out requirement of axis folding.
Message DenseBackwardPrep(const Call& call, const Array<Message>& in_messages) {
  return Message({1}, false);
}

// Dense consumes the sacle axis during trasformation.
Expr DenseBackwardTransform(const Call& call, const Message& message, const Expr& scale,
                            const BackwardTransformer& transformer) {
  if (!message.defined()) {
    return transformer->NormalCallTransform(call.operator->());
  }
  Expr data = transformer->Transform(call->args[0], NullValue<Message>(), NullValue<Expr>());
  Expr weight = transformer->Transform(call->args[1], NullValue<Message>(), NullValue<Expr>());
  Expr wscale = ExpandBiasToMatchAxis(scale, 2, {0});
  weight = Multiply(weight, wscale);
  return Call(call->op, {data, weight}, call->attrs, call->type_args);
}

RELAY_REGISTER_OP("nn.dense").set_attr<FBackwardPrep>("FScaleAxisBackwardPrep", DenseBackwardPrep);

RELAY_REGISTER_OP("nn.dense")
    .set_attr<FBackwardTransform>("FScaleAxisBackwardTransform", DenseBackwardTransform);

Expr BackwardFoldScaleAxis(const Expr& data) {
  return make_object<BackwardTransformerNode>()->Fold(data);
}

}  // namespace fold_scale_axis

namespace transform {

Pass ForwardFoldScaleAxis() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::fold_scale_axis::ForwardFoldScaleAxis(f));
      };
  return CreateFunctionPass(pass_func, 3, "ForwardFoldScaleAxis", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.ForwardFoldScaleAxis").set_body_typed(ForwardFoldScaleAxis);

Pass BackwardFoldScaleAxis() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::fold_scale_axis::BackwardFoldScaleAxis(f));
      };
  return CreateFunctionPass(pass_func, 3, "BackwardFoldScaleAxis", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.BackwardFoldScaleAxis").set_body_typed(BackwardFoldScaleAxis);

Pass FoldScaleAxis() {
  // FoldScaleAxis pass contains the following three passes. Therefore, we can
  // register it as a sequential pass.
  Pass pass = Sequential({BackwardFoldScaleAxis(), ForwardFoldScaleAxis(), FoldConstant()},
                         "FoldScaleAxis");
  return pass;
}

TVM_REGISTER_GLOBAL("relay._transform.FoldScaleAxis").set_body_typed(FoldScaleAxis);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
