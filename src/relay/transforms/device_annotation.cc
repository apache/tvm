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
 * \file deivce_annotation.cc
 * \brief Passes to rewrite annotated program and retrieve the device allocation
 * of expression.
 *
 * The following passes are performed:
 *  1. Validate the unnecessary and redundant annotation.
 *  2. Rewrite the annotated program and insert data copy operators.
 *  3. Collect the device allocation of each expression.
 */

#include <tvm/tir/expr.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {

namespace {

bool IsOnDeviceNode(const ExprNode* node) {
  if (!node->IsInstance<CallNode>()) return false;
  const auto* call_node = static_cast<const CallNode*>(node);
  return call_node->attrs.as<OnDeviceAttrs>();
}

bool IsDeviceCopyNode(const ExprNode* node) {
  if (!node->IsInstance<CallNode>()) return false;
  const auto* call_node = static_cast<const CallNode*>(node);
  return call_node->attrs.as<DeviceCopyAttrs>();
}

}  // namespace

class ValidateAnnotation : private ExprVisitor {
 public:
  static std::unordered_map<const ExprNode*, int> Validate(const Expr& expr) {
    ValidateAnnotation valid;
    valid(expr);
    return valid.annotation_map_;
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    ExprVisitor::VisitExpr_(call_node);
    if (IsOnDeviceNode(call_node)) {
      int device_type = GetDeviceId(call_node);
      if (annotation_map_.count(call_node)) {
        CHECK_EQ(annotation_map_.at(call_node), device_type)
            << "An expression node can only be annotated to one device.";
      } else {
        annotation_map_.insert({call_node, GetDeviceId(call_node)});
      }

      CHECK_EQ(call_node->args.size(), 1U);
      const auto* node = call_node->args[0].operator->();
      if (annotation_map_.count(node)) {
        CHECK_EQ(annotation_map_.at(node), device_type)
            << "An expression node can only be annotated to one device.";
      } else {
        annotation_map_.insert({node, GetDeviceId(call_node)});
      }
    }
  }

  void VisitExpr_(const TupleGetItemNode* get_elem) final {
    ExprVisitor::VisitExpr_(get_elem);
    const auto* tn = get_elem->tuple.operator->();
    if (annotation_map_.count(tn)) {
      annotation_map_.insert({get_elem, annotation_map_.at(tn)});
    }
  }

  /*
   * \brief Get the device type of the annotation node.
   * \param call_node The on_device annotation call node.
   * \return The device type.
   */
  int GetDeviceId(const CallNode* call_node) {
    CHECK(IsOnDeviceNode(call_node))
        << "The input call node must be on_device node.";
    const OnDeviceAttrs* on_device_attr = call_node->attrs.as<OnDeviceAttrs>();
    return on_device_attr->device_type;
  }

  std::unordered_map<const ExprNode*, int> annotation_map_;
};

// Replace the use of an expression with the output of a `copy_device` operator
// if the `on_device` operator takes the annotated expr as an input.
//
// This actually replaces annotation ops with device copy ops and connects any
// two dependent expressions with a `device_copy` op when needed. Note that the
// device type of a `device_copy` op is identical to that of the destination op
// since it is where the data should be copied to.
class RewriteAnnotation : public ExprMutator {
 public:
  Expr Rewrite(const Expr& expr, int fallback_device) {
    fallback_device_ = fallback_device;
    annotation_map_ = ValidateAnnotation::Validate(expr);
    return this->VisitExpr(expr);
  }

  Expr VisitExpr_(const LetNode* op) final {
    Expr value = GetDeviceCopyExpr(op->value, op);
    Expr body = GetDeviceCopyExpr(op->body, op);

    if (value.same_as(op->value) && body.same_as(op->body)) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Expr new_let = Let(op->var, value, body);
      UpdateAnnotationMap(op, new_let.operator->());
      return this->VisitExpr(new_let);
    }
  }

  Expr VisitExpr_(const TupleNode* op) {
    Array<Expr> fields;
    bool annotated = false;
    for (const auto& field : op->fields) {
      annotated |= NeedDeviceCopy(field.operator->(), op);
      fields.push_back(GetDeviceCopyExpr(field, op));
    }

    if (annotated) {
      Expr new_tuple = Tuple(fields);
      UpdateAnnotationMap(op, new_tuple.operator->());
      return this->VisitExpr(new_tuple);
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr tuple = op->tuple;
    if (NeedDeviceCopy(tuple.operator->(), op)) {
      Expr new_expr =
          TupleGetItem(GetDeviceCopyExpr(tuple, op), op->index);
      UpdateAnnotationMap(op, new_expr.operator->());
      return this->VisitExpr(new_expr);
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  Expr VisitExpr_(const IfNode* if_node) final {
    Expr cond = GetDeviceCopyExpr(if_node->cond, if_node);
    Expr true_br = GetDeviceCopyExpr(if_node->true_branch, if_node);
    Expr false_br = GetDeviceCopyExpr(if_node->false_branch, if_node);

    if (if_node->cond.same_as(cond) && if_node->true_branch.same_as(true_br) &&
        if_node->false_branch.same_as(false_br)) {
      return ExprMutator::VisitExpr_(if_node);
    } else {
      Expr new_if = If(cond, true_br, false_br);
      UpdateAnnotationMap(if_node, new_if.operator->());
      return this->VisitExpr(new_if);
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    if (IsOnDeviceNode(call_node)) {
      return this->VisitExpr(call_node->args[0]);
    }

    if (IsDeviceCopyNode(call_node)) {
      return ExprMutator::VisitExpr_(call_node);
    }

    Array<Expr> new_args;
    bool annotated = false;
    for (const auto& arg : call_node->args) {
      annotated |= NeedDeviceCopy(arg.operator->(), call_node);
      new_args.push_back(GetDeviceCopyExpr(arg, call_node));
    }

    if (annotated) {
      Call new_call = Call(call_node->op, new_args, call_node->attrs,
                                     call_node->type_args);

      UpdateAnnotationMap(call_node, new_call.operator->());
      return this->VisitExpr(new_call);
    } else {
      return ExprMutator::VisitExpr_(call_node);
    }
  }

 private:
  void UpdateAnnotationMap(const ExprNode* old_node, const ExprNode* new_node) {
    const auto it = annotation_map_.find(old_node);
    if (it == annotation_map_.end()) {
      annotation_map_.insert({new_node, fallback_device_});
    } else {
      annotation_map_.insert({new_node, it->second});
    }
    this->memo_[GetRef<Expr>(old_node)] = GetRef<Expr>(new_node);
  }

  Expr GetDeviceCopyExpr(const Expr& src, const ExprNode* dst) {
    const auto* src_node = src.operator->();
    if (!NeedDeviceCopy(src_node, dst)) return src;

    const auto sit = annotation_map_.find(src_node);
    if (sit == annotation_map_.end()) {
      const auto dit = annotation_map_.find(dst);
      CHECK(dit != annotation_map_.end())
          << "Device copy op is not required when both src and dst ops are not "
             "annotated.";
      return CreateDeviceCopy(src, fallback_device_, dit->second);
    } else {
      const auto dit = annotation_map_.find(dst);
      int dst_dev_type =
          dit == annotation_map_.end() ? fallback_device_ : dit->second;
      return CreateDeviceCopy(src, sit->second, dst_dev_type);
    }
  }

  // Check if a device copy op is need between two ops.
  bool NeedDeviceCopy(const ExprNode* src, const ExprNode* dst) {
    if (annotation_map_.count(src)) {
      int src_dev_type = annotation_map_.at(src);
      if (annotation_map_.count(dst)) {
        return src_dev_type != annotation_map_.at(dst);
      } else {
        return src_dev_type != fallback_device_;
      }
    } else {
      if (annotation_map_.count(dst)) {
        // Though data copy op could be inserted whenever the `src` and `dst`
        // ops are annotated to different devices, it leads to high overhead.
        //
        // Here we need across device data transferring only when `src` is a
        // CallNode or FunctionNode and the `dst` is annotated with any device
        // id other than fallback_device_.
        if (src->IsInstance<CallNode>() || src->IsInstance<FunctionNode>()) {
          return annotation_map_.at(dst) != fallback_device_;
        } else {
          // There shouldn't be any copy nodes between var/constant and another
          // expression.
          return !(src->IsInstance<VarNode>() || src->IsInstance<ConstantNode>());
        }
      } else {
        return false;
      }
    }
  }

  /*
   * \brief Create an operator to copy data from the source device to the
   * destination device.
   * \param src The source expression that produces data to be copied.
   * \param src_dev_type The device type where the data is copied from.
   * \param dst_dev_type The device type where the data is copied to.
   * \return The created call node.
   */
  Call CreateDeviceCopy(const Expr& src, int src_dev_type, int dst_dev_type) {
    auto attrs = make_object<DeviceCopyAttrs>();
    attrs->src_dev_type = src_dev_type;
    attrs->dst_dev_type = dst_dev_type;
    static const Op& op = Op::Get("device_copy");
    Call device_copy = Call(op, {src}, Attrs(attrs), {});
    annotation_map_.insert({device_copy.operator->(), dst_dev_type});
    return device_copy;
  }

  std::unordered_map<const ExprNode*, int> annotation_map_;
  int fallback_device_;
};

// Get all annotation expressions.
class AnnotatationVisitor : private ExprVisitor {
 public:
  static Map<Expr, Integer> GetAnnotations(const Expr& expr) {
    AnnotatationVisitor visitor;
    visitor(expr);
    return visitor.annotations_;
  }
 private:
  void VisitExpr_(const CallNode* call_node) {
    if (IsOnDeviceNode(call_node)) {
      const auto* attr = call_node->attrs.as<OnDeviceAttrs>();
      annotations_.Set(GetRef<Expr>(call_node), attr->device_type);
    }
    ExprVisitor::VisitExpr_(call_node);
  }
  Map<Expr, Integer> annotations_;
};

/*
 * \brief Return device allocation map based on the post order traversed graph.
 * For the following program:
 * .. code-block:: python
 *     x = relay.var("x")
 *     y = relay.var("y")
 *     add = relay.add(x, y)
 *     sqrt = relay.sqrt(add)
 *     log = relay.log(add)
 *     subtract = relay.subtract(sqrt, log)
 *     exp = relay.exp(subtract)
 *
 * Suppose we have annotated add, sqrt, and log with device 1, 2, and 3,
 * respectively. The fallback/default device is 4. After Rewriting the
 * program, we can have the following graph, where each copy op has both
 * source and destination device type denoting which device the data should be
 * copied from and to.
 *
 *         x     y
 *          \   /
 *          add/1
 *          /   \
 *       copy1  copy2
 *         |     |
 *      sqrt/2 log/3
 *         |     |
 *       copy3 copy4
 *          \   /
 *        subtract
 *            |
 *           exp
 *
 * To Get the device mapping of each expression, we need to propagate the
 * device information from the copy ops. This can be done in two passes.
 *  -Pass 1: Propagating the source device type to ops in a bottom-up way to the
 *           ancestors until encountering another copy op. For example, this way
 *           provides add, x, and y device types from the copy operator, `copy1`.
 *  -Pass 2: Propagating the destination device type of "the last" copy op to the
 *           remain nodes. For instance, this offers `subtract` and `exp` the
 *           same device type as `copy3`.
 */

class DeviceInfo {
 public:
  static Map<Expr, Integer> GetDeviceMap(const Expr& expr) {
    DeviceInfo device_info;
    device_info.post_visitor_ = PostDfsOrderVisitor();
    device_info.post_visitor_.Visit(expr);
    if (device_info.post_visitor_.num_device_copy_ops_ > 0) {
      device_info.PropagateDeviceId();
      return device_info.device_map_;
    } else {
      return Map<Expr, Integer>();
    }
  }

 private:
  class PostDfsOrderVisitor : private ExprVisitor {
   public:
    void Visit(const Expr& expr) {
      if (const auto* fn = expr.as<FunctionNode>()) {
        for (const auto& param : fn->params) {
          this->VisitExpr(param);
        }
        this->VisitExpr(fn->body);
      } else {
        this->VisitExpr(expr);
      }
    }

   private:
    // Post order traversal.
    void VisitExpr_(const FunctionNode* fn) final {
      // TODO(zhiics) Skip annotation of function node for now.
    }

    void VisitExpr_(const ConstantNode* cn) final {
      post_dfs_order_.push_back(std::make_pair(cn, has_copy_));
    }

    void VisitExpr_(const CallNode* call) final {
      // Skip annotation nodes.
      if (!IsOnDeviceNode(call)) {
        if (GetDeviceCopyNode(call)) {
          num_device_copy_ops_++;
          bool has_copy_prev = has_copy_;
          has_copy_ = true;
          ExprVisitor::VisitExpr_(call);
          post_dfs_order_.push_back(std::make_pair(call, has_copy_));
          has_copy_ = has_copy_prev;
        } else {
          ExprVisitor::VisitExpr_(call);
          post_dfs_order_.push_back(std::make_pair(call, has_copy_));
        }
      }
    }

    void VisitExpr_(const TupleNode* tn) final {
      ExprVisitor::VisitExpr_(tn);
      // TODO(zhiics) Skip annotation of tuple node for now.
    }

    void VisitExpr_(const TupleGetItemNode* op) final {
      ExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const VarNode* vn) final {
      post_dfs_order_.push_back(std::make_pair(vn, has_copy_));
    }

    void VisitExpr_(const LetNode* ln) final {
      ExprVisitor::VisitExpr_(ln);
      post_dfs_order_.push_back(std::make_pair(ln, has_copy_));
    }

    void VisitExpr_(const IfNode* in) final {
      ExprVisitor::VisitExpr_(in);
      post_dfs_order_.push_back(std::make_pair(in, has_copy_));
    }


    int num_device_copy_ops_{0};
    bool has_copy_ = false;
    std::vector<std::pair<const ExprNode*, bool>> post_dfs_order_;
    friend DeviceInfo;
  };

  /*
   * \brief Returns a device copy node based on the current expr node. It
   * returns a device copy node either the current expr node is a device copy
   * node or the current expr node is a function node whose body is a device
   * copy node (i.e. the fused function of a device copy call node).
   */
  static const ExprNode* GetDeviceCopyNode(const ExprNode* node) {
    if (IsDeviceCopyNode(node)) {
      return node;
    } else if (node->IsInstance<CallNode>()) {
      const auto* call_node = static_cast<const CallNode*>(node);
      if (const auto* fn = call_node->op.as<FunctionNode>()) {
        const ExprNode* body = fn->body.operator->();
        if (IsDeviceCopyNode(body)) {
          return body;
        }
      }
    }
    return nullptr;
  }

  void PropagateDeviceId() {
    // Bottom-up propagation.
    int out_dev_type = BottomUpPropagation();
    // propagation for remained nodes.
    FillPropagation(out_dev_type);
  }

  int BottomUpPropagation() {
    const CallNode* last_copy_node = nullptr;
    int cur_dev_type = -1;
    int out_dev_type = -1;
    for (auto it = post_visitor_.post_dfs_order_.crbegin();
         it != post_visitor_.post_dfs_order_.crend(); ++it) {
      if (const auto* node = GetDeviceCopyNode(it->first)) {
        CHECK(node->IsInstance<CallNode>());
        last_copy_node = static_cast<const CallNode*>(node);
        const auto* attrs = last_copy_node->attrs.as<DeviceCopyAttrs>();
        cur_dev_type = attrs->src_dev_type;
        if (out_dev_type == -1) out_dev_type = attrs->dst_dev_type;
        if (it->second) device_map_.Set(GetRef<Expr>(it->first),
                                        attrs->dst_dev_type);
      } else if (last_copy_node) {
        Expr expr = GetRef<Expr>(it->first);
        CHECK_EQ(device_map_.count(expr), 0U);
        if (it->second) device_map_.Set(expr, cur_dev_type);
      }
    }
      return out_dev_type;
  }

  void FillPropagation(int out_dev_type) {
    for (const auto& it : post_visitor_.post_dfs_order_) {
        Expr expr = GetRef<Expr>(it.first);
        if (!it.second) device_map_.Set(expr, out_dev_type);
    }
  }


  PostDfsOrderVisitor post_visitor_;
  Map<Expr, Integer> device_map_;
};

Expr RewriteAnnotatedOps(const Expr& expr, int fallback_device) {
  RewriteAnnotation rewrote = RewriteAnnotation();
  Expr new_expr = rewrote.Rewrite(expr, fallback_device);

  // Remove OnDevice operators. Note that these operators are only present at the
  // leaves after annotation. Therefore, we can simply reconstruct the
  // Function/Expr by removing them directly.
  if (const FunctionNode* fn = new_expr.as<FunctionNode>()) {
    auto params = fn->params;
    auto body = fn->body;
    std::vector<Expr> new_body;
    if (const TupleNode* tuple = body.as<TupleNode>()) {
      for (const auto& field : tuple->fields) {
        if (!IsOnDeviceNode(field.operator->())) {
          new_body.push_back(field);
        }
      }
      CHECK_GT(new_body.size(), 0U);
      if (new_body.size() == 1) {
        return Function(params, new_body[0], Type(nullptr),
                                  fn->type_params, fn->attrs);
      } else if (tuple->fields.size() == new_body.size()) {
          return new_expr;
      } else {
        Tuple tuple_body = Tuple(new_body);
        return Function(params, tuple_body, Type(nullptr),
                                  fn->type_params, fn->attrs);
      }
    } else {
      return new_expr;
    }
  } else if (const TupleNode* tuple = new_expr.as<TupleNode>()) {
    std::vector<Expr> new_fields;
    for (const auto& field : tuple->fields) {
      if (!IsOnDeviceNode(field.operator->())) {
        new_fields.push_back(field);
      }
    }
    CHECK_GT(new_fields.size(), 0U);
    if (tuple->fields.size() == new_fields.size()) {
      return new_fields.size() == 1 ? new_fields[0] : new_expr;
    } else {
      return new_fields.size() == 1 ? new_fields[0]
                                    : Tuple(new_fields);
    }
  } else {
    return new_expr;
  }
}

Map<Expr, Integer> CollectDeviceInfo(const Expr& expr) {
  return DeviceInfo::GetDeviceMap(expr);
}

Map<Expr, Integer> CollectDeviceAnnotationOps(const Expr& expr) {
  return AnnotatationVisitor::GetAnnotations(expr);
}

TVM_REGISTER_GLOBAL("relay.analysis.CollectDeviceInfo")
.set_body_typed(CollectDeviceInfo);

TVM_REGISTER_GLOBAL("relay.analysis.CollectDeviceAnnotationOps")
.set_body_typed(CollectDeviceAnnotationOps);

namespace transform {

Pass RewriteAnnotatedOps(int fallback_device) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(relay::RewriteAnnotatedOps(f, fallback_device));
  };
  return CreateFunctionPass(pass_func, 1, "RewriteAnnotatedOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.RewriteDeviceAnnotation")
.set_body_typed(RewriteAnnotatedOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
