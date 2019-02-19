/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file deivce_annotation.cc
 * \brief Passes to rewrite annotated program and retrieve the device allocation
 * of expression.
 *
 * The following passes are performed:
 *  1. Validate the unnecessary and redundant annotation.
 *  2. Rewrite the annotated program and insert data copy operators.
 *  3. Collect the device allocation of each expression.
 */

#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pass.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {

namespace {

bool IsOnDeviceNode(const ExprNode* node) {
  const auto* call_node = dynamic_cast<const CallNode*>(node);
  return call_node != nullptr && call_node->attrs.as<OnDeviceAttrs>();
}

bool IsDeviceCopyNode(const ExprNode* node) {
  const auto* call_node = dynamic_cast<const CallNode*>(node);
  return call_node != nullptr && call_node->attrs.as<DeviceCopyAttrs>();
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
    ExprVisitor::VisitExpr_(call_node);
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
      Expr new_let = LetNode::make(op->var, value, body);
      UpdateAnnotationMap(op, new_let.operator->());
      return this->VisitExpr(new_let);
    }
  }

  Expr VisitExp_(const TupleNode* op) {
    Array<Expr> fields;
    bool annotated = false;
    for (const auto& field : fields) {
      annotated |= NeedDeviceCopy(field.operator->(), op);
      fields.push_back(GetDeviceCopyExpr(field, op));
    }

    if (annotated) {
      Expr new_tuple = TupleNode::make(fields);
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
          TupleGetItemNode::make(GetDeviceCopyExpr(tuple, op), op->index);
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
      Expr new_if = IfNode::make(cond, true_br, false_br);
      UpdateAnnotationMap(if_node, new_if.operator->());
      return this->VisitExpr(new_if);
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    if (IsOnDeviceNode(call_node) || IsDeviceCopyNode(call_node)) {
      return ExprMutator::VisitExpr_(call_node);
    }

    Array<Expr> new_args;
    bool annotated = false;
    for (const auto& arg : call_node->args) {
      annotated |= NeedDeviceCopy(arg.operator->(), call_node);
      new_args.push_back(GetDeviceCopyExpr(arg, call_node));
    }

    if (annotated) {
      Call new_call = CallNode::make(call_node->op, new_args, call_node->attrs,
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
        if (src->is_type<CallNode>() || src->is_type<FunctionNode>()) {
          return annotation_map_.at(dst) != fallback_device_;
        } else {
          return false;
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
    auto attrs = make_node<DeviceCopyAttrs>();
    attrs->src_dev_type = src_dev_type;
    attrs->dst_dev_type = dst_dev_type;
    static const Op& op = Op::Get("device_copy");
    Call device_copy = CallNode::make(op, {src}, Attrs(attrs), {});
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
 *  -Pass 2: Propagating the destination device type of "the last" copy op in a
 *           top-down manner to the nodes on the output paths. For instance,
 *           this offers `subtract` and `exp` the same device type as `copy3`.
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
      post_dfs_order_.push_back(cn);
    }

    void VisitExpr_(const CallNode* call) final {
      // Skip annotation nodes.
      if (!IsOnDeviceNode(call)) {
        ExprVisitor::VisitExpr_(call);
        post_dfs_order_.push_back(call);

        if (GetDeviceCopyNode(call)) {
          num_device_copy_ops_++;
        }
      }
    }

    void VisitExpr_(const TupleNode* tn) final {
      ExprVisitor::VisitExpr_(tn);
      // TODO(zhiics) Skip annotation of tuple node for now.
    }

    void VisitExpr_(const TupleGetItemNode* op) final {
      ExprVisitor::VisitExpr_(op);
      post_dfs_order_.push_back(op);
    }

    void VisitExpr_(const VarNode* vn) final { post_dfs_order_.push_back(vn); }

    void VisitExpr_(const LetNode* ln) final {
      ExprVisitor::VisitExpr_(ln);
      post_dfs_order_.push_back(ln);
    }

    void VisitExpr_(const IfNode* in) final {
      ExprVisitor::VisitExpr_(in);
      post_dfs_order_.push_back(in);
    }

    int num_device_copy_ops_{0};
    std::vector<const ExprNode*> post_dfs_order_;
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
    } else if (const auto* call_node = dynamic_cast<const CallNode*>(node)) {
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
    BottomUpPropagation();
    // Top-down propagation.
    TopDownPropagation();
  }

  void BottomUpPropagation() {
    const CallNode* last_copy_node = nullptr;
    int cur_dev_type = -1;
    for (auto it = post_visitor_.post_dfs_order_.crbegin();
         it != post_visitor_.post_dfs_order_.crend(); ++it) {
      if (const auto* node = GetDeviceCopyNode(*it)) {
        last_copy_node = dynamic_cast<const CallNode*>(node);
        const auto* attrs = last_copy_node->attrs.as<DeviceCopyAttrs>();
        cur_dev_type = attrs->src_dev_type;
        device_map_.Set(GetRef<Expr>(*it), attrs->dst_dev_type);
      } else if (last_copy_node) {
        Expr expr = GetRef<Expr>(*it);
        CHECK_EQ(device_map_.count(expr), 0U);
        device_map_.Set(expr, cur_dev_type);
      }
    }
  }

  void TopDownPropagation() {
    const CallNode* last_copy_node = nullptr;
    int cur_dev_type = -1;
    for (const auto& it : post_visitor_.post_dfs_order_) {
      if (const auto* node = GetDeviceCopyNode(it)) {
        last_copy_node = dynamic_cast<const CallNode*>(node);
        const auto* attrs = last_copy_node->attrs.as<DeviceCopyAttrs>();
        cur_dev_type = attrs->dst_dev_type;
      } else if (last_copy_node) {
        Expr expr = GetRef<Expr>(it);
        if (device_map_.count(expr) == 0) {
          device_map_.Set(expr, cur_dev_type);
        }
      }
    }
  }

  PostDfsOrderVisitor post_visitor_;
  Map<Expr, Integer> device_map_;
};

Expr RewriteAnnotatedOps(const Expr& expr, int fallback_device) {
  RewriteAnnotation rewrote = RewriteAnnotation();
  return rewrote.Rewrite(expr, fallback_device);
}

Map<Expr, Integer> CollectDeviceInfo(const Expr& expr) {
  return DeviceInfo::GetDeviceMap(expr);
}

Map<Expr, Integer> CollectDeviceAnnotationOps(const Expr& expr) {
  return AnnotatationVisitor::GetAnnotations(expr);
}

TVM_REGISTER_API("relay._ir_pass.CollectDeviceInfo")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = CollectDeviceInfo(args[0]);
});

TVM_REGISTER_API("relay._ir_pass.RewriteDeviceAnnotation")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = RewriteAnnotatedOps(args[0], args[1]);
});

TVM_REGISTER_API("relay._ir_pass.CollectDeviceAnnotationOps")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = CollectDeviceAnnotationOps(args[0]);
});

}  // namespace relay
}  // namespace tvm
