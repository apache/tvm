/*!
 * Copyright (c) 2019 by Contributors
 *
 * \file mac_count.cc
 * \brief Pass to roughly count the number of MACs (Multiply-Accumulate) 
 * operations of a model. Only MACs in CONV and Dense ops are counted.
 * This pass is valid after the type infer pass is called,
 * otherwise the count is 0.
 */

#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

namespace {

bool IsConv2DNode(const ExprNode* node) {
  const auto* call_node = dynamic_cast<const CallNode*>(node);
  return call_node != nullptr && call_node->attrs.as<Conv2DAttrs>();
}

bool IsDenseNode(const ExprNode* node) {
  const auto* call_node = dynamic_cast<const CallNode*>(node);
  return call_node != nullptr && call_node->attrs.as<DenseAttrs>();
}

}  // namespace

class MacCounter : private ExprVisitor {
 public:
  MacCounter() {
      count_ = 0;
  }
  static int64_t GetTotalMacNumber(const Expr& expr) {
    MacCounter counter;
    counter(expr);
    return counter.count_;
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    if (IsConv2DNode(call_node)) {
      count_ += ComputeConv2DMacs(call_node);
    } else if (IsDenseNode(call_node)) {
      count_ += ComputeDenseMacs(call_node);
    }
    ExprVisitor::VisitExpr_(call_node);
  }

  /*
   * \brief Get the number of MACs of a CONV 2D node.
   * \param call_node The CONV 2D call node.
   * \return The number of MACs.
   */
  int64_t ComputeConv2DMacs(const CallNode* call_node) {
    CHECK(IsConv2DNode(call_node))
        << "The input call node must be a CONV 2D node.";
    if (!call_node->checked_type_.defined()) {
      LOG(WARNING) << "The infer type pass should be called before the mac count pass";
      return 0;
    }
    Array<Expr> args = call_node->args;
    CHECK(args.size() == 2)
        << "The number of input arguments of a CONV 2D node should be 2.";
    const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
    Array<IndexExpr> data_shape = data_type->shape;
    CHECK(data_shape.size() == 4)
        << "The dimension of an input tensor to Dense node should be 4.";
    int64_t input_channel = static_cast<int64_t>(data_shape[1].as<IntImm>()->value);
    const auto* conv_2d_attr = call_node->attrs.as<Conv2DAttrs>();
    Array<IndexExpr> kernel_size = conv_2d_attr->kernel_size;
    CHECK(kernel_size.size() == 2)
        << "The dimension of the kernel size in Conv 2D should be 2.";
    const auto* expr = call_node->checked_type().as<TensorTypeNode>();
    Array<IndexExpr> output_tensor = expr->shape;
    CHECK(output_tensor.size() == 4)
        << "The dimension of the output tensor in Conv 2D should be 4.";
    int64_t count = input_channel * GetCartesianProd(output_tensor) * GetCartesianProd(kernel_size);
    return count;
  }

  /*
   * \brief Get the number of MACs of a Dense node.
   * \param call_node The Dense call node.
   * \return The number of MACs.
   */
  int64_t ComputeDenseMacs(const CallNode* call_node) {
    CHECK(IsDenseNode(call_node))
        << "The input call node must be a Dense node.";
    if (!call_node->checked_type_.defined()) {
      LOG(WARNING) << "The infer type pass should be called before the mac count pass";
      return 0;
    }
    Array<Expr> args = call_node->args;
    CHECK(args.size() == 2)
        << "The number of input arguments of a Dense node should be 2.";
    const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
    const auto* weight_type = args[1]->checked_type().as<TensorTypeNode>();
    Array<IndexExpr> data_shape = data_type->shape;
    Array<IndexExpr> weight_shape = weight_type->shape;
    CHECK(data_shape.size() == 2 && weight_shape.size() == 2)
        << "The dimension of an input tensor to Dense node should be 2.";
    int64_t d1 = static_cast<int64_t>(data_shape[0].as<IntImm>()->value);
    int64_t d2 = static_cast<int64_t>(data_shape[1].as<IntImm>()->value);
    int64_t d3 = static_cast<int64_t>(weight_shape[0].as<IntImm>()->value);
    int64_t d4 = static_cast<int64_t>(weight_shape[1].as<IntImm>()->value);
    CHECK(d2 == d4)
        << "The dimensions of input arguments do not match.";
    int64_t count = d1 * d2 * d3;
    return count;
  }

  int64_t GetCartesianProd(Array<IndexExpr> arr) {
    int64_t ret = 1;
    for (size_t i = 0; i < arr.size(); i++) {
      const auto* intImm = arr[i].as<IntImm>();
      ret *= static_cast<int64_t>(intImm->value);
    }
    return ret;
  }

  int64_t count_;
};

int64_t GetTotalMacNumber(const Expr& expr) {
  return MacCounter::GetTotalMacNumber(expr);
}

TVM_REGISTER_API("relay._ir_pass.GetTotalMacNumber")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = GetTotalMacNumber(args[0]);
});

}  // namespace relay
}  // namespace tvm
