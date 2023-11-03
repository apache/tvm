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
 * \file src/relay/transforms/quantize_fake_quantization.cc
 * \brief A pass for taking fake quantized graphs and converting them
 * to actual integer operations.
 */

#include "fake_quantization_to_integer.h"

#include <tvm/ir/affine_type.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>

#include <unordered_map>

#include "../qnn/utils.h"

namespace tvm {
namespace relay {

/* Description of FakeQuantizationToInteger
 *
 * The purpose of this pass is to find regions of the graph that follow
 * the general pattern:
 *
 *   x    w
 *   |    |
 *   dq   dq
 *    \   /
 *     op1
 *      |
 *     op2
 *      |
 *      q
 *
 * and convert them into subgraphs with actual integer operations on x and w
 *
 * The pass does this via a multi-pass approach:
 *
 * The main pass is a MixedModeMutator that traverses the full graph searching for
 * quantize operations
 *
 * The second pass is an ExprVisitor that recursively searches for subgraphs leading to the
 * quantize for subtraphs bounded by dequantize operations. This pass extracts the affine
 * types of the inputs for later processing, where affine denotes the transformation
 * x_real = (x_affine - zero_point) * scale
 *
 * The third pass is an ExprMutator that recursively rewrites the subgraphs using packed funcs
 * registered with the FTVMFakeQuantizationToInteger attribute. These packed funcs rewrite
 * the ops based on the affine types of their inputs and then return the affine types of the
 * new rewriten ops to pass that information down the stack during rewrite.
 *
 * After the second and third passes run, the first pass replaces the quantize with the
 * rewritten subgraph and the processing continues
 *
 *
 * After that an additional QAT pass can be enabled by use_qat flag. The goal of the pass is to find
 * operations in those regions(which were not successfully converted by the main pass) that can
 * still be converted into quantized form. The idea is to find and transform operations with
 * dequantized inputs one by one individually. Only operations for which all parameters can be
 * explicitly calculated are allowed. For example, if on the above general  pattern op2 is not
 * registered with the FTVMFakeQuantizationToInteger attribute, op1 operation can still be
 * converted. Converted pattern below:
 *
 *   x    w
 *   |    |
 *    \   /
 *     op1
 *      |
 *     dq
 *      |
 *     op2
 *      |
 *      q
 *
 * This pass works in the same multi-pass approach.
 */

using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using ExprMap = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>;
using AffineTypeMap = Map<Expr, AffineType>;

using FTVMFakeQuantizationToInteger =
    runtime::TypedPackedFunc<Array<ObjectRef>(const Expr& expr, const AffineTypeMap& map)>;

const ExprSet SubgraphExtractor::GetSubgraph(const Expr& expr) {
  VisitExpr(expr);
  ExprSet subgraph;
  if (is_fake_quantized_) {
    for (auto kv : this->visit_counter_) {
      if (auto call_node = GetRef<ObjectRef>(kv.first).as<CallNode>()) {
        if (call_node->op != quantize_op_) {
          subgraph.insert(Downcast<Expr>(GetRef<ObjectRef>(kv.first)));
        }
      }
    }
  }
  return subgraph;
}
const AffineTypeMap SubgraphExtractor::GetAffineTypes() { return affine_types_; }
void SubgraphExtractor::VisitExpr(const Expr& expr) {
  // When looking for fake quantized subgraphs, we only support data-flow regions of the graph,
  // i.e. call nodes/tuples/constants/etc. If we see anything else (like control flow) we
  // abort the rewrite.
  if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
      expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
      expr.as<ConstantNode>() == nullptr) {
    DLOG(INFO) << "FakeQuantizationToInteger found a non-dataflow op inside"
               << " a fake quantize region, aborting this rewrite";
    is_fake_quantized_ = false;
  } else {
    ExprVisitor::VisitExpr(expr);
  }
}

void SubgraphExtractor::VisitExpr_(const CallNode* call_node) {
  const Op test_op = Downcast<Op>(call_node->op);
  if (call_node->op == quantize_op_) {
    const auto* attrs = call_node->attrs.as<qnn::QuantizeAttrs>();
    ICHECK(attrs != nullptr);
    // Only look at arg0 for quantize
    VisitExpr(call_node->args[0]);
    // Collect type of quantize ops
    affine_types_.Set(
        GetRef<Expr>(call_node),
        TensorAffineType(call_node->args[1], call_node->args[2], attrs->out_dtype, attrs->axis));
  } else if (call_node->op == dequantize_op_) {
    const auto* attrs = call_node->attrs.as<qnn::DequantizeAttrs>();
    ICHECK(attrs != nullptr);
    // Collect type of dequantize ops
    affine_types_.Set(
        GetRef<Expr>(call_node),
        TensorAffineType(call_node->args[1], call_node->args[2],
                         call_node->args[0]->checked_type().as<TensorTypeNode>()->dtype,
                         attrs->axis));
  } else {
    // run normally on everything else.
    ExprVisitor::VisitExpr_(call_node);
  }
}

class SubgraphMutator : public ExprMutator {
 public:
  SubgraphMutator(ExprSet subgraph, AffineTypeMap affine_types, bool hard_fail,
                  const std::unordered_set<String>& optional_qnn_ops)
      : subgraph_(subgraph),
        affine_types_(affine_types),
        hard_fail_(hard_fail),
        optional_qnn_ops_(optional_qnn_ops) {}

  Expr MutateSubgraph(const Expr& expr) {
    if (subgraph_.size() == 0) {
      return expr;
    }
    const CallNode* quantize_node = expr.as<CallNode>();
    ICHECK(quantize_node);
    ICHECK(quantize_node->op == quantize_op_);
    out_type_ = affine_types_[expr];
    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    static auto opt_fqfq =
        Op::HasAttrMap("FTVMOptionalFakeQuantizationToInteger")
            ? Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMOptionalFakeQuantizationToInteger")
            : fqfq;
    for (auto node : subgraph_) {
      const Op op = Downcast<Op>(node.as<CallNode>()->op);
      if (!fqfq.count(Downcast<Op>(op)) &&
          !(optional_qnn_ops_.count(op->name) && opt_fqfq.count(Downcast<Op>(op)))) {
        // Only modify the subgraph if we have translation
        // rules for every op
        if (hard_fail_) {
          LOG(FATAL) << "Found no rewrite rule for " << AsText(op, false) << std::endl;
        } else {
          DLOG(INFO) << "Found no rewrite rule for " << AsText(op, false) << std::endl;
          return expr;
        }
      }
    }
    try {
      return Mutate(expr);
    } catch (std::exception& e) {
      if (hard_fail_) {
        LOG(FATAL) << e.what();
      } else {
        DLOG(INFO) << "Ran into an error rewriting a subgraph, skipping" << expr << std::endl;
        return expr;
      }
    }
  }

 protected:
  Expr VisitExpr_(const CallNode* call_node) {
    Expr out;

    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    static auto opt_fqfq =
        Op::HasAttrMap("FTVMOptionalFakeQuantizationToInteger")
            ? Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMOptionalFakeQuantizationToInteger")
            : fqfq;
    Op op = Downcast<Op>(call_node->op);
    if (fqfq.count(op) || (optional_qnn_ops_.count(op->name) && opt_fqfq.count(op))) {
      Expr expr;
      if (op == dequantize_op_) {
        expr = GetRef<Expr>(call_node);
      } else {
        expr = ExprMutator::VisitExpr_(call_node);
        // Set the current op to the output type, useful if we can't deduce output parameters
        // from input parameters
        affine_types_.Set(expr, out_type_);
      }
      // Call the rewrite
      Array<ObjectRef> vals = (fqfq.count(op) ? fqfq : opt_fqfq)[op](expr, affine_types_);
      // Save the outputs of the rewrite
      ICHECK(vals.size() == 2)
          << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
          << AsText(op, false);
      out = Downcast<Expr>(vals[0]);
      affine_types_.Set(out, Downcast<AffineType>(vals[1]));
    } else {
      ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
                    << AsText(GetRef<Expr>(call_node), false);
    }
    return out;
  }

  Expr VisitExpr_(const TupleNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto new_node = expr.as<TupleNode>();
    Array<TensorAffineType> types;
    for (Expr field : new_node->fields) {
      ICHECK(affine_types_[field].as<TensorAffineTypeNode>());
      types.push_back(Downcast<TensorAffineType>(affine_types_[field]));
    }
    affine_types_.Set(expr, TupleAffineType(types));
    return expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto tuple_type = affine_types_[expr.as<TupleGetItemNode>()->tuple].as<TupleAffineTypeNode>();
    affine_types_.Set(expr, tuple_type->types[node->index]);
    return expr;
  }

  ExprSet subgraph_;
  AffineTypeMap affine_types_;
  AffineType out_type_;
  const bool hard_fail_;
  const std::unordered_set<String>& optional_qnn_ops_;
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
};

class FakeQuantizationRewriter : public MixedModeMutator {
 public:
  explicit FakeQuantizationRewriter(bool hard_fail,
                                    const std::unordered_set<String>& optional_qnn_ops)
      : hard_fail_(hard_fail), optional_qnn_ops_(optional_qnn_ops) {}

 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call_node = post.as<CallNode>()) {
      if (call_node->op == quantize_op_) {
        SubgraphExtractor extractor;
        ExprSet subgraph = extractor.GetSubgraph(GetRef<Expr>(pre));
        AffineTypeMap affine_types = extractor.GetAffineTypes();

        ExprSet post_subgraph;
        AffineTypeMap post_affine_types;

        for (auto kv : affine_types) {
          if (pre == kv.first.as<CallNode>()) {
            // we havent memoized the current op yet
            post_affine_types.Set(post, kv.second);
          } else {
            post_affine_types.Set(memo_.at(kv.first), kv.second);
          }
        }
        for (auto expr : subgraph) {
          post_subgraph.insert(memo_[expr]);
        }
        Expr out = SubgraphMutator(post_subgraph, post_affine_types, hard_fail_, optional_qnn_ops_)
                       .MutateSubgraph(post);
        return out;
      }
    }
    return post;
  }
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const bool hard_fail_;
  const std::unordered_set<String>& optional_qnn_ops_;
};

/* Checks if the operation to convert QAT pass is enabled.
 * The following conditions must be satisfied:
 * 1. operations registered for FTVMFakeQuantizationToInteger;
 * 2. Unary operators or operators with the TensorAffineType calculated during
 * FTVMFakeQuantizationToInteger conversion;
 * 3. Not one of the "key" operations: requantize,quantize and dequantize(they are at the boundaries
 * of regions defined to be quantized).
 */
bool is_op_enabled_for_optional_fq2i(const CallNode* call_node) {
  const Op op = Downcast<Op>(call_node->op);
  static auto fqfq = Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
  static std::unordered_set<relay::Expr, tvm::ObjectHash, tvm::ObjectEqual> ops = {
      Op::Get("broadcast_to"),
      Op::Get("clip"),
      Op::Get("expand_dims"),
      Op::Get("max"),
      Op::Get("maximum"),
      Op::Get("min"),
      Op::Get("minimum"),
      Op::Get("nn.avg_pool2d"),
      Op::Get("nn.batch_flatten"),
      Op::Get("nn.batch_matmul"),
      Op::Get("nn.bias_add"),
      Op::Get("nn.conv2d"),
      Op::Get("nn.conv2d_transpose"),
      Op::Get("nn.dense"),
      Op::Get("nn.depth_to_space"),
      Op::Get("nn.global_avg_pool2d"),
      Op::Get("nn.max_pool2d"),
      Op::Get("nn.pad"),
      Op::Get("nn.relu"),
      Op::Get("reshape"),
      Op::Get("split"),
      Op::Get("squeeze"),
      Op::Get("strided_slice"),
      Op::Get("transpose")};

  return ops.find(call_node->op) != ops.end() && fqfq.count(Downcast<Op>(op));
}

class QATSubgraphExtractor : public ExprVisitor {
 public:
  const ExprSet GetSubgraph(const Expr& expr) {
    expr_call_node_ = expr.as<CallNode>();
    ICHECK(expr_call_node_ != nullptr);
    ICHECK(is_op_enabled_for_optional_fq2i(expr_call_node_));

    VisitExpr(expr);

    ExprSet subgraph;
    if (is_fake_quantized_) {
      for (auto kv : this->visit_counter_) {
        if (auto call_node = GetRef<ObjectRef>(kv.first).as<CallNode>()) {
          if (call_node != expr_call_node_) {
            subgraph.insert(Downcast<Expr>(GetRef<ObjectRef>(kv.first)));
          }
        }
      }
    }
    return subgraph;
  }
  const AffineTypeMap GetAffineTypes() { return affine_types_; }
  void VisitExpr(const Expr& expr) override {
    // When looking for fake quantized subgraphs, we only support data-flow regions of the graph,
    // i.e. call nodes/tuples/constants/etc. If we see anything else (like control flow) we
    // abort the rewrite.
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
        expr.as<ConstantNode>() == nullptr) {
      DLOG(INFO) << "FakeQuantizationToInteger found a non - dataflow op inside a fake quantize "
                    "region, aborting this rewrite";
      is_fake_quantized_ = false;
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

 protected:
  void VisitExpr_(const CallNode* call_node) override {
    if (call_node->op == dequantize_op_) {
      const auto* attrs = call_node->attrs.as<qnn::DequantizeAttrs>();
      ICHECK(attrs != nullptr);

      affine_types_.Set(
          GetRef<Expr>(call_node),
          TensorAffineType(
              call_node->args[1], call_node->args[2],
              tvm::relay::transform::InferTypeLocal(call_node->args[0]).as<TensorTypeNode>()->dtype,
              attrs->axis));
    } else if (call_node == expr_call_node_) {
      for (auto arg : call_node->args) {
        VisitExpr(arg);
      }
    } else {
      // run normally on everything else.
      ExprVisitor::VisitExpr_(call_node);
    }
  }

  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  bool is_fake_quantized_ = true;
  AffineTypeMap affine_types_;
  const CallNode* expr_call_node_ = nullptr;
};

class QATSubgraphMutator : public ExprMutator {
 public:
  QATSubgraphMutator(ExprSet subgraph, AffineTypeMap affine_types, bool hard_fail,
                     const std::unordered_set<String>& optional_qnn_ops)
      : subgraph_(subgraph),
        affine_types_(affine_types),
        hard_fail_(hard_fail),
        optional_qnn_ops_(optional_qnn_ops) {}

  Expr MutateSubgraph(const Expr& expr) {
    if (subgraph_.size() == 0) {
      return expr;
    }

    quantize_node_ = expr.as<CallNode>();
    ICHECK(quantize_node_);
    ICHECK(is_op_enabled_for_optional_fq2i(quantize_node_));

    for (auto node : subgraph_) {
      const Op op = Downcast<Op>(node.as<CallNode>()->op);

      if (node.as<CallNode>()->op != dequantize_op_) {
        if (hard_fail_) {
          LOG(FATAL) << "Not dequantization was found in the input arguments for"
                     << AsText(op, false) << std::endl;
        } else {
          DLOG(INFO) << "Not dequantization was found in the input arguments for "
                     << AsText(op, false) << std::endl;
          return expr;
        }
      }
    }
    try {
      return Mutate(expr);
    } catch (std::exception& e) {
      if (hard_fail_) {
        throw e;
      } else {
        DLOG(INFO) << "Ran into an error rewriting a subgraph, skipping" << expr << std::endl;
        return expr;
      }
    }
  }

 protected:
  Expr VisitExpr_(const CallNode* call_node) {
    Expr out;
    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    static auto opt_fqfq =
        Op::HasAttrMap("FTVMOptionalFakeQuantizationToInteger")
            ? Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMOptionalFakeQuantizationToInteger")
            : fqfq;

    Op op = Downcast<Op>(call_node->op);
    if (fqfq.count(op) || (optional_qnn_ops_.count(op->name) && opt_fqfq.count(op))) {
      Expr expr;
      if (op == dequantize_op_) {
        expr = GetRef<Expr>(call_node);
      } else {
        expr = ExprMutator::VisitExpr_(call_node);
      }
      // Call the rewrite
      Array<ObjectRef> vals = (fqfq.count(op) ? fqfq : opt_fqfq)[op](expr, affine_types_);
      // Save the outputs of the rewrite
      ICHECK(vals.size() == 2)
          << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
          << AsText(op, false);
      out = Downcast<Expr>(vals[0]);

      affine_types_.Set(out, Downcast<AffineType>(vals[1]));

      if (call_node == quantize_node_) {
        out = qnn::MakeDequantize(out, vals[1].as<TensorAffineTypeNode>()->scale,
                                  vals[1].as<TensorAffineTypeNode>()->zero_point,
                                  vals[1].as<TensorAffineTypeNode>()->axis);
      }
    } else {
      ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
                    << AsText(GetRef<Expr>(call_node), false);
    }
    return out;
  }

  Expr VisitExpr_(const TupleNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto new_node = expr.as<TupleNode>();
    Array<TensorAffineType> types;
    for (Expr field : new_node->fields) {
      ICHECK(affine_types_[field].as<TensorAffineTypeNode>());
      types.push_back(Downcast<TensorAffineType>(affine_types_[field]));
    }
    affine_types_.Set(expr, TupleAffineType(types));
    return expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    Expr expr = ExprMutator::VisitExpr_(node);
    auto tuple_type = affine_types_[expr.as<TupleGetItemNode>()->tuple].as<TupleAffineTypeNode>();
    affine_types_.Set(expr, tuple_type->types[node->index]);
    return expr;
  }

  ExprSet subgraph_;
  AffineTypeMap affine_types_;
  const bool hard_fail_;
  const std::unordered_set<String>& optional_qnn_ops_;
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  const CallNode* quantize_node_ = nullptr;
};

class QATRewriter : public MixedModeMutator {
 public:
  explicit QATRewriter(bool hard_fail, const std::unordered_set<String>& optional_qnn_ops)
      : hard_fail_(hard_fail), optional_qnn_ops_(optional_qnn_ops) {}

 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call_node = post.as<CallNode>()) {
      const Op op = Downcast<Op>(call_node->op);
      if (is_op_enabled_for_optional_fq2i(call_node)) {
        QATSubgraphExtractor extractor;
        ExprSet subgraph = extractor.GetSubgraph(post);
        AffineTypeMap affine_types = extractor.GetAffineTypes();
        Expr out = QATSubgraphMutator(subgraph, affine_types, hard_fail_, optional_qnn_ops_)
                       .MutateSubgraph(post);
        return out;
      }
    }
    return post;
  }
  const bool hard_fail_;
  const std::unordered_set<String>& optional_qnn_ops_;
};

Expr FakeQuantizationToInteger(const Expr& expr, const IRModule& mod, bool hard_fail, bool use_qat,
                               const Array<String>& optional_qnn_ops) {
  const std::unordered_set<String> optional_qnn_ops_(optional_qnn_ops.begin(),
                                                     optional_qnn_ops.end());
  auto fq_expr = FakeQuantizationRewriter(hard_fail, optional_qnn_ops_).Mutate(expr);
  if (use_qat) {
    fq_expr = tvm::relay::InferType(fq_expr);
    fq_expr = QATRewriter(hard_fail, optional_qnn_ops_).Mutate(fq_expr);
  }
  return fq_expr;
}

namespace transform {

Pass FakeQuantizationToInteger(bool hard_fail, bool use_qat,
                               const Array<String>& optional_qnn_ops) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            FakeQuantizationToInteger(f, m, hard_fail, use_qat, optional_qnn_ops));
      };
  return CreateFunctionPass(pass_func, 0, "FakeQuantizationToInteger", {"InferType", "DivToMul"});
}

TVM_REGISTER_GLOBAL("relay._transform.FakeQuantizationToInteger")
    .set_body_typed(FakeQuantizationToInteger);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
