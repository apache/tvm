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

#include <tvm/ir/affine_type.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>

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
 */

using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using ExprMap = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>;
using AffineTypeMap = Map<Expr, AffineType>;

using FTVMFakeQuantizationToInteger =
    runtime::TypedPackedFunc<Array<ObjectRef>(const Expr& expr, const AffineTypeMap& map)>;

class SubgraphExtractor : public ExprVisitor {
 public:
  const ExprSet GetSubgraph(const Expr& expr) {
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
  const AffineTypeMap GetAffineTypes() { return affine_types_; }
  void VisitExpr(const Expr& expr) override {
    // When looking for fake quantized subgraphs, we only support data-flow regions of the graph,
    // i.e. call nodes/tuples/constants/etc. If we see anything else (like control flow) we
    // abort the rewrite.
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr && expr.as<TupleGetItemNode>() == nullptr &&
        expr.as<ConstantNode>() == nullptr) {
      LOG(INFO) << "FakeQuantizationToInteger found a non-dataflow op inside"
                << " a fake quantize region, aborting this rewrite";
      is_fake_quantized_ = false;
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

 protected:
  void VisitExpr_(const CallNode* call_node) override {
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

  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
  bool is_fake_quantized_ = true;
  AffineTypeMap affine_types_;
};

class SubgraphMutator : public ExprMutator {
 public:
  SubgraphMutator(ExprSet subgraph, AffineTypeMap affine_types)
      : subgraph_(subgraph), affine_types_(affine_types) {}

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
    for (auto node : subgraph_) {
      if (!fqfq.count(Downcast<Op>(node.as<CallNode>()->op))) {
        // Only modify the subgraph if we have translation
        // rules for every op
        return expr;
      }
    }
    return Mutate(expr);
  }

 protected:
  Expr VisitExpr_(const CallNode* call_node) {
    Expr out;

    static auto fqfq =
        Op::GetAttrMap<FTVMFakeQuantizationToInteger>("FTVMFakeQuantizationToInteger");
    Op op = Downcast<Op>(call_node->op);
    if (fqfq.count(op)) {
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
      Array<ObjectRef> vals = fqfq[op](expr, affine_types_);
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
  const Op quantize_op_ = Op::Get("qnn.quantize");
  const Op dequantize_op_ = Op::Get("qnn.dequantize");
};

class FakeQuantizationRewriter : public MixedModeMutator {
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
        Expr out = SubgraphMutator(post_subgraph, post_affine_types).MutateSubgraph(post);
        return out;
      }
    }
    return post;
  }
  const Op quantize_op_ = Op::Get("qnn.quantize");
};

Expr FakeQuantizationToInteger(const Expr& expr, const IRModule& mod) {
  return FakeQuantizationRewriter().Mutate(expr);
}

namespace transform {

Pass FakeQuantizationToInteger() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FakeQuantizationToInteger(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "FakeQuantizationToInteger", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FakeQuantizationToInteger")
    .set_body_typed(FakeQuantizationToInteger);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
