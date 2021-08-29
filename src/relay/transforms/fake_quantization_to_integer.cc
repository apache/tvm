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

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

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

namespace tvm {
namespace relay {

/*!
 * \brief AffineType representation
 * \sa AffineType
 */
class AffineTypeNode : public Object {
 public:
  /*! \brief The scale of this type */
  Expr scale;
  /*! \brief The zero point of this type */
  Expr zero_point;
  /*! \brief The data type of this type */
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("scale", &scale);
    v->Visit("zero_point", &zero_point);
    v->Visit("dtype", &dtype);
  }

  bool SEqualReduce(const AffineTypeNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(scale, other->scale) && equal(zero_point, other->zero_point) &&
           equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(scale);
    hash_reduce(zero_point);
    hash_reduce(dtype);
  }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "AffineTypeNode";
  TVM_DECLARE_BASE_OBJECT_INFO(AffineTypeNode, Object);
};

/*!
 * \brief Managed reference to AffineTypes.
 * \sa AffineTypeNode
 */
class AffineType : public ObjectRef {
 public:
  TVM_DLL AffineType(Expr scale, Expr zero_point, DataType dtype) {
    ObjectPtr<AffineTypeNode> n = make_object<AffineTypeNode>();
    n->scale = std::move(scale);
    n->zero_point = std::move(zero_point);
    n->dtype = std::move(dtype);
    data_ = std::move(n);
  }
  TVM_DEFINE_OBJECT_REF_METHODS(AffineType, ObjectRef, AffineTypeNode);
};

TVM_REGISTER_NODE_TYPE(AffineTypeNode);

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
    if (expr.as<CallNode>() == nullptr && expr.as<OpNode>() == nullptr &&
        expr.as<TupleNode>() == nullptr) {
      is_fake_quantized_ = false;
    } else {
      ExprVisitor::VisitExpr(expr);
    }
  }

 protected:
  void VisitExpr_(const CallNode* call_node) override {
    if (call_node->op == quantize_op_) {
      // Only look at arg0 for quantize
      VisitExpr(call_node->args[0]);
      // Collect type of quantize ops
      affine_types_.Set(GetRef<Expr>(call_node),
                        AffineType(call_node->args[1], call_node->args[2],
                                   call_node->checked_type().as<TensorTypeNode>()->dtype));
    } else if (call_node->op == dequantize_op_) {
      // Collect type of dequantize ops
      affine_types_.Set(GetRef<Expr>(call_node),
                        AffineType(call_node->args[1], call_node->args[2],
                                   call_node->args[0]->checked_type().as<TensorTypeNode>()->dtype));
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
      // Save teh outputs of the rewrite
      ICHECK(vals.size() == 4)
          << "got the wrong number of returned arguments from FTVMFakeQuantizationToInteger for "
          << AsText(op, false);
      out = Downcast<Expr>(vals[0]);
      affine_types_.Set(out, AffineType(Downcast<Expr>(vals[1]), Downcast<Expr>(vals[2]),
                                        DataType(String2DLDataType(Downcast<String>(vals[3])))));
    } else {
      ICHECK(false) << "When rewriting a fake quantized graph, found an invalid node "
                    << AsText(GetRef<Expr>(call_node), false);
    }
    return out;
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
