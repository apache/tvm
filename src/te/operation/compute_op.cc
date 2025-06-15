/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
5B * "License"); you may not use this file except in compliance
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
 * \brief Compute Op.
 * \file compute_op.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComputeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ComputeOpNode*>(node.get());
      p->stream << "compute(" << op->name << ", body=" << op->body << ", axis=" << op->axis
                << ", reduce_axis=" << op->reduce_axis << ", tag=" << op->tag
                << ", attrs=" << op->attrs << ")";
    });

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op);

static inline void AssertReduceEqual(const tir::ReduceNode* a, const tir::ReduceNode* b) {
  const char* shared_text =
      "When a TE compute node produces multiple outputs, "
      "each of which is a reduction, "
      "each reduction must be structurally identical, "
      "except for the ReduceNode::value_index.  ";

  StructuralEqual eq;

  ICHECK(a->combiner.same_as(b->combiner)) << shared_text << "However, the reduction operation "
                                           << a->combiner << " does not match " << b->combiner;
  ICHECK(a->source.same_as(b->source))
      << shared_text << "However, the input " << a->source << " does not match " << b->source;
  ICHECK(eq(a->axis, b->axis)) << shared_text << "However, the reduction axis " << a->axis
                               << " does not match " << b->axis;
  ICHECK(eq(a->condition, b->condition)) << shared_text << "However, the predicate " << a->condition
                                         << " does not match " << b->condition;
  ICHECK(eq(a->init, b->init)) << shared_text << "However, the initial value " << a->init
                               << " does not match " << b->init;
}

int ComputeOpNode::num_outputs() const { return body.size(); }

DataType ComputeOpNode::output_dtype(size_t idx) const {
  ICHECK_LT(idx, num_outputs());
  return body[idx].dtype();
}

Array<PrimExpr> BaseComputeOpNode::output_shape(size_t idx) const {
  ICHECK_LT(idx, num_outputs());
  // for now, all outputs of a BaseComputeOp have the same shape
  Array<PrimExpr> shape;
  for (const auto& ivar : this->axis) {
    const Range& r = ivar->dom;
    shape.push_back(r->extent);
  }
  return shape;
}

Tensor compute(Array<PrimExpr> shape, FCompute fcompute, std::string name, std::string tag,
               Map<String, ffi::Any> attrs) {
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(IntImm(shape[i]->dtype, 0), shape[i]),
                              Var(os.str(), shape[i].dtype()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOp(name, tag, attrs, axis, {fcompute(args)}).output(0);
}

Array<Tensor> compute(Array<PrimExpr> shape, FBatchCompute fcompute, std::string name,
                      std::string tag, Map<String, ffi::Any> attrs) {
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(IntImm(shape[i]->dtype, 0), shape[i]),
                              Var(os.str(), shape[i].dtype()), kDataPar));
    args.push_back(axis.back()->var);
  }

  Operation op = ComputeOp(name, tag, attrs, axis, fcompute(args));
  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

ComputeOp::ComputeOp(std::string name, std::string tag, Map<String, ffi::Any> attrs,
                     Array<IterVar> axis, Array<PrimExpr> body) {
  if (!attrs.defined()) {
    attrs = Map<String, ffi::Any>();
  }
  auto n = make_object<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->body = std::move(body);
  if (n->body[0]->IsInstance<tir::ReduceNode>()) {
    const tir::ReduceNode* reduce = n->body[0].as<tir::ReduceNode>();
    n->reduce_axis = reduce->axis;
  }
  VerifyComputeOp(n.get());
  data_ = std::move(n);
}

TVM_FFI_REGISTER_GLOBAL("te.ComputeOp")
    .set_body_typed([](std::string name, std::string tag, Optional<Map<String, ffi::Any>> attrs,
                       Array<IterVar> axis, Array<PrimExpr> body) {
      return ComputeOp(name, tag, attrs.value_or({}), axis, body);
    });

// The schedule related logics
Array<Tensor> ComputeOpNode::InputTensors() const {
  Array<Tensor> ret;
  std::unordered_set<Tensor> visited;
  for (auto& e : body) {
    tir::PostOrderVisit(e, [&ret, &visited](const ObjectRef& n) {
      if (auto* pload = n.as<tir::ProducerLoadNode>()) {
        Tensor t = Downcast<Tensor>(pload->producer);
        if (!visited.count(t)) {
          ret.push_back(t);
          visited.insert(t);
        }
      }
    });
  }
  return ret;
}

enum class ComputeType { kNormal, kCrossThreadReduction, kTensorize };

namespace {
/*!
 * \brief Verify if ComputeOp is valid with respect to Reduce operations.
 *
 *  The following two properties are verified:
 *  (1) All Reduce operations must exist at top level.
 *  (2) For a list of operations, if one is Reduce, then the others
 *      must be Reduce as well; and their inputs should have the
 *      same attribute except value_index.
 */
class ComputeVerifier final : protected tir::ExprVisitor {
 public:
  /// Special member functions
  //@{
  explicit ComputeVerifier(const ComputeOpNode* compute)
      : compute_(compute), reduce_(compute->body[0].as<tir::ReduceNode>()) {}
  virtual ~ComputeVerifier() = default;
  ComputeVerifier(const ComputeVerifier&) = delete;
  ComputeVerifier(ComputeVerifier&&) = delete;
  ComputeVerifier& operator=(const ComputeVerifier&) = delete;
  ComputeVerifier& operator=(ComputeVerifier&&) = delete;
  //@}

  /// Interface to perform compute verification
  void Run() {
    for (const PrimExpr e : compute_->body) {
      // Check for consistency of top level reductions
      const tir::ReduceNode* reduce = e.as<tir::ReduceNode>();
      ICHECK((reduce && reduce_) || (!reduce && !reduce_)) << "All ComputeOp should be consistent "
                                                           << "with being Reduce operation or not.";

      if (reduce && reduce_) {
        AssertReduceEqual(reduce, reduce_);
      }

      level_ = 0;
      ExprVisitor::VisitExpr(e);
    }
  }

 protected:
  /// Visitor implementation
  //@{
  void VisitExpr(const PrimExpr& n) final {
    ++level_;
    ExprVisitor::VisitExpr(n);
    --level_;
  }

  void VisitExpr_(const tir::ReduceNode* op) final {
    // Check for non top level reductions
    ICHECK(0 == level_) << "Reductions are only allowed at the top level of compute. "
                        << "Please create another tensor for further composition.";
  }
  //@}

 private:
  const ComputeOpNode* compute_{nullptr};   ///< ComputeOpNode to verify
  const tir::ReduceNode* reduce_{nullptr};  ///< Top level Reduce operation
  int level_{0};                            ///< Level of op being processed
};
}  // namespace

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op) {
  ComputeVerifier v(op);
  v.Run();
}

}  // namespace te
}  // namespace tvm
