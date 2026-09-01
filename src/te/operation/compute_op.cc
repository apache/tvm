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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace te {
using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  OperationNode::RegisterReflection();
  BaseComputeOpNode::RegisterReflection();
  ComputeOpNode::RegisterReflection();
}

// Pattern A (RM): auto-default repr from reflection.

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op);

static inline void AssertReduceEqual(const te::ReduceNode* a, const te::ReduceNode* b) {
  const char* shared_text =
      "When a TE compute node produces multiple outputs, "
      "each of which is a reduction, "
      "each reduction must be structurally identical, "
      "except for the ReduceNode::value_index.  ";

  ffi::StructuralEqual eq;

  TVM_FFI_ICHECK(a->combiner.same_as(b->combiner))
      << shared_text << "However, the reduction operation " << a->combiner << " does not match "
      << b->combiner;
  TVM_FFI_ICHECK(a->source.same_as(b->source))
      << shared_text << "However, the input " << a->source << " does not match " << b->source;
  TVM_FFI_ICHECK(eq(a->axis, b->axis))
      << shared_text << "However, the reduction axis " << a->axis << " does not match " << b->axis;
  TVM_FFI_ICHECK(eq(a->condition, b->condition))
      << shared_text << "However, the predicate " << a->condition << " does not match "
      << b->condition;
  TVM_FFI_ICHECK(eq(a->init, b->init))
      << shared_text << "However, the initial value " << a->init << " does not match " << b->init;
}

int ComputeOpNode::num_outputs() const { return body.size(); }

PrimType ComputeOpNode::output_dtype(size_t idx) const {
  TVM_FFI_ICHECK_LT(idx, num_outputs());
  return body[idx].ty();
}

ffi::Array<PrimExpr> BaseComputeOpNode::output_shape(size_t idx) const {
  TVM_FFI_ICHECK_LT(idx, num_outputs());
  // for now, all outputs of a BaseComputeOp have the same shape
  ffi::Array<PrimExpr> shape;
  for (const auto& ivar : this->axis) {
    const Range& r = ivar->dom;
    shape.push_back(r->extent);
  }
  return shape;
}

Tensor compute(ffi::Array<PrimExpr> shape, FCompute fcompute, std::string name, std::string tag,
               ffi::Map<ffi::String, ffi::Any> attrs) {
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<PrimVar> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(IntImm(shape[i].ty(), 0), shape[i]),
                              PrimVar(os.str(), shape[i].ty()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOp(name, tag, attrs, axis, {fcompute(args)}).output(0);
}

ffi::Array<Tensor> compute(ffi::Array<PrimExpr> shape, FBatchCompute fcompute, std::string name,
                           std::string tag, ffi::Map<ffi::String, ffi::Any> attrs) {
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<PrimVar> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(IntImm(shape[i].ty(), 0), shape[i]),
                              PrimVar(os.str(), shape[i].ty()), kDataPar));
    args.push_back(axis.back()->var);
  }

  Operation op = ComputeOp(name, tag, attrs, axis, fcompute(args));
  ffi::Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

ComputeOp::ComputeOp(std::string name, std::string tag, ffi::Map<ffi::String, ffi::Any> attrs,
                     ffi::Array<IterVar> axis, ffi::Array<PrimExpr> body) {
  if (!attrs.defined()) {
    attrs = ffi::Map<ffi::String, ffi::Any>();
  }
  auto n = ffi::make_object<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->body = std::move(body);
  if (n->body[0]->IsInstance<te::ReduceNode>()) {
    const te::ReduceNode* reduce = n->body[0].as<te::ReduceNode>();
    n->reduce_axis = reduce->axis;
  }
  VerifyComputeOp(n.get());
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("te.ComputeOp", [](std::string name, std::string tag,
                                           ffi::Optional<ffi::Map<ffi::String, ffi::Any>> attrs,
                                           ffi::Array<IterVar> axis, ffi::Array<PrimExpr> body) {
    return ComputeOp(name, tag, attrs.value_or({}), axis, body);
  });
}

// The schedule related logics
ffi::Array<Tensor> ComputeOpNode::InputTensors() const {
  ffi::Array<Tensor> ret;
  std::unordered_set<Tensor> visited;
  auto visit = [&ret, &visited](const PrimExpr& e) {
    tirx::PostOrderVisit(e, [&ret, &visited](const ffi::ObjectRef& n) {
      if (auto call = n.as<Call>(); call.has_value() && IsTensorLoad(call.value())) {
        Tensor t = GetTensorFromLoad(call.value());
        if (!visited.count(t)) {
          ret.push_back(t);
          visited.insert(t);
        }
      }
    });
  };
  for (const PrimExpr& e : body) {
    if (const auto* reduce = e.as<te::ReduceNode>()) {
      for (const IterVar& axis : reduce->axis) {
        visit(axis->dom->min);
        visit(axis->dom->extent);
      }
      for (const PrimExpr& source : reduce->source) visit(source);
      for (const PrimExpr& init : reduce->init) visit(init);
      visit(reduce->condition);
    } else {
      visit(e);
    }
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
class ComputeVerifier final : protected tirx::ExprVisitor {
 public:
  /// Special member functions
  //@{
  explicit ComputeVerifier(const ComputeOpNode* compute)
      : compute_(compute), reduce_(compute->body[0].as<te::ReduceNode>()) {}
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
      const te::ReduceNode* reduce = e.as<te::ReduceNode>();
      TVM_FFI_ICHECK((reduce && reduce_) || (!reduce && !reduce_))
          << "All ComputeOp should be consistent "
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
  void VisitExpr(const Expr& n) final {
    ++level_;
    ExprVisitor::VisitExpr(n);
    --level_;
  }

  void VisitExpr_(const OpaqueExprNode* op) final {
    const auto* reduce =
        op->IsInstance<te::ReduceNode>() ? static_cast<const te::ReduceNode*>(op) : nullptr;
    if (reduce == nullptr) {
      ExprVisitor::VisitExpr_(op);
      return;
    }

    TVM_FFI_ICHECK(0 == level_) << "Reductions are only allowed at the top level of compute. "
                                << "Please create another tensor for further composition.";
    for (const PrimExpr& expr : reduce->combiner->result) this->VisitExpr(expr);
    for (const PrimExpr& expr : reduce->combiner->identity_element) this->VisitExpr(expr);
    for (const PrimExpr& expr : reduce->source) this->VisitExpr(expr);
    for (const PrimExpr& expr : reduce->init) this->VisitExpr(expr);
    this->VisitExpr(reduce->condition);
  }
  //@}

 private:
  const ComputeOpNode* compute_{nullptr};  ///< ComputeOpNode to verify
  const te::ReduceNode* reduce_{nullptr};  ///< Top level Reduce operation
  int level_{0};                           ///< Level of op being processed
};
}  // namespace

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op) {
  ComputeVerifier v(op);
  v.Run();
}

}  // namespace te
}  // namespace tvm
