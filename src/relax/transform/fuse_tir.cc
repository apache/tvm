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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

#include "../../tirx/ir/functor_common.h"

namespace tvm {
namespace tirx {

/*!
 * \brief Match symbolic vars according to the given PrimExpr, and update the var_remap.
 * Will throw errors if there is a mismatch.
 */
class SymbolicMatcher : ExprFunctor<void(const Expr& n, const PrimExpr& other)> {
 public:
  explicit SymbolicMatcher(arith::AnalyzerObj* analyzer, ffi::Map<tirx::Var, PrimExpr>* var_remap)
      : analyzer_(analyzer), var_remap_(var_remap) {}

  void Match(const ffi::Array<PrimExpr>& params, const ffi::Array<PrimExpr>& args) {
    TVM_FFI_ICHECK_EQ(params.size(), args.size());
    for (size_t i = 0; i < params.size(); ++i) {
      Match(params[i], args[i]);
    }
  }
  void Match(const PrimExpr& param, const PrimExpr& arg) {
    VisitExpr(param, arg);
    must_prove_ = analyzer_->Simplify(Substitute(must_prove_, *var_remap_));
    TVM_FFI_ICHECK(!is_zero(must_prove_));
  }

 private:
  void VisitExpr(const Expr& expr, const PrimExpr& other) final {
    PrimExpr node = expr.as_or_throw<PrimExpr>();
    if (node.same_as(other)) {
      return;
    } else if (node.ty().code() != other.ty().code()) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << node << " with dtype " << node.ty()->dtype
          << " cannot match to argument " << other << " with dtype " << other.ty()->dtype;
    } else {
      ExprFunctor::VisitExpr(expr, other);
    }
  }

#define TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(OpName)                       \
  void VisitExpr_(const OpName* op, const PrimExpr& other) {             \
    const auto* rhs = other.as<OpName>();                                \
    if (rhs) {                                                           \
      VisitExpr(op->a, rhs->a);                                          \
      VisitExpr(op->b, rhs->b);                                          \
    } else {                                                             \
      must_prove_ = must_prove_ && (ffi::GetRef<PrimExpr>(op) == other); \
    }                                                                    \
  }

  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(AddNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(SubNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(MulNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(DivNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(ModNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(EQNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(NENode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(LTNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(LENode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(GTNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(GENode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(AndNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(OrNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(MinNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(MaxNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(FloorDivNode);
  TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(FloorModNode);

  void VisitExpr_(const IntImmNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<IntImmNode>();
    if (!rhs || (op->value != rhs->value)) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << ffi::GetRef<PrimExpr>(op)
          << " expected an integer argument with value " << op->value << ", "
          << "but was provided with the argument " << other;
    }
  }

  void VisitExpr_(const FloatImmNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<FloatImmNode>();
    if (!rhs || (op->value != rhs->value)) {
      TVM_FFI_THROW(InternalError) << "Parameter expression " << ffi::GetRef<PrimExpr>(op)
                                   << " expected an float argument with value " << op->value << ", "
                                   << "but was provided with the argument " << other;
    }
  }

  void VisitExpr_(const CastNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<CastNode>();
    if (!rhs) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << ffi::GetRef<PrimExpr>(op) << " expected an cast to "
          << op->ty.as_or_throw<PrimType>()->dtype << " as the argument, "
          << "but was provided with the argument " << other;
    }
    VisitExpr(op->value, rhs->value);
  }

  void VisitExpr_(const VarNode* op, const PrimExpr& rhs) {
    auto lhs = ffi::GetRef<Var>(op);
    PrimType lhs_ty = op->ty.as_or_throw<PrimType>();

    if (lhs.same_as(rhs)) {
      // Reference identity, no further checks needed.
    } else if (lhs_ty.code() != rhs.ty().code()) {
      TVM_FFI_THROW(InternalError)
          << "Parameter expression " << lhs << " with dtype " << lhs_ty->dtype
          << " cannot match to argument " << rhs << " with dtype " << rhs.ty()->dtype;
    } else if (auto it = var_remap_->find(lhs); it != var_remap_->end()) {
      VisitExpr((*it).second, rhs);
    } else {
      var_remap_->Set(lhs, rhs);
    }
  }

  void VisitExpr_(const SelectNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<SelectNode>();
    if (rhs) {
      VisitExpr(op->true_value, rhs->true_value);
      VisitExpr(op->false_value, rhs->false_value);
    } else {
      must_prove_ = must_prove_ && (ffi::GetRef<PrimExpr>(op) == other);
    }
  }

  arith::AnalyzerObj* analyzer_;
  ffi::Map<tirx::Var, PrimExpr>* var_remap_;
  PrimExpr must_prove_ = IntImm::Bool(true);
};

/*!
 * \brief Substitute a given source buffer with a given target buffer in statements or expressions.
 */
class FuseTIRBufferSubstitutor : private StmtExprMutator {
 public:
  explicit FuseTIRBufferSubstitutor(const ffi::Map<Buffer, Buffer>& buffer_map,
                                    const ffi::Map<Var, PrimExpr>& var_map) {
    buffer_remap_ = buffer_map;
    for (const auto& [var, value] : var_map) {
      var_remap_.Set(var, value);
    }
    for (const auto& [src, tgt] : buffer_map) {
      var_remap_.Set(src->data, tgt->data);
    }
  }

  Stmt Substitute(Stmt stmt) { return this->VisitStmt(std::move(stmt)); }

  Buffer SubstituteAllocatedBuffer(Buffer buffer) {
    TVM_FFI_ICHECK(buffer_remap_.find(buffer) == buffer_remap_.end());
    ffi::Array<PrimExpr> shape = MutateArray(
        buffer->shape, [this](const PrimExpr& expr) { return this->VisitPrimExpr(expr); });
    ffi::Array<PrimExpr> strides = MutateArray(
        buffer->strides, [this](const PrimExpr& expr) { return this->VisitPrimExpr(expr); });
    PrimExpr elem_offset = this->VisitPrimExpr(buffer->elem_offset);
    if (shape.same_as(buffer->shape) && strides.same_as(buffer->strides) &&
        elem_offset.same_as(buffer->elem_offset)) {
      return buffer;
    } else {
      auto n = ffi::make_object<BufferNode>(*buffer.get());
      n->shape = std::move(shape);
      n->strides = std::move(strides);
      n->elem_offset = std::move(elem_offset);
      Buffer new_buffer(n);
      this->buffer_remap_.Set(buffer, new_buffer);
      return new_buffer;
    }
  }

 private:
  Expr VisitExpr_(const VarNode* _op) final {
    if (auto it = var_remap_.find(ffi::GetRef<Var>(_op)); it != var_remap_.end()) {
      return (*it).second;
    } else {
      return ffi::GetRef<Var>(_op);
    }
  }

  Expr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = StmtExprMutator::VisitExpr_(_op).as_or_throw<BufferLoad>();
    const Buffer& buffer = SubstituteBuffer(load->buffer);
    if (buffer.same_as(load->buffer)) {
      return load;

    } else {
      auto n = ffi::make_object<BufferLoadNode>(*load.get());
      n->buffer = buffer;
      return BufferLoad(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = StmtExprMutator::VisitStmt_(_op).as_or_throw<BufferStore>();
    const Buffer& buffer = SubstituteBuffer(store->buffer);
    if (buffer.same_as(store->buffer)) {
      return store;

    } else {
      auto n = ffi::make_object<BufferStoreNode>(*store.get());
      n->buffer = buffer;
      return BufferStore(n);
    }
  }

  Stmt VisitStmt_(const SBlockNode* _op) final {
    SBlock block = StmtMutator::VisitStmt_(_op).as_or_throw<SBlock>();

    // Define the mutation functions.

    auto f_mutate_match_buffers = [this](const MatchBufferRegion& match_buffer) {
      const Buffer& src_buffer = SubstituteBuffer(match_buffer->source->buffer);
      const Buffer& tgt_buffer = SubstituteAllocatedBuffer(match_buffer->buffer);
      ffi::Array<Range> region = MutateRegion(match_buffer->source->region);
      if (src_buffer.same_as(match_buffer->source->buffer) &&
          tgt_buffer.same_as(match_buffer->buffer) &&
          region.same_as(match_buffer->source->region)) {
        return match_buffer;
      } else {
        auto n = ffi::make_object<MatchBufferRegionNode>(*match_buffer.get());
        n->buffer = tgt_buffer;
        n->source = BufferRegion(src_buffer, region);
        return MatchBufferRegion(n);
      }
    };

    auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
      const Buffer& buffer = SubstituteBuffer(buffer_region->buffer);
      const ffi::Array<Range>& region = MutateRegion(buffer_region->region);
      if (buffer.same_as(buffer_region->buffer) && region.same_as(buffer_region->region)) {
        return buffer_region;
      } else {
        return BufferRegion(buffer, region);
      }
    };

    // Step 1. Mutate `match_buffers`.
    ffi::Array<MatchBufferRegion> match_buffers =
        MutateArray(block->match_buffers, f_mutate_match_buffers);
    // Step 2. Mutate the read/write region.
    ffi::Array<BufferRegion> reads = MutateArray(block->reads, f_mutate_read_write_region);
    ffi::Array<BufferRegion> writes = MutateArray(block->writes, f_mutate_read_write_region);
    // Step 3. Mutate the Allocate Buffers.
    ffi::Array<Buffer> alloc_buffers =
        MutateArray(block->alloc_buffers,
                    [this](const Buffer& buffer) { return SubstituteAllocatedBuffer(buffer); });

    reads = UnionAccessRegion(reads);
    writes = UnionAccessRegion(writes);

    if (reads.same_as(block->reads) &&    //
        writes.same_as(block->writes) &&  //
        match_buffers.same_as(block->match_buffers) &&
        alloc_buffers.same_as(block->alloc_buffers)) {
      return block;

    } else {
      auto n = CopyOnWrite(block.get());
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->match_buffers = std::move(match_buffers);
      n->alloc_buffers = std::move(alloc_buffers);
      return SBlock(n);
    }
  }

 private:
  /*! \brief Mapping from src buffer to tgt buffer. */
  ffi::Map<tirx::Buffer, tirx::Buffer> buffer_remap_;
  /*! \brief Mapping from src tirx var to tgt var. */
  ffi::Map<tirx::Var, Expr> var_remap_;

  ffi::Array<tirx::BufferRegion> UnionAccessRegion(const ffi::Array<BufferRegion>& regions) const {
    // For now we only allow Buffer access the same elements.
    // e.g. `[A[vi, vj], A[vi, vj]]` is a legal pattern but need to union to `A[vi, vj]`
    // However, `A[vi, vj], A[vi, vj + 1]` is not allow for now.
    // Note: the order of return region should remain the same as the first occurrence of the region
    ffi::Array<BufferRegion> ret;
    std::unordered_map<const BufferNode*, ffi::Array<Range>> buffer_region_set;

    for (const BufferRegion& region : regions) {
      auto it = buffer_region_set.find(region->buffer.get());
      if (it == buffer_region_set.end()) {
        ret.push_back(region);
        buffer_region_set[region->buffer.get()] = region->region;
      }
    }

    if (ret.size() == regions.size()) {
      return regions;
    } else {
      return ret;
    }
  }

  inline Buffer SubstituteBuffer(const Buffer& buffer) const {
    auto it = buffer_remap_.find(buffer);
    if (it != buffer_remap_.end()) {
      return (*it).second;
    } else {
      return buffer;
    }
  }

  inline ffi::Array<Range> MutateRegion(const ffi::Array<Range>& region) {
    return MutateArray(region, [this](const Range& range) {
      PrimExpr min = this->VisitPrimExpr(range->min);
      PrimExpr extent = this->VisitPrimExpr(range->extent);
      if (min.same_as(range->min) && extent.same_as(range->extent)) {
        return range;
      } else {
        return Range::FromMinExtent(min, extent);
      }
    });
  }
};

/*! \brief A mutator which detect block name duplication and deduplicate the names. */
class SBlockNameDeduplicator : public tirx::StmtMutator {
 private:
  Stmt VisitStmt_(const SBlockNode* op) final {
    SBlock block = tirx::StmtMutator::VisitStmt_(op).as_or_throw<SBlock>();

    ffi::String name = GetUniqueName(block->name_hint);

    if (name == block->name_hint) {
      return block;

    } else {
      ffi::ObjectPtr<SBlockNode> n = CopyOnWrite(block.get());
      n->name_hint = std::move(name);
      return Stmt(n);
    }
  }

  ffi::String GetUniqueName(const ffi::String& prefix) {
    std::string str_prefix = std::string(prefix);

    // Find where the trailing digits start
    size_t base_len = str_prefix.length();
    while (base_len > 0 && std::isdigit(str_prefix[base_len - 1])) {
      --base_len;
    }

    std::string base_name;
    int64_t start_num = 0;
    bool has_suffix = base_len < str_prefix.length();

    if (has_suffix) {
      base_name = str_prefix.substr(0, base_len);
      try {
        start_num = std::stoll(str_prefix.substr(base_len));
      } catch (const std::out_of_range&) {
        // Fallback: if the number is too large, treat the whole string as a base name.
        has_suffix = false;
        base_name = str_prefix;
      }
    } else {
      base_name = str_prefix;
    }

    // Check if the original name is available
    ffi::String candidate = prefix;
    if (!name_count_.count(candidate)) {
      name_count_[candidate] = 0;
      return candidate;
    }

    // Generate unique name by incrementing the numeric suffix
    int64_t counter = has_suffix ? start_num + 1 : 1;
    while (true) {
      candidate = ffi::String(base_name + std::to_string(counter));
      if (!name_count_.count(candidate)) {
        name_count_[candidate] = 0;
        return candidate;
      }
      ++counter;
      TVM_FFI_ICHECK_GT(counter, 0)
          << "Counter overflow when generating unique block name for prefix: " << prefix;
    }
  }

  /*! \brief The count map to make block name unique. */
  std::unordered_map<ffi::String, int> name_count_;
};

}  // namespace tirx

namespace relax {

static ffi::Array<int64_t> GetInplaceOutputIndices(const ffi::Array<int64_t>& inplace_indices,
                                                   int num_inputs) {
  ffi::Array<int64_t> ret;
  int last_idx = num_inputs;
  for (int64_t i : inplace_indices) {
    if (i >= 0) {
      ret.push_back(i);
    } else {
      TVM_FFI_ICHECK_EQ(i, -1)
          << "The only negative index expected in inplace_indices is -1, but got " << i;
      ret.push_back(last_idx);
      last_idx++;
    }
  }

  return ret;
}

class RelaxToTIRVarMapCollector : public ExprVisitor {
 public:
  explicit RelaxToTIRVarMapCollector(const IRModule& mod) : mod_(mod) {}
  static ffi::Map<Expr, tirx::Buffer> Collect(const IRModule& mod, const Function& func) {
    RelaxToTIRVarMapCollector visitor(mod);
    visitor(func->body);
    return visitor.relax_to_tir_var_map_;
  }

 private:
  void VisitBinding_(const VarBindingNode* binding) final {
    current_var_ = binding->var;
    ExprVisitor::VisitBinding_(binding);
  }

  void VisitExpr_(const CallNode* call) {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");

    TVM_FFI_ICHECK(call->op.same_as(call_tir_op_) || call->op.same_as(call_tir_inplace_op_))
        << "Only call_tir and call_tir_inplace are supported in primitive function, but got: "
        << ffi::GetRef<Expr>(call);
    CollectVarMapping(call, current_var_, call->op.same_as(call_tir_inplace_op_));
  }

  void CollectVarMapping(const CallNode* call, const Expr& lhs_var, bool in_place) {
    GlobalVar gv = call->args[0].as_or_throw<GlobalVar>();
    tirx::PrimFunc prim_func_ = mod_->Lookup(gv).as_or_throw<tirx::PrimFunc>();
    const auto& buffer_map = prim_func_->buffer_map;
    const auto& tir_args = prim_func_->params;

    const auto& relax_args = call->args[1].as_or_throw<Tuple>()->fields;

    ffi::Array<Expr> relax_results;
    if (lhs_var->IsInstance<TupleNode>()) {
      relax_results = lhs_var.as_or_throw<Tuple>()->fields;
    } else {
      TVM_FFI_ICHECK(lhs_var->IsInstance<VarNode>())
          << "The lhs_var is expected to be either tuple or var";
      relax_results = {lhs_var.as_or_throw<Var>()};
    }

    size_t num_inputs = relax_args.size();
    size_t num_outputs = relax_results.size();

    ffi::Array<int64_t> output_idxs;
    if (in_place) {
      const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>();
      TVM_FFI_ICHECK(attrs) << "Must have CallTIRInplaceAttrs for an in-place call";
      output_idxs = GetInplaceOutputIndices(attrs->inplace_indices, num_inputs);
    } else {
      for (size_t i = num_inputs; i < num_inputs + num_outputs; i++) {
        output_idxs.push_back(i);
      }
    }

    // If the `expr` is already seen (present in the map), validate whether the mapped buffer is
    // structurally equal to the `new_buf` passed
    auto ValidateBufferCompatibility = [this](tirx::Buffer new_buf, Expr expr) {
      if (auto it = relax_to_tir_var_map_.find(expr); it != relax_to_tir_var_map_.end()) {
        TVM_FFI_ICHECK(ffi::StructuralEqual()((*it).second, new_buf))
            << "Inconsistent buffers " << (*it).second << " and " << new_buf
            << " mapped to the same relax var: " << expr;
      }
    };
    for (size_t i = 0; i < tir_args.size(); ++i) {
      const auto& tir_var = tir_args[i];
      if (auto tir_buffer = buffer_map.Get(tir_var)) {
        if (i < num_inputs) {
          const auto& relax_var = relax_args[i];
          ValidateBufferCompatibility(tir_buffer.value(), relax_var);
          relax_to_tir_var_map_.Set(relax_var, tir_buffer.value());
        }
        if (auto it = std::find(output_idxs.begin(), output_idxs.end(), i);
            it != output_idxs.end()) {
          int result_idx = it - output_idxs.begin();
          const auto& relax_var = relax_results[result_idx];
          ValidateBufferCompatibility(tir_buffer.value(), relax_var);
          relax_to_tir_var_map_.Set(relax_var, tir_buffer.value());
        }
      }
    }
  }

 private:
  /*! \brief The IRModule */
  const IRModule& mod_;
  ffi::Map<Expr, tirx::Buffer> relax_to_tir_var_map_;
  Var current_var_;
};

class FusedTIRConstructor : public ExprVisitor {
 public:
  /*!
   * \brief Construct a fused TIR PrimFunc from a relax sub-function
   * \param mod The IRModule
   * \param gv The global var of relax subfunction to be fused into one PrimFunc
   * \return The fused TIR PrimFunc and the in-place indices (non-empty for an in-place call)
   */
  static std::pair<tirx::PrimFunc, ffi::Array<int64_t>> GetFusedTIR(const IRModule& mod,
                                                                    const GlobalVar& gv) {
    FusedTIRConstructor visitor(mod, gv->name_hint);
    BaseFunc f = mod->Lookup(gv);
    TVM_FFI_ICHECK(f->IsInstance<relax::FunctionNode>())
        << "Expected relax functions, but got: " << f->GetTypeKey();
    TVM_FFI_ICHECK(f->HasNonzeroAttr(relax::attr::kPrimitive))
        << "Expected a function with attr `kPrimitive`";
    visitor(f.as_or_throw<relax::Function>());
    ffi::Array<int64_t> inplace_indices;
    for (size_t idx : visitor.inplace_indices_) {
      inplace_indices.push_back(static_cast<int64_t>(idx));
    }
    return {visitor.fused_tir_, inplace_indices};
  }

 private:
  explicit FusedTIRConstructor(const IRModule& mod, const ffi::String& func_name)
      : mod_(mod), func_name_(func_name) {}

  void VisitExpr_(const FunctionNode* func) final {
    auto relax_to_tir_var_map =
        RelaxToTIRVarMapCollector::Collect(mod_, ffi::GetRef<Function>(func));
    std::vector<ffi::Variant<tirx::PrimVar, tirx::Buffer>> prim_func_params;
    for (const Var& relax_param : func->params) {
      size_t size_before = prim_func_params.size();
      CollectPrimFuncParams(relax_param, &prim_func_params, relax_to_tir_var_map.Get(relax_param));

      auto param_buffers = [&]() -> ffi::Array<tirx::Buffer> {
        ffi::Array<tirx::Buffer> out;
        for (size_t i = size_before; i < prim_func_params.size(); i++) {
          if (auto buf = prim_func_params[i].as<tirx::Buffer>()) {
            out.push_back(buf.value());
          }
        }
        return out;
      }();

      func_info_.expr2buffers.Set(relax_param, param_buffers);
    }

    // Preserve the Relax function's parameter order.  Tensor and primitive
    // parameters are both explicit call_tir arguments, while output buffers
    // are appended after the complete explicit argument prefix.
    for (const auto& param : prim_func_params) {
      if (auto opt = param.as<tirx::Buffer>()) {
        auto buffer = opt.value();
        // Differentiate buffer name and param name by adding prefix
        // `p_` to the buffer name.  Every symbol should be unique in
        // TVMScript, and while they can be de-deplicated when
        // printed, it's more readable when done explicitly.  Since
        // Buffer is used more than param it gets the name with better
        // readability.
        tirx::Var param = tirx::Var("p_" + buffer->name, PointerType::VoidPointerTy());
        func_info_.params.push_back(param);
        func_info_.buffer_map.Set(param, buffer);
      } else if (auto var = param.as<tirx::PrimVar>()) {
        func_info_.params.push_back(var.value());
      }
    }

    // Step 2. Visit Function body and create intermediate buffers
    ExprVisitor::VisitExpr_(func);

    // Step 3. Create and remap buffers for function output
    Expr body = func->body->body;
    auto it = func_info_.expr2buffers.find(body);
    TVM_FFI_ICHECK(it != func_info_.expr2buffers.end())
        << "Fail to detect output buffers for function body";

    const ffi::Array<tirx::Buffer>& buffers = (*it).second;

    // map of input buffers to indices (helpful for detecting in-place inputs)
    std::unordered_map<tirx::Buffer, size_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_to_idx;
    std::unordered_map<tirx::Var, size_t> input_to_idx;
    for (size_t i = 0; i < func_info_.params.size(); i++) {
      input_to_idx[func_info_.params[i]] = i;
    }
    for (auto [var, buffer] : func_info_.buffer_map) {
      if (auto it = input_to_idx.find(var); it != input_to_idx.end()) {
        buffer_to_idx[buffer] = (*it).second;
      }
    }

    // numbered separately because the number of output *vars* might differ from the
    // number of outputs if there are in-place inputs
    int out_idx = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
      // Do not add output vars for in-place inputs
      // (i.e., already listed in the buffer map. This would result
      // in duplicates in the buffer map otherwise)
      if (auto it = buffer_to_idx.find(buffers[i]); it != buffer_to_idx.end()) {
        auto idx = (*it).second;
        TVM_FFI_ICHECK(!inplace_indices_.count(idx))
            << "In-place index " << idx << " used twice! An argument must be aliased.";
        inplace_indices_.insert(idx);
        continue;
      }

      tirx::Var param =
          tirx::Var("p_output" + std::to_string(out_idx), PointerType::VoidPointerTy());
      out_idx++;
      func_info_.buffer_map.Set(param, buffers[i]);
      func_info_.params.push_back(param);
      func_info_.output_buffers.insert(buffers[i].get());
    }

    // Step 4. Create PrimFunc
    fused_tir_ = ConstructFunc();
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    // Update expr2buffers by visiting values.
    this->VisitExpr(binding->value);
    auto it = func_info_.expr2buffers.find(binding->value);
    if (it != func_info_.expr2buffers.end()) {
      // assign binding var to the buffers of the value
      func_info_.expr2buffers.Set(binding->var, (*it).second);
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported binding value: " << binding->value;
    }
  }

  void VisitBinding_(const MatchCastNode* match_cast) final {
    TVM_FFI_THROW(InternalError) << "MatchCast is unsupported in primitive functions";
  }

  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");

    TVM_FFI_ICHECK(call->op.same_as(call_tir_op_) || call->op.same_as(call_tir_inplace_op_))
        << "Only call_tir and call_tir_inplace are supported in primitive function, but got: "
        << ffi::GetRef<Expr>(call);

    // Step 1. Get Global var and PrimFunc
    GlobalVar gv = call->args[0].as_or_throw<GlobalVar>();
    tirx::PrimFunc prim_func_ = mod_->Lookup(gv).as_or_throw<tirx::PrimFunc>();

    // Step 2. Renew all vars/buffer definitions and blocks to avoid duplication
    tirx::PrimFunc prim_func = s_tir::RenewDefs(prim_func_);

    // Step 3. Check functions are all schedulable funcs. i.e. the body of func is root block
    // TODO(Siyuan): support un-schedulable functions.
    TVM_FFI_ICHECK(prim_func->body->IsInstance<tirx::SBlockRealizeNode>())
        << "Only schedulable functions (whose body is the root block) can be fused";
    const tirx::SBlockRealize& root_realize = prim_func->body.as_or_throw<tirx::SBlockRealize>();
    const tirx::SBlock& root_block = root_realize->block;

    // Step 4. Add all the original alloc_buffers and body to the fused function.
    func_info_.alloc_buffers.insert(func_info_.alloc_buffers.end(),
                                    root_block->alloc_buffers.begin(),
                                    root_block->alloc_buffers.end());
    func_info_.bodies.push_back(root_block->body);

    // Step 5. Map input arguments to buffer
    MapInputBuffer(prim_func, call->args[1]);
    const ffi::Array<ffi::Array<PrimExpr>>& output_buffer_shapes = GetCallTIROutputShapes(call);

    AllocateIntermediateBuffer(call, prim_func, output_buffer_shapes);

    // Update fused func name
    func_info_.global_name += "_" + gv->name_hint;
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item) final {
    ExprVisitor::VisitExpr_(tuple_get_item);
    auto it = func_info_.expr2buffers.find(tuple_get_item->tuple);
    if (it != func_info_.expr2buffers.end()) {
      int begin_buf_idx = 0;
      int end_buf_idx = 0;
      const TupleType& tuple_ty = tuple_get_item->tuple->ty.as_or_throw<TupleType>();
      for (int i = 0; i < tuple_get_item->index; ++i) {
        begin_buf_idx += GetTotalTensorSize(tuple_ty->fields[i]);
      }
      end_buf_idx = begin_buf_idx + GetTotalTensorSize(tuple_ty->fields[tuple_get_item->index]);
      func_info_.expr2buffers.Set(
          ffi::GetRef<Expr>(tuple_get_item),
          {(*it).second.begin() + begin_buf_idx, (*it).second.begin() + end_buf_idx});
    }
  }

  void VisitExpr_(const TupleNode* tuple) final {
    ExprVisitor::VisitExpr_(tuple);
    ffi::Array<tirx::Buffer> buffers;
    for (const Expr& expr : tuple->fields) {
      auto it = func_info_.expr2buffers.find(expr);
      if (it != func_info_.expr2buffers.end()) {
        buffers.insert(buffers.end(), (*it).second.begin(), (*it).second.end());
      }
    }
    if (!buffers.empty()) {
      func_info_.expr2buffers.Set(ffi::GetRef<Expr>(tuple), buffers);
    }
  }

  void VisitExpr_(const ConstantNode* op) final {
    TVM_FFI_THROW(InternalError) << "Relax.Constant is not supported in primitive functions.";
  }

  /*!
   * \brief Get the number of outputs for a call_tir node.
   * \return The number of outputs.
   */
  static ffi::Array<ffi::Array<PrimExpr>> GetCallTIROutputShapes(const CallNode* call) {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");
    TVM_FFI_ICHECK(call->op.same_as(call_tir_op_) || call->op.same_as(call_tir_inplace_op_));
    TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1);
    auto get_tensor_shape =
        [](const TensorTypeNode* ty) {
          const auto* shape_expr = ty->shape.as<ShapeExprNode>();
          TVM_FFI_ICHECK(shape_expr)
              << "FuseTIR expects all parameters are Tensors with symbolic shape.";
          return shape_expr->values;
        };
    if (const auto* tuple_ty = call->ty_args[0].as<TupleTypeNode>()) {
      ffi::Array<ffi::Array<PrimExpr>> shapes;
      for (const Type& field : tuple_ty->fields) {
        const auto* tensor_ty = field.as<TensorTypeNode>();
        TVM_FFI_ICHECK(tensor_ty) << "CallTIR ty_args are expected to be TensorType or Tuple of "
                                     "TensorType, but got "
                                  << call->ty_args[0];
        shapes.push_back(get_tensor_shape(tensor_ty));
      }
      return shapes;
    } else if (const auto* tensor_ty = call->ty_args[0].as<TensorTypeNode>()) {
      return {get_tensor_shape(tensor_ty)};
    } else {
      TVM_FFI_ICHECK(tensor_ty) << "CallTIR ty_args are expected to be TensorType or Tuple of "
                                   "TensorType, but got "
                                << call->ty_args[0];
      throw;
    }
  }

  /*! \brief Map old TIR func param buffer to new buffer, and then update `buffer_subst_map` */
  void MapArgsToBuffer(const ffi::Array<Expr> args, const ffi::Array<tirx::Buffer>& buffers) {
    size_t buffer_idx = 0;
    for (const Expr& arg : args) {
      if (const auto* v = arg.as<VarNode>()) {
        auto it = func_info_.expr2buffers.find(ffi::GetRef<Var>(v));
        // Substitute the buffer with the already allocated one if it is an intermediate var
        if (it != func_info_.expr2buffers.end()) {
          for (const tirx::Buffer& target_buffer : (*it).second) {
            TVM_FFI_ICHECK_LT(buffer_idx, buffers.size());
            const tirx::Buffer& buffer = buffers[buffer_idx];
            func_info_.symbolic_var_matcher.Match(buffer->shape, target_buffer->shape);
            func_info_.buffer_subst_map.Set(buffer, target_buffer);
            buffer_idx++;
          }
        }
      }
    }
    // Make sure every buffer is mapped.
    TVM_FFI_ICHECK_EQ(buffer_idx, buffers.size());
  }

  /*!
   * \brief Update buffer mapping `func_info_.buffer_subst_map` for input args
   * \param func The old TIR PrimFunc
   * \param output_size The number of output params. All output params are at the end of param list.
   */
  void MapInputBuffer(const tirx::PrimFunc& func, const relax::Expr& args) {
    ffi::Array<Expr> arg_list;
    ffi::Array<tirx::Buffer> buffer_list;
    ffi::Array<Expr> call_args = args.as_or_throw<Tuple>()->fields;

    TVM_FFI_ICHECK_GE(func->params.size(), call_args.size());
    for (size_t i = 0; i < call_args.size(); ++i) {
      const Expr& arg = call_args[i];
      const tirx::Var& param = func->params[i];
      if (func->buffer_map.count(param)) {
        arg_list.push_back(arg);
        buffer_list.push_back(func->buffer_map.at(param));
      } else {
        auto prim_arg = arg.as<PrimExpr>();
        TVM_FFI_CHECK(prim_arg.has_value(), TypeError)
            << "Expected scalar parameter " << param
            << " to receive an individual primitive expression, but " << arg << " has type "
            << GetType(arg);
        func_info_.symbolic_var_matcher.Match(param.as_or_throw<PrimExpr>(), prim_arg.value());
      }
    }

    MapArgsToBuffer(arg_list, buffer_list);
  }

  static ffi::Array<tirx::Var> GetPrimFuncOutputParams(const tirx::PrimFunc& func,
                                                       const ffi::Array<int64_t>& output_indices) {
    size_t n = func->params.size();
    size_t output_size = output_indices.size();
    TVM_FFI_ICHECK_GE(n, output_size);

    ffi::Array<tirx::Var> ret;
    for (int64_t idx : output_indices) {
      int i = static_cast<int>(idx);
      const tirx::Var& param = func->params[static_cast<size_t>(i)];
      TVM_FFI_ICHECK(param->ty.as<PointerTypeNode>())
          << "The output params of a PrimFunc must be buffer handles, but parameter " << i
          << " has type " << param->ty;
      ret.push_back(param);
    }
    return ret;
  }

  /*!
   * \brief Allocate buffer(s) and update `func_info.expr2buffers` if the PrimFunc output(s) are
   * intermediate results.
   * \param expr The relax Expr, which can be binding vars or binding values.
   * \param func The old TIR PrimFunc
   * \param output_shapes The shape of output params.
   */
  void AllocateIntermediateBuffer(const CallNode* call, const tirx::PrimFunc& func,
                                  const ffi::Array<ffi::Array<PrimExpr>>& output_shapes) {
    bool is_inplace = call->op.same_as(Op::Get("relax.call_tir_inplace"));

    size_t n = func->params.size();
    int num_inputs = call->args[1].as_or_throw<Tuple>()->fields.size();
    size_t output_size = output_shapes.size();
    TVM_FFI_ICHECK_GE(n, output_size);
    ffi::Array<tirx::Buffer> output_buffers;
    ffi::Array<int64_t> output_idxs;
    if (is_inplace) {
      const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>();
      TVM_FFI_ICHECK(attrs) << "Must have CallTIRInplaceAttrs for an in-place call";
      output_idxs = GetInplaceOutputIndices(attrs->inplace_indices, num_inputs);
    } else {
      for (size_t i = 0; i < output_size; i++) {
        output_idxs.push_back(num_inputs + i);
      }
    }

    ffi::Array<tirx::Var> output_params = GetPrimFuncOutputParams(func, output_idxs);
    for (size_t i = 0; i < output_size; ++i) {
      const tirx::Var& param = output_params[i];
      const tirx::Buffer& buffer = func->buffer_map.at(param);

      // if this is an inplace output, do not do an intermediate allocation
      if (output_idxs[i] < num_inputs) {
        auto it = func_info_.buffer_subst_map.find(buffer);
        TVM_FFI_ICHECK(it != func_info_.buffer_subst_map.end())
            << "Inplace output buffer " << buffer << " must be mapped to a defined input";
        output_buffers.push_back((*it).second);
        continue;
      }

      auto unify_name_hints = [this, &buffer]() {
        ffi::String base_name = buffer->name;
        ffi::String unique_name = base_name + "_intermediate";
        size_t unique_id = 0;
        std::unordered_set<std::string> names;

        for (auto& _buffer : func_info_.alloc_buffers) {
          names.insert(_buffer->name);
        }

        while (names.find(unique_name) != names.end()) {
          unique_name = unique_name + "_" + std::to_string(++unique_id);
        }
        return unique_name;
      };
      // Update buffer with new symbolic shape according to the ty
      auto n = ffi::make_object<tirx::BufferNode>(*buffer.get());
      n->shape = output_shapes[i];
      n->name = unify_name_hints();
      tirx::Buffer new_buffer(n);
      func_info_.alloc_buffers.push_back(new_buffer);
      output_buffers.push_back(new_buffer);

      // Match the shape of the output buffer with the shape
      func_info_.symbolic_var_matcher.Match(buffer->shape, n->shape);
      func_info_.buffer_subst_map.Set(buffer, new_buffer);
    }
    // Update expr2buffers
    func_info_.expr2buffers.Set(ffi::GetRef<Expr>(call), output_buffers);
  }

  /*!
   * \brief Collect TIR func params and buffers with specified relax type and shape
   * \param ty The type
   * \param name_hint The name hint for params and buffers
   * \param out The vector into which to collect the params/buffers
   */
  static void CollectPrimFuncParams(const Var& relax_param,
                                    std::vector<ffi::Variant<tirx::PrimVar, tirx::Buffer>>* out,
                                    const ffi::Optional<tirx::Buffer>& tir_buffer_param) {
    auto ty = GetType(relax_param);

    TVM_FFI_CHECK(!ty.as<TupleTypeNode>(), InternalError)
        << "All tuple parameters should be expanded before this point in FuseTIR.  "
        << "However, parameter " << relax_param << " has type " << ty;

    auto name_hint = relax_param->name;

    if (const auto* tensor = ty.as<TensorTypeNode>()) {
      // Case 1. The relax param is a Tensor, we directly create a tirx var and buffer
      const auto* shape_expr = tensor->shape.as<ShapeExprNode>();
      TVM_FFI_ICHECK(shape_expr) << "FuseTIR expects all Tensor parameters have a known shape.";
      PrimType dtype = tensor->dtype.value();
      tirx::Buffer buffer;
      if (tir_buffer_param.has_value()) {
        buffer = tirx::decl_buffer(shape_expr->values, dtype, name_hint,
                                   tir_buffer_param.value().scope());
      } else {
        buffer = tirx::decl_buffer(shape_expr->values, dtype, name_hint);
      }
      out->push_back(std::move(buffer));

    } else if (ty.as<PrimTypeNode>()) {
      // Case 2. The relax param is a scalar, so its canonical Var is a TIR parameter.
      out->push_back(relax_param.as_or_throw<tirx::PrimVar>());

    } else if (const auto* shape_expr = ty.as<ShapeTypeNode>()) {
      // Case 3. The relax param is a tuple of scalars, each represented as a tirx var
      for (const auto& var : shape_expr->values.value()) {
        auto prim_var = var.as<tirx::PrimVar>();
        TVM_FFI_ICHECK(prim_var.has_value());
        out->push_back(prim_var.value());
      }
    } else {
      TVM_FFI_THROW(TypeError) << "The param type of PrimFunc is expected to be "
                               << "Tensor, PrimExpr, or ShapeExpr, "
                               << "but got " << ty->GetTypeKey();
    }
  }

  /*!
   * \brief Construct fused TIR func with collected FuseFuncInfo
   * \return The fused TIR
   */
  tirx::PrimFunc ConstructFunc() {
    ffi::Map<ffi::String, Any> attr_map;
    attr_map.Set(tirx::attr::kNoAlias, true);
    attr_map.Set(tvm::attr::kSTir, true);
    tirx::FuseTIRBufferSubstitutor subst(func_info_.buffer_subst_map,
                                         func_info_.symbolic_var_remap);
    TVM_FFI_ICHECK(func_info_.global_name != "fused");
    // Remove output buffers from func_info_.alloc_buffers
    ffi::Array<tirx::Buffer> alloc_buffers;
    for (const tirx::Buffer& buf : func_info_.alloc_buffers) {
      if (func_info_.output_buffers.count(buf.get()) == 0) {
        alloc_buffers.push_back(subst.SubstituteAllocatedBuffer(buf));
      }
    }
    tirx::Stmt body = tirx::SBlockNameDeduplicator()(tirx::SeqStmt::Flatten(func_info_.bodies));

    body = subst.Substitute(body);
    body = tirx::SBlock({}, {}, {}, "root", std::move(body), std::nullopt, alloc_buffers);
    body = tirx::SBlockRealize({}, IntImm::Bool(true), body.as_or_throw<tirx::SBlock>());
    tirx::PrimFunc func(func_info_.params, body, VoidType(), func_info_.buffer_map,
                        DictAttrs(attr_map));
    // Renew function defs to prevent using the same symbolic vars in different functions
    return s_tir::RenewDefs(func);
  }

  /*! \brief Get DynTensor numbers from recursive Tuples. */
  static size_t GetTotalTensorSize(const Type& ty) {
    if (ty.as<TensorTypeNode>()) {
      return 1;
    } else if (const auto* tuple_ty = ty.as<TupleTypeNode>()) {
      size_t num = 0;
      for (const Type& ty : tuple_ty->fields) {
        num += GetTotalTensorSize(ty);
      }
      return num;
    } else {
      TVM_FFI_THROW(InternalError) << "TensorType and TupleType are expect, but got: " << ty;
      return 0;
    }
  }

  /********** Function Info **********/

  /*! \brief auxiliary information for FuseTIR */
  struct FuseFuncInfo {
    /*! \brief The arguments for calling prim_func */
    ffi::Array<Expr> arguments;
    /*!
     * \brief The map from each dataflow var (intermediate var) to the corresponding buffers
     * allocated in the fused func
     */
    ffi::Map<Expr, ffi::Array<tirx::Buffer>> expr2buffers;
    /*! \brief The buffers to allocate in the fused func*/
    ffi::Array<tirx::Buffer> alloc_buffers;
    /*! \brief The bodies of the original funcs, which is also the body of the fused func. */
    ffi::Array<tirx::Stmt> bodies;
    /*! \brief The params of the fused function*/
    ffi::Array<tirx::Var> params;
    /*!
     * \brief The map from buffer in original functions to corresponding buffer in the fused
     * function
     */
    ffi::Map<tirx::Buffer, tirx::Buffer> buffer_subst_map;
    /*! \brief The `buffer_map` in the fused function*/
    ffi::Map<tirx::Var, tirx::Buffer> buffer_map;
    /*! \brief The output buffers in the function buffer_map*/
    std::unordered_set<const tirx::BufferNode*> output_buffers;
    /*! \brief The name of the fused function */
    std::string global_name = "fused";

    /*! \brief The map from symbolic var to its value in the fused function
     *
     * This is used in the default initialization of
     * `symbolic_var_matcher`, and must be before it in the struct
     * order.
     */
    ffi::Map<tirx::Var, PrimExpr> symbolic_var_remap;

    /*! \brief The map from symbolic var to its value in the fused function
     *
     * This is used in the default initialization of
     * `symbolic_var_matcher`, and must be before it in the struct
     * order.
     */
    arith::Analyzer analyzer;

    /*! \brief The map from symbolic var to its corresponding var in the fused function */
    tirx::SymbolicMatcher symbolic_var_matcher =
        tirx::SymbolicMatcher(analyzer.get(), &symbolic_var_remap);
  };

  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The name hint for the input func. */
  ffi::String func_name_;
  /*! \brief The helper info to fuse TIR prim_func */
  FuseFuncInfo func_info_;
  /*! \brief The tirx function after fusion*/
  tirx::PrimFunc fused_tir_;
  /*! \brief Indices of inputs that are used for in-place computation */
  std::unordered_set<size_t> inplace_indices_;
};

std::vector<size_t> GetTupleAccessedIndices(const FunctionNode* func, const Var& tuple_var) {
  // Need to be ordered
  std::vector<size_t> indices;
  PostOrderVisit(func->body, [&indices, tuple_var](Expr e) {
    if (auto tup_get = e.as<TupleGetItemNode>(); tup_get && tup_get->tuple.same_as(tuple_var)) {
      if (std::find(indices.begin(), indices.end(), tup_get->index) == indices.end()) {
        indices.push_back(tup_get->index);
      }
    }
  });
  return indices;
}

/*!
 * \brief The helper class to fuse TIR functions and build a new module which calls the fused TIR.
 */
class TIRFuseMutator : public ExprMutator {
 public:
  static IRModule Transform(IRModule mod) {
    // Collect all primitive relax functions
    ffi::Map<GlobalVar, Function> primitive_relax;
    for (const auto& gvar : mod->GetGlobalVars()) {
      const auto& base_func = mod->Lookup(gvar);
      // Only fuse primitive relax functions
      if (base_func->HasNonzeroAttr(attr::kPrimitive)) {
        if (auto func = base_func.as<relax::Function>()) {
          primitive_relax.Set(gvar, func.value());
        }
      }
    }

    if (primitive_relax.empty()) {
      return mod;
    }

    mod.CopyOnWrite();

    IRModule updates;
    std::unordered_map<GlobalVar, Replacement> replacements;

    // Since TIRFuseMutator will delete bunch of PrimFunc, we create an empty block builder.

    // Step 1. Fuse all primitive relax functions, store the result in `fused_tir_funcs_`
    for (const auto& [old_gvar, func] : primitive_relax) {
      const auto& [prim_func, indices] = FusedTIRConstructor::GetFusedTIR(mod, old_gvar);

      GlobalVar new_gvar(old_gvar->name_hint);
      UpdateType(new_gvar, GetType(prim_func));

      mod->Remove(old_gvar);
      updates->Add(new_gvar, prim_func);
      replacements[old_gvar] = Replacement{new_gvar, func, indices};
    }

    TIRFuseMutator mutator(replacements);

    // Step 2. Update all non-primitive relax functions and add it, with the dependent function,
    // into the new IRModule

    for (const auto& [gv, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionNode>()) {
        TVM_FFI_ICHECK(!func->HasNonzeroAttr(attr::kPrimitive))
            << "Module should not contain any primitive relax functions at this point";
        relax::Function update_func = mutator.VisitExpr(func).as_or_throw<Function>();
        if (!update_func.same_as(func)) {
          updates->Add(gv, update_func);
        }
      }
    }

    // Step 4. Copy over updated functions and return.
    mod->Update(updates);
    return mod;
  }

 private:
  struct Replacement {
    GlobalVar fused_tir_gvar;
    Function original_function;
    ffi::Array<int64_t> inplace_indices;
  };

  explicit TIRFuseMutator(std::unordered_map<GlobalVar, Replacement> replacements)
      : replacements_(replacements) {}

  using ExprMutator::VisitExpr_;

  // Get shape from call tirx
  static Expr GetCallTIRShape(Type ty) {
    if (auto* tuple = ty.as<TupleTypeNode>()) {
      ffi::Array<Expr> fields = tuple->fields.Map([&](Type x) { return GetCallTIRShape(x); });
      return Tuple(fields);
    } else {
      auto* tensor = ty.as<TensorTypeNode>();
      TVM_FFI_ICHECK(tensor) << "FuseTIR can only take tensor or tuple type";
      auto* shape_expr = tensor->shape.as<ShapeExprNode>();
      TVM_FFI_ICHECK(shape_expr) << "FuseTIR requires all intermediate values have shape";
      return ffi::GetRef<ShapeExpr>(shape_expr);
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op_ = Op::Get("relax.call_tir_inplace");

    Call call = builder_->Normalize(ExprMutator::VisitExpr_(op)).as_or_throw<Call>();

    auto opt_gvar = call->op.as<GlobalVar>();
    if (!opt_gvar) {
      // Case 1. The Call isn't a relax-to-relax function call, no need to update.
      return call;
    }
    GlobalVar old_gvar = opt_gvar.value();

    auto it = replacements_.find(old_gvar);
    if (it == replacements_.end()) {
      // Case 2. The callee function is not a primitive relax
      // function, no need to update.
      return call;
    }
    const Replacement& replacement = it->second;
    const GlobalVar& fused_tir_gv = replacement.fused_tir_gvar;
    const Function& relax_func = replacement.original_function;

    // Case 3. It calls a primitive relax function, update the call
    // into a call_tir or call_tir_inplace.

    // Step a. Collect all relax/symbolic arguments.  Tuple arguments
    // are not supported by PrimFunc, so this step verifies that
    // ExpandTupleArguments has already removed them.
    ffi::Array<Expr> arg_list;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto arg = call->args[i];
      auto ty = GetType(arg);

      TVM_FFI_CHECK(
          !relax_func->params[i]->ty->IsInstance<TupleTypeNode>() && !ty.as<TupleTypeNode>(),
          InternalError)
          << "All tuple parameters should be expanded before this point in FuseTIR.  "
          << "However, argument " << arg << " with type " << arg->ty << " is passed as argument "
          << i << " to Primitive Relax function " << old_gvar << ", which expects parameter "
          << relax_func->params[i] << " to have type " << relax_func->params[i]->ty;

      if (const auto* shape = ty.as<ShapeTypeNode>()) {
        TVM_FFI_ICHECK(shape->values.has_value())
            << "FuseTIR requires all shape input has ty value.";
        for (const PrimExpr& prim_value : shape->values.value()) {
          TVM_FFI_ICHECK(prim_value.as<tirx::PrimVar>())
              << "All shape inputs are expected to be single tirx var.";
          arg_list.push_back(prim_value);
        }
      } else if (ty.as<PrimTypeNode>()) {
        if (auto literal = arg.as<PrimExpr>()) {
          arg_list.push_back(literal.value());
        } else {
          TVM_FFI_THROW(TypeError) << "FuseTIR expects scalar arguments to be PrimExpr, "
                                   << "but received " << arg;
        }

      } else {
        arg_list.push_back(arg);
      }
    }

    // Step b. Create call_tir or call_tir_inplace
    ffi::Array<Expr> call_args = {fused_tir_gv, Tuple(arg_list)};
    Op call_op = call_tir_op_;
    Attrs call_attrs = call->attrs;
    if (replacement.inplace_indices.size()) {
      call_op = call_tir_inplace_op_;
      auto inplace_attrs = ffi::make_object<CallTIRInplaceAttrs>();
      inplace_attrs->inplace_indices = replacement.inplace_indices;
      call_attrs = Attrs(inplace_attrs);
    }
    return Call(Type::Missing(), call_op, call_args, call_attrs, {GetType(call)});
  }

 private:
  /*! \brief The map from global var to how it should be replaced
   *
   * Has one entry for each primitive relax function in the IRModule.
   */
  std::unordered_map<GlobalVar, Replacement> replacements_;
};

IRModule FuseTIR(IRModule mod) {
  mod = TIRFuseMutator::Transform(mod);
  return mod;
}

namespace transform {

Pass FuseTIR() {
  auto pass_func =  //
      [=](IRModule m, PassContext pc) { return relax::FuseTIR(m); };
  auto inner_pass = CreateModulePass(/*pass_function=*/pass_func,   //
                                     /*opt_level=*/0,               //
                                     /*pass_name=*/"FuseTIRInner",  //
                                     /*required=*/{});
  return tvm::transform::Sequential(
      {
          ExpandTupleArguments(),
          RemoveUnusedParameters(),
          inner_pass,
          DeadCodeElimination(),
      },
      "FuseTIR");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.FuseTIR", FuseTIR);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
