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
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>

#include "../../relay/analysis/graph_partitioner.h"
#include "../../support/arena.h"
#include "../../tir/ir/functor_common.h"

namespace tvm {
namespace tir {

// TODO(Siyuan): move it to somewhere under tir folder
/*!
 * \brief Match symbolic vars according to the given PrimExpr, and update the var_remap.
 * Will throw errors if there is a mismatch.
 */
class SymbolicMatcher : ExprFunctor<void(const PrimExpr& n, const PrimExpr& other)> {
 public:
  explicit SymbolicMatcher(arith::Analyzer* analyzer, Map<tir::Var, PrimExpr>* var_remap)
      : analyzer_(analyzer), var_remap_(var_remap) {}

  void Match(const Array<PrimExpr>& params, const Array<PrimExpr>& args) {
    CHECK_EQ(params.size(), args.size());
    for (size_t i = 0; i < params.size(); ++i) {
      Match(params[i], args[i]);
    }
  }
  void Match(const PrimExpr& param, const PrimExpr& arg) {
    VisitExpr(param, arg);
    must_prove_ = analyzer_->Simplify(Substitute(must_prove_, *var_remap_));
    CHECK(!is_zero(must_prove_));
  }

 private:
  void VisitExpr(const PrimExpr& node, const PrimExpr& other) {
    if (node.same_as(other)) {
      return;
    } else if (node.dtype().code() != other.dtype().code()) {
      LOG(FATAL) << "Parameter expression " << node << " with dtype " << node.dtype()
                 << " cannot match to argument " << other << " with dtype " << other.dtype();
    } else {
      ExprFunctor::VisitExpr(node, other);
    }
  }

#define TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(OpName)                  \
  void VisitExpr_(const OpName* op, const PrimExpr& other) {        \
    const auto* rhs = other.as<OpName>();                           \
    if (rhs) {                                                      \
      VisitExpr(op->a, rhs->a);                                     \
      VisitExpr(op->b, rhs->b);                                     \
    } else {                                                        \
      must_prove_ = must_prove_ && (GetRef<PrimExpr>(op) == other); \
    }                                                               \
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
      LOG(FATAL) << "Parameter expression " << GetRef<PrimExpr>(op)
                 << " expected an integer argument with value " << op->value << ", "
                 << "but was provided with the argument " << other;
    }
  }

  void VisitExpr_(const FloatImmNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<FloatImmNode>();
    if (!rhs || (op->value != rhs->value)) {
      LOG(FATAL) << "Parameter expression " << GetRef<PrimExpr>(op)
                 << " expected an float argument with value " << op->value << ", "
                 << "but was provided with the argument " << other;
    }
  }

  void VisitExpr_(const CastNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<CastNode>();
    if (!rhs) {
      LOG(FATAL) << "Parameter expression " << GetRef<PrimExpr>(op) << " expected an cast to "
                 << op->dtype << " as the argument, "
                 << "but was provided with the argument " << other;
    }
    VisitExpr(op->value, rhs->value);
  }

  void VisitExpr_(const VarNode* op, const PrimExpr& rhs) {
    auto lhs = GetRef<Var>(op);

    if (lhs.same_as(rhs)) {
      // Reference identity, no further checks needed.
    } else if (op->dtype.code() != rhs->dtype.code()) {
      LOG(FATAL) << "Parameter expression " << GetRef<PrimExpr>(op) << " with dtype " << op->dtype
                 << " cannot match to argument " << rhs << " with dtype " << rhs.dtype();
    } else if (auto it = var_remap_->find(lhs); it != var_remap_->end()) {
      VisitExpr((*it).second, rhs);
    } else {
      var_remap_->Set(lhs, rhs);
    }
  }

  arith::Analyzer* analyzer_;
  Map<tir::Var, PrimExpr>* var_remap_;
  PrimExpr must_prove_ = Bool(true);
};

/*!
 * \brief Substitute a given source buffer with a given target buffer in statements or expressions.
 */
class FuseTIRBufferSubstitutor : private StmtExprMutator {
 public:
  explicit FuseTIRBufferSubstitutor(const Map<Buffer, Buffer>& buffer_map,
                                    const Map<Var, PrimExpr>& var_map) {
    buffer_remap_ = buffer_map;
    var_remap_ = var_map;
    for (const auto& [src, tgt] : buffer_map) {
      var_remap_.Set(src->data, tgt->data);
    }
  }

  Stmt Substitute(Stmt stmt) { return this->VisitStmt(std::move(stmt)); }

  Buffer SubstituteAllocatedBuffer(Buffer buffer) {
    ICHECK(buffer_remap_.find(buffer) == buffer_remap_.end());
    Array<PrimExpr> shape =
        MutateArray(buffer->shape, [this](const PrimExpr& expr) { return this->VisitExpr(expr); });
    Array<PrimExpr> strides = MutateArray(
        buffer->strides, [this](const PrimExpr& expr) { return this->VisitExpr(expr); });
    PrimExpr elem_offset = this->VisitExpr(buffer->elem_offset);
    if (shape.same_as(buffer->shape) && strides.same_as(buffer->strides) &&
        elem_offset.same_as(buffer->elem_offset)) {
      return buffer;
    } else {
      auto n = make_object<BufferNode>(*buffer.get());
      n->shape = std::move(shape);
      n->strides = std::move(strides);
      n->elem_offset = std::move(elem_offset);
      Buffer new_buffer(n);
      this->buffer_remap_.Set(buffer, new_buffer);
      return new_buffer;
    }
  }

 private:
  PrimExpr VisitExpr_(const VarNode* _op) final {
    if (auto it = var_remap_.find(GetRef<Var>(_op)); it != var_remap_.end()) {
      return (*it).second;
    } else {
      return GetRef<PrimExpr>(_op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    const Buffer& buffer = SubstituteBuffer(load->buffer);
    if (buffer.same_as(load->buffer)) {
      return std::move(load);
    } else {
      auto n = make_object<BufferLoadNode>(*load.get());
      n->buffer = buffer;
      return BufferLoad(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    const Buffer& buffer = SubstituteBuffer(store->buffer);
    if (buffer.same_as(store->buffer)) {
      return std::move(store);
    } else {
      auto n = make_object<BufferStoreNode>(*store.get());
      n->buffer = buffer;
      return BufferStore(n);
    }
  }

  Stmt VisitStmt_(const BlockNode* _op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(_op));

    // Define the mutation functions.

    auto f_mutate_match_buffers = [this](const MatchBufferRegion& match_buffer) {
      const Buffer& src_buffer = SubstituteBuffer(match_buffer->source->buffer);
      const Buffer& tgt_buffer = SubstituteAllocatedBuffer(match_buffer->buffer);
      Region region = MutateRegion(match_buffer->source->region);
      if (src_buffer.same_as(match_buffer->source->buffer) &&
          tgt_buffer.same_as(match_buffer->buffer) &&
          region.same_as(match_buffer->source->region)) {
        return match_buffer;
      } else {
        auto n = make_object<MatchBufferRegionNode>(*match_buffer.get());
        n->buffer = tgt_buffer;
        n->source = BufferRegion(src_buffer, region);
        return MatchBufferRegion(n);
      }
    };

    auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
      const Buffer& buffer = SubstituteBuffer(buffer_region->buffer);
      const Region& region = MutateRegion(buffer_region->region);
      if (buffer.same_as(buffer_region->buffer) && region.same_as(buffer_region->region)) {
        return buffer_region;
      } else {
        return BufferRegion(buffer, region);
      }
    };

    // Step 1. Mutate `match_buffers`.
    Array<MatchBufferRegion> match_buffers =
        MutateArray(block->match_buffers, f_mutate_match_buffers);
    // Step 2. Mutate the read/write region.
    Array<BufferRegion> reads = MutateArray(block->reads, f_mutate_read_write_region);
    Array<BufferRegion> writes = MutateArray(block->writes, f_mutate_read_write_region);
    // Step 3. Mutate the Allocate Buffers.
    Array<Buffer> alloc_buffers = MutateArray(block->alloc_buffers, [this](const Buffer& buffer) {
      return SubstituteAllocatedBuffer(buffer);
    });

    reads = UnionAccessRegion(reads);
    writes = UnionAccessRegion(writes);

    if (reads.same_as(block->reads) &&    //
        writes.same_as(block->writes) &&  //
        match_buffers.same_as(block->match_buffers) &&
        alloc_buffers.same_as(block->alloc_buffers)) {
      return std::move(block);
    } else {
      auto n = CopyOnWrite(block.get());
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->match_buffers = std::move(match_buffers);
      n->alloc_buffers = std::move(alloc_buffers);
      return Block(n);
    }
  }

 private:
  /*! \brief Mapping from src buffer to tgt buffer. */
  Map<tir::Buffer, tir::Buffer> buffer_remap_;
  /*! \brief Mapping from src tir var to tgt var. */
  Map<tir::Var, PrimExpr> var_remap_;

  Array<tir::BufferRegion> UnionAccessRegion(const Array<BufferRegion>& regions) const {
    // For now we only allow Buffer access the same elements.
    // e.g. `[A[vi, vj], A[vi, vj]]` is a legal pattern but need to union to `A[vi, vj]`
    // However, `A[vi, vj], A[vi, vj + 1]` is not allow for now.
    // Note: the order of return region should remain the same as the first occurrence of the region
    Array<BufferRegion> ret;
    std::unordered_map<const BufferNode*, Region> buffer_region_set;

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

  inline Region MutateRegion(const Region& region) {
    return MutateArray(region, [this](const Range& range) {
      const PrimExpr& min = this->VisitExpr(range->min);
      const PrimExpr& extent = this->VisitExpr(range->extent);
      if (min.same_as(range->min) && extent.same_as(range->extent)) {
        return range;
      } else {
        return Range::FromMinExtent(min, extent);
      }
    });
  }
};

/*! \brief A mutator which detect block name duplication and deduplicate the names. */
class BlockNameDeduplicator : public tir::StmtMutator {
 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(tir::StmtMutator::VisitStmt_(op));

    String name = GetUniqueName(block->name_hint);

    if (name == block->name_hint) {
      return std::move(block);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(block.get());
      n->name_hint = std::move(name);
      return Stmt(n);
    }
  }

  String GetUniqueName(const String& prefix) {
    String unique_prefix = prefix;
    auto it = name_count_.find(prefix);
    while (name_count_.count(unique_prefix)) {
      unique_prefix = prefix + "_" + std::to_string(++it->second);
    }
    name_count_[unique_prefix] = 0;
    return unique_prefix;
  }

  // TODO(relax-team): It should detects the number suffix and do renaming properly
  // e.g. GetUniqueName("name1") should return "name2" instead of "name10".
  /*! \brief The count map to make block name unique. */
  std::unordered_map<String, int> name_count_;
};

}  // namespace tir

namespace relax {

class FusedTIRConstructor : public ExprVisitor {
 public:
  /*!
   * \brief Construct a fused TIR PrimFunc from a relax sub-function
   * \param mod The IRModule
   * \param gv The global var of relax subfunction to be fused into one PrimFunc
   * \return The fused TIR PrimFunc
   */
  static tir::PrimFunc GetFusedTIR(const IRModule& mod, const GlobalVar& gv) {
    FusedTIRConstructor visitor(mod, gv->name_hint);
    BaseFunc f = mod->Lookup(gv);
    CHECK(f->IsInstance<relax::FunctionNode>())
        << "Expected relax functions, but got: " << f->GetTypeKey();
    CHECK(f->HasNonzeroAttr(relax::attr::kPrimitive))
        << "Expected a function with attr `kPrimitive`";
    visitor(Downcast<relax::Function>(f));
    return visitor.fused_tir_;
  }

 private:
  explicit FusedTIRConstructor(const IRModule& mod, const String& func_name)
      : mod_(mod), func_name_(func_name) {}

  void VisitExpr_(const FunctionNode* func) final {
    std::vector<Variant<tir::Var, tir::Buffer>> prim_func_params;
    for (const Var& relax_param : func->params) {
      size_t size_before = prim_func_params.size();
      CollectPrimFuncParams(relax_param, &prim_func_params);

      auto param_buffers = [&]() -> Array<tir::Buffer> {
        Array<tir::Buffer> out;
        for (size_t i = size_before; i < prim_func_params.size(); i++) {
          if (auto buf = prim_func_params[i].as<tir::Buffer>()) {
            out.push_back(buf.value());
          }
        }
        return out;
      }();

      func_info_.expr2buffers.Set(relax_param, param_buffers);
    }

    // Move all scalar params after buffer params.  To ensure that the
    // order is deterministic and predictable for testing purposes,
    // std::stable_sort is used instead of std::sort.
    std::stable_sort(prim_func_params.begin(), prim_func_params.end(),
                     [](const auto& a, const auto& b) {
                       bool a_is_var = a.template as<tir::VarNode>();
                       bool b_is_var = b.template as<tir::VarNode>();
                       return a_is_var < b_is_var;
                     });

    for (const auto& param : prim_func_params) {
      if (auto opt = param.as<tir::Buffer>()) {
        auto buffer = opt.value();
        // Differentiate buffer name and param name by adding prefix
        // `p_` to the buffer name.  Every symbol should be unique in
        // TVMScript, and while they can be de-deplicated when
        // printed, it's more readable when done explicitly.  Since
        // Buffer is used more than param it gets the name with better
        // readability.
        tir::Var param = tir::Var("p_" + buffer->name, PrimType(DataType::Handle()));
        func_info_.params.push_back(param);
        func_info_.buffer_map.Set(param, buffer);
      }
    }

    // Step 2. Visit Function body and create intermediate buffers
    ExprVisitor::VisitExpr_(func);

    // Step 3. Create and remap buffers for function output
    ICHECK(func->body->IsInstance<SeqExprNode>())
        << "Function body is expected to be a SeqExpr, but got: " << func->body->GetTypeKey();
    Expr body = Downcast<SeqExpr>(func->body)->body;
    auto it = func_info_.expr2buffers.find(body);
    ICHECK(it != func_info_.expr2buffers.end())
        << "Fail to detect output buffers for function body";
    const Array<tir::Buffer>& buffers = (*it).second;
    for (size_t i = 0; i < buffers.size(); ++i) {
      tir::Var param = tir::Var("p_output" + std::to_string(i), PrimType(DataType::Handle()));
      func_info_.buffer_map.Set(param, buffers[i]);
      func_info_.params.push_back(param);
      func_info_.output_buffers.insert(buffers[i].get());
    }

    // Step 4. Append symbolic vars
    for (const auto& param : prim_func_params) {
      if (auto var = param.as<tir::Var>()) {
        func_info_.params.push_back(var.value());
      }
    }

    // Step 5. Create PrimFunc
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
      LOG(FATAL) << "Unsupported binding value: " << binding->value;
    }
  }

  void VisitBinding_(const MatchCastNode* match_cast) final {
    LOG(FATAL) << "MatchCast is unsupported in primitive functions";
  }

  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    ICHECK(call->op == call_tir_op_)
        << "Only call_tir is supported in primitive function, but got: " << GetRef<Expr>(call);

    // Step 1. Get Global var and PrimFunc
    GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    tir::PrimFunc prim_func_ = Downcast<tir::PrimFunc>(mod_->Lookup(gv));

    // Step 2. Renew all vars/buffer definitions and blocks to avoid duplication
    tir::PrimFunc prim_func = tir::RenewDefs(prim_func_);

    // Step 3. Check functions are all schedulable funcs. i.e. the body of func is root block
    // TODO(Siyuan): support un-schedulable functions.
    ICHECK(prim_func->body->IsInstance<tir::BlockRealizeNode>())
        << "Only schedulable functions (whose body is the root block) can be fused";
    const tir::BlockRealize& root_realize = Downcast<tir::BlockRealize>(prim_func->body);
    const tir::Block& root_block = root_realize->block;

    // Step 4. Add all the original alloc_buffers and body to the fused function.
    func_info_.alloc_buffers.insert(func_info_.alloc_buffers.end(),
                                    root_block->alloc_buffers.begin(),
                                    root_block->alloc_buffers.end());
    func_info_.bodies.push_back(root_block->body);

    // Step 5. Map input arguments to buffer
    MapInputBuffer(prim_func, call->args[1]);
    const Array<Array<PrimExpr>>& output_buffer_shapes = GetCallTIROutputShapes(call);

    AllocateIntermediateBuffer(GetRef<Expr>(call), prim_func, output_buffer_shapes);

    // Step 6. Update tir_vars
    if (call->args.size() > 2) {
      ICHECK(call->args.size() == 3);
      const Expr& tir_vars = call->args[2];
      if (const auto* shape_expr = tir_vars.as<ShapeExprNode>()) {
        const auto& args = shape_expr->values;
        size_t num_params = prim_func->params.size();
        ICHECK_GE(num_params, args.size());
        for (size_t i = 0; i < args.size(); ++i) {
          const tir::Var& param = prim_func->params[num_params - args.size() + i];
          func_info_.symbolic_var_matcher.Match(param, args[i]);
        }
      } else {
        LOG(FATAL) << "TIR vars should be a shape expr, but got: " << tir_vars->GetTypeKey();
      }
    }
    // Update fused func name
    func_info_.global_name += "_" + gv->name_hint;
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item) final {
    ExprVisitor::VisitExpr_(tuple_get_item);
    auto it = func_info_.expr2buffers.find(tuple_get_item->tuple);
    if (it != func_info_.expr2buffers.end()) {
      int begin_buf_idx = 0;
      int end_buf_idx = 0;
      const TupleType& tuple_type = Downcast<TupleType>(tuple_get_item->tuple->checked_type());
      for (int i = 0; i < tuple_get_item->index; ++i) {
        begin_buf_idx += GetTotalTensorSize(tuple_type->fields[i]);
      }
      end_buf_idx = begin_buf_idx + GetTotalTensorSize(tuple_type->fields[tuple_get_item->index]);
      func_info_.expr2buffers.Set(
          GetRef<Expr>(tuple_get_item),
          {(*it).second.begin() + begin_buf_idx, (*it).second.begin() + end_buf_idx});
    }
  }

  void VisitExpr_(const TupleNode* tuple) final {
    ExprVisitor::VisitExpr_(tuple);
    Array<tir::Buffer> buffers;
    for (const Expr& expr : tuple->fields) {
      auto it = func_info_.expr2buffers.find(expr);
      if (it != func_info_.expr2buffers.end()) {
        buffers.insert(buffers.end(), (*it).second.begin(), (*it).second.end());
      }
    }
    if (!buffers.empty()) {
      func_info_.expr2buffers.Set(GetRef<Expr>(tuple), buffers);
    }
  }

  void VisitExpr_(const ConstantNode* op) final {
    LOG(FATAL) << "Relax.Constant is not supported in primitive functions.";
  }

  /*!
   * \brief Get the number of outputs for a call_tir node.
   * \return The number of outputs.
   */
  static Array<Array<PrimExpr>> GetCallTIROutputShapes(const CallNode* call) {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    ICHECK(call->op.same_as(call_tir_op_));
    ICHECK_EQ(call->sinfo_args.size(), 1);
    auto get_tensor_shape = [](const TensorStructInfoNode* sinfo) {
      const auto* shape_expr = sinfo->shape.as<ShapeExprNode>();
      CHECK(shape_expr) << "FuseTIR expects all parameters are Tensors with symbolic shape.";
      return shape_expr->values;
    };
    if (const auto* tuple_sinfo = call->sinfo_args[0].as<TupleStructInfoNode>()) {
      Array<Array<PrimExpr>> shapes;
      for (const StructInfo& field : tuple_sinfo->fields) {
        const auto* tensor_sinfo = field.as<TensorStructInfoNode>();
        CHECK(tensor_sinfo) << "CallTIR sinfo_args are expected to be TensorStructInfo or Tuple of "
                               "TensorStructInfo, but got "
                            << call->sinfo_args[0];
        shapes.push_back(get_tensor_shape(tensor_sinfo));
      }
      return shapes;
    } else if (const auto* tensor_sinfo = call->sinfo_args[0].as<TensorStructInfoNode>()) {
      return {get_tensor_shape(tensor_sinfo)};
    } else {
      CHECK(tensor_sinfo) << "CallTIR sinfo_args are expected to be TensorStructInfo or Tuple of "
                             "TensorStructInfo, but got "
                          << call->sinfo_args[0];
      throw;
    }
  }

  /*! \brief Map old TIR func param buffer to new buffer, and then update `buffer_subst_map` */
  void MapArgsToBuffer(const Array<Expr> args, const Array<tir::Buffer>& buffers) {
    size_t buffer_idx = 0;
    for (const Expr& arg : args) {
      if (const auto* v = arg.as<VarNode>()) {
        auto it = func_info_.expr2buffers.find(GetRef<Var>(v));
        // Substitute the buffer with the already allocated one if it is an intermediate var
        if (it != func_info_.expr2buffers.end()) {
          for (const tir::Buffer& target_buffer : (*it).second) {
            ICHECK_LT(buffer_idx, buffers.size());
            const tir::Buffer& buffer = buffers[buffer_idx];
            func_info_.symbolic_var_matcher.Match(buffer->shape, target_buffer->shape);
            func_info_.buffer_subst_map.Set(buffer, target_buffer);
            buffer_idx++;
          }
        }
      }
    }
    // Make sure every buffers are mapped.
    ICHECK_EQ(buffer_idx, buffers.size());
  }

  /*!
   * \brief Update buffer mapping `func_info_.buffer_subst_map` for input args
   * \param func The old TIR PrimFunc
   * \param output_size The number of output params. All output params are at the end of param list.
   */
  void MapInputBuffer(const tir::PrimFunc& func, const relax::Expr& args) {
    Array<Expr> arg_list;
    Array<tir::Buffer> buffer_list;
    if (const auto* arg_tuple = args.as<TupleNode>()) {
      arg_list = arg_tuple->fields;
    } else {
      arg_list = {args};
    }

    ICHECK_GE(func->params.size(), arg_list.size());
    for (size_t i = 0; i < arg_list.size(); ++i) {
      const tir::Var& param = func->params[i];
      const tir::Buffer& buffer = func->buffer_map.at(param);
      buffer_list.push_back(buffer);
    }

    MapArgsToBuffer(arg_list, buffer_list);
  }

  static Array<tir::Var> GetPrimFuncOutputParams(const tir::PrimFunc& func, size_t output_size) {
    size_t n = func->params.size();
    int symbolic_var_index = -1;
    ICHECK_GE(n, output_size);
    for (size_t i = 0; i < n; ++i) {
      const tir::Var& param = func->params[i];
      if (param->dtype.is_int() || param->dtype.is_uint()) {
        if (symbolic_var_index == -1) symbolic_var_index = i;
      } else if (param->dtype.is_handle()) {
        CHECK(symbolic_var_index == -1) << "The scalar input should be at the ending of the "
                                           "parameter list.";
      } else {
        LOG(FATAL) << "The params of PrimFunc are expected to be Buffer handle or scalar, but got: "
                   << param->dtype;
      }
    }
    size_t end_index = symbolic_var_index == -1 ? n : symbolic_var_index;
    ICHECK_GE(end_index, output_size);
    size_t begin_index = end_index - output_size;
    Array<tir::Var> output_params{func->params.begin() + begin_index,
                                  func->params.begin() + end_index};
    return output_params;
  }

  /*!
   * \brief Allocate buffer(s) and update `func_info.expr2buffers` if the PrimFunc output(s) are
   * intermediate results.
   * \param expr The relax Expr, which can be binding vars or binding values.
   * \param func The old TIR PrimFunc
   * \param output_shapes The shape of output params.
   */
  void AllocateIntermediateBuffer(const Expr& expr, const tir::PrimFunc& func,
                                  const Array<Array<PrimExpr>>& output_shapes) {
    size_t n = func->params.size();
    size_t output_size = output_shapes.size();
    ICHECK_GE(n, output_size);
    // Allocate intermediate buffer
    Array<tir::Buffer> alloc_buffers;
    Array<tir::Var> output_params = GetPrimFuncOutputParams(func, output_size);
    for (size_t i = 0; i < output_size; ++i) {
      const tir::Var& param = output_params[i];
      const tir::Buffer& buffer = func->buffer_map.at(param);

      auto unify_name_hints = [this, &buffer]() {
        String base_name = buffer->name;
        String unique_name = base_name + "_intermediate";
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
      // Update buffer with new symbolic shape according to the sinfo
      auto n = make_object<tir::BufferNode>(*buffer.get());
      n->shape = output_shapes[i];
      n->name = unify_name_hints();
      tir::Buffer new_buffer(n);
      func_info_.alloc_buffers.push_back(new_buffer);
      alloc_buffers.push_back(new_buffer);

      // Match the shape of the output buffer with the shape
      func_info_.symbolic_var_matcher.Match(buffer->shape, n->shape);
      func_info_.buffer_subst_map.Set(buffer, new_buffer);
    }
    // Update expr2buffers
    func_info_.expr2buffers.Set(expr, alloc_buffers);
  }

  /*!
   * \brief Collect TIR func params and buffers with specified relax type and shape
   * \param struct_info The struct info
   * \param name_hint The name hint for params and buffers
   * \param out The vector into which to collect the params/buffers
   */
  static void CollectPrimFuncParams(const Var& relax_param,
                                    std::vector<Variant<tir::Var, tir::Buffer>>* out) {
    auto struct_info = GetStructInfo(relax_param);

    CHECK(!struct_info.as<TupleStructInfoNode>())
        << "InternalError: "
        << "All tuple parameters should be expanded before this point in FuseTIR.  "
        << "However, parameter " << relax_param << " has struct info " << struct_info;

    auto name_hint = relax_param->name_hint();

    if (const auto* tensor = struct_info.as<TensorStructInfoNode>()) {
      // Case 1. The relax param is a Tensor, we directly create a tir var and buffer
      const auto* shape_expr = tensor->shape.as<ShapeExprNode>();
      ICHECK(shape_expr) << "FuseTIR expects all Tensor parameters have a known shape.";
      DataType dtype = tensor->dtype;
      tir::Buffer buffer = tir::decl_buffer(shape_expr->values, dtype, name_hint);
      out->push_back(std::move(buffer));

    } else if (const auto* prim_value = struct_info.as<PrimStructInfoNode>()) {
      // Case 2. The relax param is a scalar, we directly create a tir var
      ICHECK(prim_value->value->IsInstance<tir::VarNode>());
      out->push_back(Downcast<tir::Var>(prim_value->value));

    } else if (const auto* shape_expr = struct_info.as<ShapeStructInfoNode>()) {
      // Case 3. The relax param is a tuple of scalars, each represented as a tir var
      for (const auto& var : shape_expr->values.value()) {
        ICHECK(var->IsInstance<tir::VarNode>());
        out->push_back(Downcast<tir::Var>(var));
      }
    } else {
      LOG(FATAL) << "TypeError: "
                 << "The param type of PrimFunc is expected to be "
                 << "Tensor, PrimValue, or ShapeExpr, "
                 << "but got " << struct_info->GetTypeKey();
    }
  }

  /*!
   * \brief Construct fused TIR func with collected FuseFuncInfo
   * \return The fused TIR
   */
  tir::PrimFunc ConstructFunc() {
    Map<String, ObjectRef> attr_map;
    attr_map.Set("tir.noalias", tir::const_true());
    tir::FuseTIRBufferSubstitutor subst(func_info_.buffer_subst_map, func_info_.symbolic_var_remap);
    ICHECK(func_info_.global_name != "fused");
    // Remove output buffers from func_info_.alloc_buffers
    Array<tir::Buffer> alloc_buffers;
    for (const tir::Buffer& buf : func_info_.alloc_buffers) {
      if (func_info_.output_buffers.count(buf.get()) == 0) {
        alloc_buffers.push_back(subst.SubstituteAllocatedBuffer(buf));
      }
    }
    tir::Stmt body = tir::BlockNameDeduplicator()(tir::SeqStmt::Flatten(func_info_.bodies));

    body = subst.Substitute(body);
    body = tir::Block({}, {}, {}, "root", std::move(body), NullOpt, alloc_buffers);
    body = tir::BlockRealize({}, Bool(true), Downcast<tir::Block>(body));
    tir::PrimFunc func(func_info_.params, body, VoidType(), func_info_.buffer_map,
                       DictAttrs(attr_map));
    // Renew function defs to prevent using the same symbolic vars in different functions
    return tir::RenewDefs(func);
  }

  /*! \brief Get DynTensor numbers from recursive Tuples. */
  static size_t GetTotalTensorSize(const Type& type) {
    if (type.as<DynTensorTypeNode>()) {
      return 1;
    } else if (const auto* tuple_type = type.as<TupleTypeNode>()) {
      size_t num = 0;
      for (const Type& type : tuple_type->fields) {
        num += GetTotalTensorSize(type);
      }
      return num;
    } else {
      LOG(FATAL) << "DynTensorType and TupleType are expect, but got: " << type;
      return 0;
    }
  }

  /********** Function Info **********/

  /*! \brief auxiliary information for FuseTIR */
  struct FuseFuncInfo {
    /*! \brief The arguments for calling prim_func */
    Array<Expr> arguments;
    /*!
     * \brief The map from each dataflow var (intermediate var) to the corresponding buffers
     * allocated in the fused func
     */
    Map<Expr, Array<tir::Buffer>> expr2buffers;
    /*! \brief The buffers to allocate in the fused func*/
    Array<tir::Buffer> alloc_buffers;
    /*! \brief The bodies of the original funcs, which is also the body of the fused func. */
    Array<tir::Stmt> bodies;
    /*! \brief The params of the fused function*/
    Array<tir::Var> params;
    /*!
     * \brief The map from buffer in original functions to corresponding buffer in the fused
     * function
     */
    Map<tir::Buffer, tir::Buffer> buffer_subst_map;
    /*! \brief The `buffer_map` in the fused function*/
    Map<tir::Var, tir::Buffer> buffer_map;
    /*! \brief The output buffers in the function buffer_map*/
    std::unordered_set<const tir::BufferNode*> output_buffers;
    /*! \brief The name of the fused function */
    std::string global_name = "fused";

    /*! \brief The map from symbolic var to its value in the fused function
     *
     * This is used in the default initialization of
     * `symbolic_var_matcher`, and must be before it in the struct
     * order.
     */
    Map<tir::Var, PrimExpr> symbolic_var_remap;

    /*! \brief The map from symbolic var to its value in the fused function
     *
     * This is used in the default initialization of
     * `symbolic_var_matcher`, and must be before it in the struct
     * order.
     */
    arith::Analyzer analyzer;

    /*! \brief The map from symbolic var to its corresponding var in the fused function */
    tir::SymbolicMatcher symbolic_var_matcher =
        tir::SymbolicMatcher(&analyzer, &symbolic_var_remap);
  };

  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The name hint for the input func. */
  String func_name_;
  /*! \brief The helper info to fuse TIR prim_func */
  FuseFuncInfo func_info_;
  /*! \brief The tir function after fusion*/
  tir::PrimFunc fused_tir_;
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
  static IRModule Transform(const IRModule& mod) {
    Map<GlobalVar, BaseFunc> funcs_to_keep;
    for (const auto& [gv, func] : mod->functions) {
      // 1. If a TIR function has global symbol, we keep the function.
      // 2. Always keep ExternFunc.
      if (const auto* prim_func = func.as<tir::PrimFuncNode>()) {
        if (prim_func->GetAttr<String>("global_symbol").defined()) {
          funcs_to_keep.Set(gv, func);
        }
      } else if (func->IsInstance<ExternFuncNode>()) {
        funcs_to_keep.Set(gv, func);
      }
    }
    // Since TIRFuseMutator will delete bunch of PrimFunc, we create an empty block builder.
    TIRFuseMutator mutator(mod);
    // Step 1. Fuse all primitive relax functions, store the result in `fused_tir_funcs_`
    for (const auto& [gv, func] : mod->functions) {
      // Only fuse primitive relax functions
      if (func->IsInstance<relax::FunctionNode>() && func->HasNonzeroAttr(attr::kPrimitive)) {
        tir::PrimFunc fused_tir = FusedTIRConstructor::GetFusedTIR(mod, gv);
        mutator.fused_tir_funcs_.Set(gv, fused_tir);
      }
    }

    // Step 2. Update all non-primitive relax functions and add it, with the dependent function,
    // into the new IRModule
    for (const auto& [gv, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionNode>() && !func->HasNonzeroAttr(attr::kPrimitive)) {
        relax::Function update_func = Downcast<Function>(mutator.VisitExpr(func));
        mutator.builder_->AddFunction(update_func, gv->name_hint);
      }
    }

    // Step 3. Add all functions that need to be kept.
    auto modified_mod = mutator.builder_->GetContextIRModule();
    for (const auto& [gv, func] : funcs_to_keep) {
      if (!modified_mod->ContainGlobalVar(gv->name_hint)) {
        modified_mod->Add(gv, func);
      }
    }

    // Step 4. Copy over module attributes and return.
    if (mod->attrs.defined()) modified_mod = WithAttrs(modified_mod, mod->attrs->dict);
    return modified_mod;
  }

 private:
  explicit TIRFuseMutator(const IRModule& mod) : mod_(mod) {}

  using ExprMutator::VisitExpr_;

  // Get shape from call tir
  static Expr GetCallTIRShape(StructInfo sinfo) {
    if (auto* tuple = sinfo.as<TupleStructInfoNode>()) {
      Array<Expr> fields = tuple->fields.Map([&](StructInfo x) { return GetCallTIRShape(x); });
      return Tuple(fields);
    } else {
      auto* tensor = sinfo.as<TensorStructInfoNode>();
      ICHECK(tensor) << "FuseTIR can only take tensor or tuple type";
      auto* shape_expr = tensor->shape.as<ShapeExprNode>();
      ICHECK(shape_expr) << "FuseTIR requires all intermediate values have shape";
      return GetRef<ShapeExpr>(shape_expr);
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");

    Call call = Downcast<Call>(builder_->Normalize(ExprMutator::VisitExpr_(op)));

    if (call->op->IsInstance<GlobalVarNode>()) {
      // Case 1. It is a relax cross function call
      GlobalVar old_gv = Downcast<GlobalVar>(call->op);
      auto relax_func = Downcast<Function>(mod_->Lookup(old_gv));
      auto it = fused_tir_funcs_.find(old_gv);
      if (it != fused_tir_funcs_.end()) {
        const tir::PrimFunc& fused_tir = (*it).second;
        // Case 1.1. It calls a primitive relax function, update the call into a call_tir
        GlobalVar fused_tir_gv = this->builder_->AddFunction(fused_tir, old_gv->name_hint);
        // Step a. Flatten all args since call_tir does not support Tuple value.
        Array<Expr> arg_list;
        Array<PrimExpr> tir_vars;
        for (size_t i = 0; i < call->args.size(); ++i) {
          auto arg = call->args[i];
          auto sinfo = GetStructInfo(arg);

          ICHECK(!relax_func->params[i]->struct_info_->IsInstance<TupleStructInfoNode>() &&
                 !sinfo.as<TupleStructInfoNode>())
              << "InternalError: "
              << "All tuple parameters should be expanded before this point in FuseTIR.  "
              << "However, argument " << arg << " with struct info " << arg->struct_info_
              << " is passed as argument " << i << " to Primitive Relax function " << old_gv
              << ", which expects parameter " << relax_func->params[i] << " to have struct info "
              << relax_func->params[i]->struct_info_;

          if (const auto* shape = sinfo.as<ShapeStructInfoNode>()) {
            CHECK(shape->values.defined())
                << "FuseTIR requires all shape input has struct_info value.";
            for (const PrimExpr& prim_value : shape->values.value()) {
              CHECK(prim_value->IsInstance<tir::VarNode>())
                  << "All shape inputs are expected to be single tir var.";
              tir_vars.push_back(prim_value);
            }
          } else if (const auto* prim_value = sinfo.as<PrimStructInfoNode>()) {
            CHECK(prim_value->value.defined())
                << "FuseTIR requires all R.Prim arguments to have a known value.";
            PrimExpr expr = prim_value->value.value();
            CHECK(expr->IsInstance<tir::VarNode>())
                << "FuseTIR currently requires all R.Prim arguments to provide a single tir::Var.";
            tir_vars.push_back(expr);

          } else {
            arg_list.push_back(arg);
          }
        }
        // Step b. Create call_tir
        Array<Expr> call_args = {fused_tir_gv, Tuple(arg_list)};
        if (!tir_vars.empty()) {
          call_args.push_back(ShapeExpr(tir_vars));
        }
        return Call(call_tir_op_, call_args, call->attrs, {GetStructInfo(call)});
      } else {
        // Case 1.2. The callee function is not primitive, nothing to do.
        return call;
      }
    } else if (call->op == call_tir_op_) {
      // Case 2. It is a call_tir, re-emit the PrimFunc.
      if (const auto* gv = call->args[0].as<GlobalVarNode>()) {
        tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(GetRef<GlobalVar>(gv)));
        GlobalVar new_gv = this->builder_->AddFunction(func, gv->name_hint);
        Array<Expr> new_args = call->args;
        new_args.Set(0, new_gv);
        return Call(call->op, new_args, call->attrs, call->sinfo_args, call->span);
      }
    }

    // Case 3. CallNode in other types. Leave it as it is.
    return call;
  }

 private:
  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The map from global var of primitive relax function to generated prim func. */
  Map<GlobalVar, tir::PrimFunc> fused_tir_funcs_;
};

IRModule FuseTIR(IRModule mod) {
  mod = TIRFuseMutator::Transform(mod);
  return mod;
}

namespace transform {

Pass FuseTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
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
      },
      "FuseTIR");
}

TVM_REGISTER_GLOBAL("relax.transform.FuseTIR").set_body_typed(FuseTIR);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
