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

#include <string>

#include "../utils.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

bool HasBuffer(const Array<BufferRegion>& buffer_regions, const Buffer& buffer) {
  for (const BufferRegion& buffer_region : buffer_regions) {
    if (buffer_region->buffer.same_as(buffer)) {
      return true;
    }
  }
  return false;
}

void RelaxBufferRegions(const Array<BufferRegion>& buffer_regions,
                        const Buffer& buffer,                    //
                        const Map<Var, arith::IntSet>& var_dom,  //
                        const Map<Var, PrimExpr>& bindings,      //
                        std::vector<NDIntSet>* relaxed_regions) {
  for (const BufferRegion& buffer_region : buffer_regions) {
    if (buffer_region->buffer.same_as(buffer)) {
      Array<arith::IntSet> relaxed_region =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions->push_back({relaxed_region.begin(), relaxed_region.end()});
    }
  }
}

class ScopeReplacer : public StmtMutator {
 public:
  static Block Replace(const BlockNode* scope_block, const Buffer& dst, const ForNode* old_loop,
                       const ForNode* new_loop) {
    ObjectPtr<BlockNode> new_scope_block = make_object<BlockNode>(*scope_block);
    new_scope_block->body = ScopeReplacer(old_loop, new_loop)(std::move(new_scope_block->body));
    new_scope_block->alloc_buffers.push_back(dst);
    return Block(new_scope_block);
  }

 private:
  explicit ScopeReplacer(const ForNode* old_loop, const ForNode* new_loop)
      : old_loop_(old_loop), new_loop_(new_loop), found_(false) {}

  Stmt VisitStmt(const Stmt& stmt) final { return found_ ? stmt : StmtMutator::VisitStmt(stmt); }
  Stmt VisitStmt_(const BlockNode* block) final { return GetRef<Block>(block); }
  Stmt VisitStmt_(const ForNode* loop) final {
    if (loop == old_loop_) {
      found_ = true;
      return GetRef<For>(new_loop_);
    }
    return StmtMutator::VisitStmt_(loop);
  }

  const ForNode* old_loop_;
  const ForNode* new_loop_;
  bool found_;
};

class ReadWriteAtBufferReplacer : public StmtExprMutator {
 public:
  explicit ReadWriteAtBufferReplacer(const Buffer& src, const Buffer& dst,
                                     Map<Block, Block>* block_sref_reuse)
      : src_(src), dst_(dst), block_sref_reuse_(block_sref_reuse) {}

 private:
  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (store->buffer.same_as(src_)) {
      ObjectPtr<BufferStoreNode> new_store = make_object<BufferStoreNode>(*store.get());
      new_store->buffer = dst_;
      return BufferStore(new_store);
    }
    return store;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (load->buffer.same_as(src_)) {
      ObjectPtr<BufferLoadNode> new_load = make_object<BufferLoadNode>(*load.get());
      new_load->buffer = dst_;
      return BufferLoad(new_load);
    }
    return load;
  }

  Stmt VisitStmt_(const BlockNode* _block) final {
    Block old_block = GetRef<Block>(_block);
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(_block));
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block.get());
    new_block->reads = ReplaceBuffer(new_block->reads, src_, dst_);
    new_block->writes = ReplaceBuffer(new_block->writes, src_, dst_);
    block_sref_reuse_->Set(old_block, Block(new_block));
    return Block(new_block);
  }

  const Buffer& src_;
  const Buffer& dst_;
  Map<Block, Block>* block_sref_reuse_;
};

struct ReadWriteAtImpl {
  template <bool is_read>
  static StmtSRef Main(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                       int buffer_index, const String& storage_scope,
                       Map<String, ObjectRef> annotations) {
    const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
    Buffer src = GetNthAccessBuffer(self, GetRef<Block>(block), buffer_index,
                                    is_read ? BufferIndexType::kRead : BufferIndexType::kWrite);
    Buffer dst = WithScope(src, storage_scope);
    ReadWriteAtImpl impl(self, loop_sref, src, dst, annotations);
    std::pair<For, BlockRealize> new_loop_block =
        impl.MakeLoopAndBlock<is_read>(src->name + "_" + storage_scope);
    StmtSRef result_block_sref =
        impl.ReplaceScopeBlock(new_loop_block.first.get(), new_loop_block.second->block.get());
    impl.UpdateBlockInfo(result_block_sref, !new_loop_block.second->iter_values.empty());
    return result_block_sref;
  }

 private:
  static Map<Var, Range> GetLoopDomain(const StmtSRefNode* loop_sref) {
    Map<Var, Range> result;
    for (const ForNode* loop; (loop = loop_sref->StmtAs<ForNode>()) != nullptr;
         loop_sref = loop_sref->parent) {
      result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    return result;
  }

  StmtSRef ReplaceScopeBlock(const ForNode* new_loop, const BlockNode* new_block) {
    StmtSRef scope_root_sref = GetScopeRoot(self_, loop_sref_,
                                            /*require_stage_pipeline=*/true);
    const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);
    Block new_scope_block = ScopeReplacer::Replace(scope_block, dst_, loop_, new_loop);
    block_sref_reuse_.Set(GetRef<Block>(scope_block), new_scope_block);
    self_->Replace(scope_root_sref, new_scope_block, block_sref_reuse_);
    return self_->stmt2ref.at(new_block);
  }

  void UpdateBlockInfo(const StmtSRef& new_block_sref, bool affine_binding) {
    BlockInfo& block_info = self_->block_info[new_block_sref];
    block_info.affine_binding = affine_binding;
    block_info.region_cover = true;
    block_info.stage_pipeline = true;
  }

  template <bool is_read>
  std::pair<For, BlockRealize> MakeLoopAndBlock(const String& new_block_name_hint) {
    Array<Stmt> subtrees = AsArray(loop_->body);
    int n_subtrees = subtrees.size();
    runtime::StorageScope scope = runtime::StorageScope::Create(dst_.scope());
    std::vector<NDIntSet> relaxed_regions;
    std::vector<int> r_pos;
    std::vector<int> w_pos;
    relaxed_regions.reserve(n_subtrees);
    r_pos.reserve(n_subtrees);
    w_pos.reserve(n_subtrees);
    // Step 1. Iterate over all subtrees
    for (int i = 0; i < n_subtrees; ++i) {
      bool r_visited = false;
      bool w_visited = false;
      auto f_visit = [this, &relaxed_regions, &r_visited, &w_visited,
                      &scope](const ObjectRef& obj) -> bool {
        const BlockRealizeNode* realize = obj.as<BlockRealizeNode>();
        if (realize == nullptr) {
          return true;
        }
        const BlockNode* block = realize->block.get();
        bool has_r = HasBuffer(block->reads, src_);
        bool has_w = HasBuffer(block->writes, src_);
        r_visited = r_visited || has_r;
        w_visited = w_visited || has_w;
        if (is_read ? has_r : has_w) {
          RelaxBufferRegions(
              /*buffer_regions=*/is_read ? block->reads : block->writes,
              /*buffer=*/src_,
              /*var_dom=*/
              arith::AsIntSet(LoopDomainOfSRefTreePath(
                  /*low_inclusive=*/GetRef<StmtSRef>(self_->stmt2ref.at(block)->parent),
                  /*high_exclusive=*/loop_sref_,
                  /*extra_relax_scope=*/scope)),
              /*bindings=*/GetBindings(GetRef<BlockRealize>(realize)),
              /*relaxed_regions=*/&relaxed_regions);
        }
        return false;
      };
      PreOrderVisit(subtrees[i], f_visit);
      if (r_visited) {
        r_pos.push_back(i);
      }
      if (w_visited) {
        w_pos.push_back(i);
      }
    }
    // Step 2. Calculate `insert_pos` and [st, ed) for buffer replacement
    int insert_pos = -1, st = -1, ed = -1;
    if (is_read) {
      ICHECK(!r_pos.empty());
      // No write after the first read
      ICHECK(w_pos.empty() || w_pos.back() < r_pos.front());
      // Can be inserted at [0, r_pos.front()], i.e. before the first read
      insert_pos = r_pos.front();
      // Buffer reads in [insert_pos, +oo) is rewritten
      st = insert_pos;
      ed = n_subtrees;
    } else {
      ICHECK(!w_pos.empty());
      // No read after the last write
      ICHECK(r_pos.empty() || r_pos.back() <= w_pos.back());
      // Can be inserted into (w_pos.back(), +oo), i.e. after the last write
      insert_pos = w_pos.back() + 1;
      st = 0;
      ed = insert_pos;
    }
    // Step 3. Calculate `domain`, the domain of buffer access
    NDIntSet relaxed = support::NDIntSetUnion(relaxed_regions);
    int ndim = relaxed.size();
    Array<Range> domain;
    domain.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      const arith::IntSet& int_set = relaxed[i];
      PrimExpr min = analyzer_->Simplify(int_set.min());
      PrimExpr extent = analyzer_->Simplify(int_set.max() + 1 - min);
      domain.push_back(Range::FromMinExtent(min, extent));
    }
    // Step 4. Insert the auto copy block and replace buffers
    ReadWriteAtBufferReplacer replacer(src_, dst_, &block_sref_reuse_);
    for (int i = st; i < ed; ++i) {
      Stmt stmt = subtrees[i];
      subtrees.Set(i, Stmt(nullptr));
      subtrees.Set(i, replacer(std::move(stmt)));
    }
    BlockRealize realize =
        is_read
            ? MakeBlock(src_, dst_, new_block_name_hint, GetLoopDomain(loop_sref_.get()), domain)
            : MakeBlock(dst_, src_, new_block_name_hint, GetLoopDomain(loop_sref_.get()), domain);
    subtrees.insert(subtrees.begin() + insert_pos, realize);
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop_);
    new_loop->body = SeqStmt(std::move(subtrees));
    return {For(new_loop), realize};
  }

  BlockRealize MakeBlock(const Buffer& copy_from, const Buffer& copy_to, const String& name_hint,
                         const Map<Var, Range>& loop_domain, Array<Range> domain) const {
    int n = domain.size();
    std::vector<Var> loop_vars;
    loop_vars.reserve(n);
    for (int i = 0; i < n; ++i) {
      loop_vars.push_back(Var("ax" + std::to_string(i)));
    }
    Map<Var, PrimExpr> bindings;
    Array<IterVar> iter_vars;
    Array<PrimExpr> iter_values;
    Array<PrimExpr> indices;
    iter_vars.reserve(n);
    iter_values.reserve(n);
    indices.reserve(n);
    for (int i = 0; i < n; ++i) {
      auto f_substitute = [&loop_domain, &bindings, &iter_vars,
                           &iter_values](const Var& var) -> Optional<PrimExpr> {
        auto it = bindings.find(var);
        if (it != bindings.end()) {
          return (*it).second;
        }
        Range range = loop_domain.at(var);
        ObjectPtr<VarNode> v = make_object<VarNode>(*var.get());
        v->name_hint = "v" + std::to_string(iter_vars.size());
        bindings.Set(var, Var(v));
        iter_values.push_back(var);
        iter_vars.push_back(IterVar(range, Var(v), IterVarType::kDataPar));
        return Var(v);
      };
      ObjectPtr<RangeNode> dom = make_object<RangeNode>(*domain[i].get());
      dom->min = Substitute(std::move(dom->min), f_substitute);
      dom->extent = Substitute(std::move(dom->extent), f_substitute);
      domain.Set(i, Range(dom));
    }
    for (int i = 0; i < n; ++i) {
      indices.push_back(domain[i]->min + loop_vars[i]);
    }
    Stmt stmt = BufferStore(copy_to, /*value=*/BufferLoad(copy_from, indices), /*indices=*/indices);
    for (int i = n - 1; i >= 0; --i) {
      stmt = For(loop_vars[i], Integer(0), domain[i]->extent, ForKind::kSerial, stmt);
    }
    return BlockRealize(
        /*values=*/iter_values,
        /*predicate=*/const_true(),
        Block(/*iter_vars=*/iter_vars,
              /*reads=*/{BufferRegion(copy_from, domain)},
              /*writes=*/{BufferRegion(copy_to, domain)},
              /*name_hint=*/name_hint,  //
              /*body=*/std::move(stmt),
              /*init=*/NullOpt,
              /*alloc_buffers=*/{},
              /*match_buffers=*/{},
              /*annotations=*/annotations_));
  }

  explicit ReadWriteAtImpl(ScheduleState self, const StmtSRef& loop_sref, const Buffer& src,
                           const Buffer& dst, Map<String, ObjectRef> annotations)
      : self_(self),
        loop_sref_(loop_sref),
        loop_(nullptr),
        src_(src),
        dst_(dst),
        annotations_(annotations),
        block_sref_reuse_(),
        analyzer_(std::make_unique<arith::Analyzer>()) {
    loop_ = TVM_SREF_TO_FOR(loop_sref);
  }

  ScheduleState self_;
  const StmtSRef& loop_sref_;
  const ForNode* loop_;
  const Buffer& src_;
  const Buffer& dst_;
  Map<String, ObjectRef> annotations_;
  Map<Block, Block> block_sref_reuse_;
  std::unique_ptr<arith::Analyzer> analyzer_;
};

StmtSRef ReadAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                int read_buffer_index, const String& storage_scope) {
  return ReadWriteAtImpl::Main<true>(self, loop_sref, block_sref, read_buffer_index, storage_scope,
                                     {{tir::attr::auto_copy, Integer(1)}});
}

StmtSRef WriteAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                 int write_buffer_index, const String& storage_scope) {
  return ReadWriteAtImpl::Main<false>(self, loop_sref, block_sref, write_buffer_index,
                                      storage_scope, {{tir::attr::auto_copy, Integer(1)}});
}

/******** Instruction Registration ********/

struct ReadAtTraits : public UnpackedInstTraits<ReadAtTraits> {
  static constexpr const char* kName = "ReadAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  StmtSRef ReadAt(ScheduleState self, const StmtSRef& loop_sref, const StmtSRef& block_sref,
                  int buffer_index, const String& storage_scope);
  static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop, BlockRV block,
                                         Integer read_buffer_index, String storage_scope) {
    return sch->ReadAt(loop, block, read_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop, String block,
                                 Integer read_buffer_index, String storage_scope) {
    PythonAPICall py("read_at");
    py.Input("loop", loop);
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct WriteAtTraits : public UnpackedInstTraits<WriteAtTraits> {
  static constexpr const char* kName = "WriteAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop, BlockRV block,
                                         Integer write_buffer_index, String storage_scope) {
    return sch->WriteAt(loop, block, write_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop, String block,
                                 Integer write_buffer_index, String storage_scope) {
    PythonAPICall py("write_at");
    py.Input("loop", loop);
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ReadAtTraits);
TVM_REGISTER_INST_KIND_TRAITS(WriteAtTraits);

}  // namespace tir
}  // namespace tvm
