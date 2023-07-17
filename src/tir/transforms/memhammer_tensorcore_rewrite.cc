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

#include "./memhammer_rewrite_rule.h"

namespace tvm {
namespace tir {

/*!
 * \brief Tile the 2 innermost loops to extent=16. This helps further tensor core rewrite.
 * \param stmt The stmt
 * \return A pair. The first is the stmt after transformation.
 *         The second is the compute location where we may add write cache.
 */
std::pair<Stmt, Optional<For>> TileWmmaBlock(Stmt stmt) {
  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  int n = loops.size();
  PrimExpr extent_last1 = loops[n - 1]->extent;
  PrimExpr extent_last2 = loops[n - 2]->extent;
  {
    arith::Analyzer analyzer;
    if (!analyzer.CanProveEqual(floormod(extent_last1, 16), 0) ||
        !analyzer.CanProveEqual(floormod(extent_last2, 16), 0)) {
      return std::make_pair(stmt, NullOpt);
    }
  }
  Var new_loop_vars[4] = {
      /*0:*/ loops[n - 2]->loop_var.copy_with_suffix("_0"),
      /*1:*/ loops[n - 1]->loop_var.copy_with_suffix("_0"),
      /*2:*/ loops[n - 2]->loop_var.copy_with_suffix("_1"),
      /*3:*/ loops[n - 1]->loop_var.copy_with_suffix("_1"),
  };
  body = Substitute(std::move(body),
                    Map<Var, PrimExpr>{
                        {loops[n - 2]->loop_var, new_loop_vars[0] * 16 + new_loop_vars[2]},
                        {loops[n - 1]->loop_var, new_loop_vars[1] * 16 + new_loop_vars[3]},
                    });
  {
    PrimExpr factor[4] = {
        /*0:*/ floordiv(extent_last2, 16),  //
        /*1:*/ floordiv(extent_last1, 16),  //
        /*3:*/ 16,                          //
        /*4:*/ 16,                          //
    };
    body = For(new_loop_vars[3], 0, factor[3], ForKind::kSerial, std::move(body));
    body = For(new_loop_vars[2], 0, factor[2], ForKind::kSerial, std::move(body));
    body = For(new_loop_vars[1], 0, factor[1], ForKind::kSerial, std::move(body));
    body = For(new_loop_vars[0], 0, factor[0], ForKind::kSerial, std::move(body));
  }
  For compute_location = Downcast<For>(body);
  for (int i = n - 3; i >= 0; i--) {
    body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind, std::move(body),
               loops[i]->thread_binding, loops[i]->annotations);
  }
  return {body, compute_location};
}

Array<Range> RelaxIndices(const Array<PrimExpr>& indices, const Array<PrimExpr>& shape,
                          const Map<Var, arith::IntSet>& var_dom) {
  Array<arith::IntSet> int_set;
  int_set.reserve(indices.size());
  for (auto& indice : indices) {
    int_set.push_back(arith::EvalSet(indice, var_dom));
  }
  int ndim = int_set.size();
  Array<Range> region;
  region.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    region.push_back(int_set[i].CoverRange(Range::FromMinExtent(0, shape[i])));
  }
  return region;
}

/*!
 * \brief Rewrite the data copy that stores to wmma fragment with wmma::load_matrix_sync
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteWmmaLoad(Stmt stmt) {
  using arith::IntSet;
  const DataType dtype = DataType::Float(16);
  const DataType int32 = DataType::Int(32);

  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  int n = loops.size();

  Map<Var, IntSet> var_dom{
      {loops[n - 1]->loop_var, IntSet::FromMinExtent(loops[n - 1]->min, loops[n - 1]->extent)},
      {loops[n - 2]->loop_var, IntSet::FromMinExtent(loops[n - 2]->min, loops[n - 2]->extent)},
  };
  // TODO(tian): the assumption that the RHS of BufferStore is BufferLoad may not be accurate
  const BufferStoreNode* buf_store = TVM_TYPE_AS(body, BufferStoreNode);
  const BufferLoadNode* buf_load = TVM_TYPE_AS(buf_store->value, BufferLoadNode);

  Buffer src_buffer = buf_load->buffer;
  Buffer tgt_buffer = buf_store->buffer;
  std::string layout = tgt_buffer.scope() == "wmma.matrix_a" ? "row_major" : "col_major";
  Buffer new_src_buffer(
      /*data=*/Var("src", PointerType(PrimType(dtype), src_buffer.scope())),
      /*dtype=*/dtype,
      /*shape=*/{Integer(16), Integer(16)},
      /*strides=*/{Var("s1", int32), Var("s0", int32)},
      /*elem_offset=*/Var("src_elem_offset", int32),
      /*name=*/"src",
      /*data_alignment=*/64,
      /*offset_factor=*/16,
      /*buffer_type=*/kDefault);
  Buffer new_tgt_buffer(
      /*data=*/Var("tgt", PointerType(PrimType(dtype), tgt_buffer.scope())),
      /*dtype=*/dtype,
      /*shape=*/{Integer(16), Integer(16)},
      /*strides=*/{},
      /*elem_offset=*/Var("tgt_elem_offset", int32),
      /*name=*/"tgt",
      /*data_alignment=*/64,
      /*offset_factor=*/16,
      /*buffer_type=*/kDefault);
  Array<Range> read_region = RelaxIndices(buf_load->indices, src_buffer->shape, var_dom);
  Array<Range> write_region = RelaxIndices(buf_store->indices, tgt_buffer->shape, var_dom);
  Stmt wmma_body = BlockRealize(
      /*iter_values=*/{},
      /*predicate=*/Bool(true),
      Block(
          /*iter_vars=*/{},
          /*reads=*/{BufferRegion(src_buffer, read_region)},
          /*writes=*/{BufferRegion(tgt_buffer, write_region)},
          /*name_hint=*/"wmma_load",
          /*body=*/
          Evaluate(Call(
              /*data=*/runtime::DataType::Handle(),
              /*op=*/builtin::tvm_load_matrix_sync(),
              {
                  /*0:*/ new_tgt_buffer->data,
                  /*1:*/ 16,
                  /*2:*/ 16,
                  /*3:*/ 16,
                  /*4:*/ floordiv(new_tgt_buffer->elem_offset, 256) +
                      floordiv(floormod(new_tgt_buffer->elem_offset, 256), 16),
                  /*5:*/
                  Call(
                      /*dtype=*/runtime::DataType::Handle(),
                      /*op=*/builtin::tvm_access_ptr(),
                      /*args=*/
                      {
                          /*0:*/ TypeAnnotation(new_src_buffer->dtype),
                          /*1:*/ new_src_buffer->data,
                          /*2:*/ new_src_buffer->elem_offset,
                          /*3:*/ new_src_buffer->strides[new_src_buffer->strides.size() - 2] * 16,
                          /*4:*/ 1,
                      }),
                  /*6:*/ new_src_buffer->strides[new_src_buffer->strides.size() - 2],
                  /*7:*/ StringImm(layout),
              })),
          /*init=*/NullOpt,
          /*alloc_buffers=*/{},
          /*match_buffers=*/
          {
              /*0:*/ MatchBufferRegion(new_src_buffer, BufferRegion(src_buffer, read_region)),
              /*1:*/ MatchBufferRegion(new_tgt_buffer, BufferRegion(tgt_buffer, write_region)),
          },
          /*annotations=*/{}));
  for (int i = n - 3; i >= 0; i--) {
    wmma_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind,
                    std::move(wmma_body), loops[i]->thread_binding, loops[i]->annotations);
  }
  return wmma_body;
}

/*!
 * \brief Rewrite the data copy that loads from wmma fragment with wmma::store_matrix_sync
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteWmmaStore(Stmt stmt) {
  using arith::IntSet;
  const DataType int32 = DataType::Int(32);

  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  int n = loops.size();

  Map<Var, IntSet> var_dom{
      {loops[n - 1]->loop_var, IntSet::FromMinExtent(loops[n - 1]->min, loops[n - 1]->extent)},
      {loops[n - 2]->loop_var, IntSet::FromMinExtent(loops[n - 2]->min, loops[n - 2]->extent)},
  };
  // TODO(tian): the assumption that the RHS of BufferStore is BufferLoad may not be accurate
  const BufferStoreNode* buf_store = TVM_TYPE_AS(body, BufferStoreNode);
  const BufferLoadNode* buf_load = nullptr;
  PostOrderVisit(buf_store->value, [&](const ObjectRef& obj) {
    const BufferLoadNode* load = obj.as<BufferLoadNode>();
    if (load && load->buffer.scope() == "wmma.accumulator") {
      ICHECK(buf_load == nullptr || buf_load->buffer.same_as(load->buffer))
          << "More than one source buffer of wmma accumulator found";
      buf_load = load;
    }
    return true;
  });
  Buffer src_buffer = buf_load->buffer;
  Buffer tgt_buffer = buf_store->buffer;

  const DataType dtype = src_buffer->dtype;

  Buffer new_src_buffer(/*data=*/Var("src", PointerType(PrimType(dtype), src_buffer.scope())),
                        /*dtype=*/dtype,
                        /*shape=*/{Integer(16), Integer(16)},
                        /*strides=*/{},
                        /*elem_offset=*/Var("src_elem_offset", int32),
                        /*name=*/"src",
                        /*data_alignment=*/64,
                        /*offset_factor=*/16,
                        /*buffer_type=*/kDefault);
  Buffer new_tgt_buffer(/*data=*/Var("tgt", PointerType(PrimType(dtype), tgt_buffer.scope())),
                        /*dtype=*/dtype,
                        /*shape=*/{Integer(16), Integer(16)},
                        /*strides=*/{Var("s1", int32), Var("s0", int32)},
                        /*elem_offset=*/Var("tgt_elem_offset", int32),
                        /*name=*/"tgt",
                        /*data_alignment=*/64,
                        /*offset_factor=*/16,
                        /*buffer_type=*/kDefault);

  Array<Range> read_region = RelaxIndices(buf_load->indices, src_buffer->shape, var_dom);
  Array<Range> write_region = RelaxIndices(buf_store->indices, tgt_buffer->shape, var_dom);
  Stmt wmma_body = BlockRealize(
      /*iter_values=*/{},  //
      /*predicate=*/Bool(true),
      Block(/*iter_vars=*/{},
            /*reads=*/{BufferRegion(src_buffer, read_region)},
            /*writes=*/{BufferRegion(tgt_buffer, write_region)},
            /*name_hint=*/"wmma_store",
            Evaluate(Call(
                /*data=*/runtime::DataType::Handle(),
                /*op=*/builtin::tvm_store_matrix_sync(),
                {/*0:*/ new_src_buffer->data,
                 /*1:*/ 16,
                 /*2:*/ 16,
                 /*3:*/ 16,
                 /*4:*/ floordiv(new_src_buffer->elem_offset, 256) +
                     floordiv(floormod(new_src_buffer->elem_offset, 256), 16),
                 /*5:*/
                 Call(
                     /*data=*/runtime::DataType::Handle(),
                     /*op=*/builtin::tvm_access_ptr(),
                     {
                         /*0:*/ TypeAnnotation(new_tgt_buffer->dtype),
                         /*1:*/ new_tgt_buffer->data,
                         /*2:*/ new_tgt_buffer->elem_offset,
                         /*3:*/ new_tgt_buffer->strides[0] * 16,
                         /*4:*/ 2,
                     }),
                 /*6:*/ new_tgt_buffer->strides[0],
                 /*7:*/ StringImm("row_major")})),
            /*init=*/NullOpt,
            /*alloc_buffers=*/{},
            /*match_buffers=*/
            {
                MatchBufferRegion(new_src_buffer, BufferRegion(src_buffer, read_region)),
                MatchBufferRegion(new_tgt_buffer, BufferRegion(tgt_buffer, write_region)),
            },
            /*annotations=*/{}));
  for (int i = n - 3; i >= 0; i--) {
    wmma_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind,
                    std::move(wmma_body), loops[i]->thread_binding, loops[i]->annotations);
  }
  return wmma_body;
}

Stmt SharedToWmma::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                           OutputSet* output) const {
  Stmt after_tiling = TileWmmaBlock(stmt).first;
  output->padding_min.Set(constraints.read_region->buffer, 8);
  return RewriteWmmaLoad(after_tiling);
}

Stmt WmmaToShared::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                           OutputSet* output) const {
  Stmt after_tiling = TileWmmaBlock(stmt).first;
  output->padding_min.Set(constraints.write_region->buffer, 8);
  return RewriteWmmaStore(after_tiling);
}

class WmmaToGlobalRewriter : public StmtExprMutator {
 public:
  WmmaToGlobalRewriter(const SeqStmtNode* tgt_stmt, const ConstraintSet& constraints)
      : tgt_stmt_(tgt_stmt), constraints_(constraints) {}

 private:
  Stmt VisitStmt_(const SeqStmtNode* op) final {
    if (op == tgt_stmt_) {
      ICHECK_EQ(op->seq.size(), 2);
      Stmt wmma_to_shared = RewriteWmmaStore(op->seq[0]);
      Stmt shared_to_global = CoalescedAccess().Rewrite(op->seq[1], constraints_, nullptr);
      return SeqStmt({wmma_to_shared, shared_to_global});
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  const SeqStmtNode* tgt_stmt_;
  const ConstraintSet& constraints_;
};

Stmt WmmaToGlobal::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                           OutputSet* output) const {
  Stmt body{nullptr};
  Optional<For> compute_location{nullptr};
  std::tie(body, compute_location) = TileWmmaBlock(stmt);
  SeqStmt seq{nullptr};
  Buffer cache_buffer;
  // Step 1. add a shared memory cache
  std::tie(body, seq) = InsertCacheStage(std::move(body), true, "shared.dyn", compute_location,
                                         constraints.outer_loops, &cache_buffer);
  output->alloc_buffer.push_back(cache_buffer);
  output->padding_min.Set(cache_buffer, 8);
  // Step 2. do coalesced rewrite and tensor core rewrite respectively for 2 parts
  WmmaToGlobalRewriter rewriter(seq.get(), constraints);
  return rewriter(body);
}

std::pair<Stmt, Optional<For>> TileMmaToGlobalBlock(Stmt stmt) {
  // i, j = sch.get_loops(block)[2:]
  // i_0, i_1 = sch.split(i, factors=[None, 8])
  // j_0, j_1 = sch.split(j, factors=[None, 8])
  // sch.reorder(i_0, j_0, i_1, j_1)
  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  int n = loops.size();
  PrimExpr extent_last1 = loops[n - 1]->extent;
  PrimExpr extent_last2 = loops[n - 2]->extent;
  {
    arith::Analyzer analyzer;
    // Only tile when both extent % 8 == 0
    if (!analyzer.CanProveEqual(floormod(extent_last1, 8), 0) ||
        !analyzer.CanProveEqual(floormod(extent_last2, 8), 0)) {
      return std::make_pair(stmt, NullOpt);
    }
  }
  Var new_loop_vars[4] = {
      /*0:*/ loops[n - 2]->loop_var.copy_with_suffix("_0"),
      /*1:*/ loops[n - 1]->loop_var.copy_with_suffix("_0"),
      /*2:*/ loops[n - 2]->loop_var.copy_with_suffix("_1"),
      /*3:*/ loops[n - 1]->loop_var.copy_with_suffix("_1"),
  };
  body = Substitute(std::move(body),
                    Map<Var, PrimExpr>{
                        {loops[n - 2]->loop_var, new_loop_vars[0] * 8 + new_loop_vars[2]},
                        {loops[n - 1]->loop_var, new_loop_vars[1] * 8 + new_loop_vars[3]},
                    });
  {
    PrimExpr factor[4] = {
        /*0:*/ floordiv(extent_last2, 8),  //
        /*1:*/ floordiv(extent_last1, 8),  //
        /*3:*/ 8,                          //
        /*4:*/ 8,                          //
    };
    body = For(new_loop_vars[3], 0, factor[3], ForKind::kSerial, std::move(body));
    body = For(new_loop_vars[2], 0, factor[2], ForKind::kSerial, std::move(body));
    body = For(new_loop_vars[1], 0, factor[1], ForKind::kSerial, std::move(body));
    body = For(new_loop_vars[0], 0, factor[0], ForKind::kSerial, std::move(body));
  }
  For compute_location = Downcast<For>(body);
  for (int i = n - 3; i >= 0; i--) {
    body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind, std::move(body),
               loops[i]->thread_binding, loops[i]->annotations);
  }
  return {body, compute_location};
}

/*!
 * \brief Rewrite the data copy that loads from mma fragment
 * \param stmt The stmt to rewrite
 * \return The stmt after rewrite
 */
Stmt RewriteMmaStore(Stmt stmt) {
  using arith::IntSet;
  const DataType int32 = DataType::Int(32);

  // Step 1. Get inner loop body
  Stmt body = stmt;
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  int n = loops.size();

  Map<Var, IntSet> var_dom{
      {loops[n - 1]->loop_var, IntSet::FromMinExtent(loops[n - 1]->min, loops[n - 1]->extent)},
      {loops[n - 2]->loop_var, IntSet::FromMinExtent(loops[n - 2]->min, loops[n - 2]->extent)},
  };

  // Step 2. Find matrixC buffer
  const BufferStoreNode* buf_store = TVM_TYPE_AS(body, BufferStoreNode);
  const BufferLoadNode* buf_load = nullptr;
  PostOrderVisit(buf_store->value, [&](const ObjectRef& obj) {
    const BufferLoadNode* load = obj.as<BufferLoadNode>();
    if (load && load->buffer.scope() == "m16n8k8.matrixC") {
      ICHECK(buf_load == nullptr || buf_load->buffer.same_as(load->buffer))
          << "More than one source buffer of mma accumulator found";
      buf_load = load;
    }
    return true;
  });

  // Step 3. Create new mma body
  // We have the assumption that two innermost loops are the 8 * 8 loop generated by
  // TileMmaToGlobalBlock. Here, we rewrite the 8 * 8 loop to have the threadIdx.x
  // binding so that each thread only read from their part of buffer, and write to
  // corresponding place in shared memory.
  // Please refer to `9.7.13.4.7. Matrix Fragments for mma.m16n8k8` in
  // https://docs.nvidia.com/cuda/archive/11.1.0/pdf/ptx_isa_7.1.pdf

  // Step 3.1. Generate new buffer
  Buffer src_buffer = buf_load->buffer;
  Buffer tgt_buffer = buf_store->buffer;
  const DataType dtype = src_buffer->dtype;
  Buffer new_src_buffer(/*data=*/Var("src", PointerType(PrimType(dtype), src_buffer.scope())),
                        /*dtype=*/dtype,
                        /*shape=*/{Integer(8), Integer(8)},
                        /*strides=*/{},
                        /*elem_offset=*/Var("src_elem_offset", int32),
                        /*name=*/"src",
                        /*data_alignment=*/64,
                        /*offset_factor=*/8,
                        /*buffer_type=*/kDefault);
  Buffer new_tgt_buffer(/*data=*/Var("tgt", PointerType(PrimType(dtype), tgt_buffer.scope())),
                        /*dtype=*/dtype,
                        /*shape=*/{Integer(8), Integer(8)},
                        /*strides=*/{Var("s1", int32), Var("s0", int32)},
                        /*elem_offset=*/Var("tgt_elem_offset", int32),
                        /*name=*/"tgt",
                        /*data_alignment=*/64,
                        /*offset_factor=*/8,
                        /*buffer_type=*/kDefault);

  // Step 3.2. Generate new r/w region
  Array<Range> read_region = RelaxIndices(buf_load->indices, src_buffer->shape, var_dom);
  Array<Range> write_region = RelaxIndices(buf_store->indices, tgt_buffer->shape, var_dom);

  // Step 3.3. Generate new inner loop body
  // for v in T.vectorized(2):
  //   tgt[tx // 4, (tx % 4) * 2 + vec] = src[tx // 4, (tx % 4) * 2 + vec]
  Var tx = Var("tx");
  Var vec = Var("vec");
  Stmt mma_body = BlockRealize(
      /*iter_values=*/{},  //
      /*predicate=*/Bool(true),
      Block(/*iter_vars=*/{},
            /*reads=*/{BufferRegion(src_buffer, read_region)},
            /*writes=*/{BufferRegion(tgt_buffer, write_region)},
            /*name_hint=*/"mma_store",
            AttrStmt(/*node=*/IterVar(
                         /*dom=*/Range::FromMinExtent(0, 32),
                         /*var=*/tx,
                         /*iter_type=*/IterVarType::kThreadIndex,
                         /*thread_tag=*/"threadIdx.x"),
                     /*attr_key=*/"thread_extent",
                     /*value=*/Integer(32),
                     /*body=*/
                     For(vec, 0, 2, ForKind::kVectorized,
                         /*body=*/
                         BufferStore(new_tgt_buffer,
                                     BufferLoad(new_src_buffer,
                                                {floordiv(tx, 4), floormod(tx, 4) * 2 + vec}),
                                     {floordiv(tx, 4), floormod(tx, 4) * 2 + vec}),
                         /*annotations=*/{})),
            /*init=*/NullOpt,
            /*alloc_buffers=*/{},
            /*match_buffers=*/
            {
                MatchBufferRegion(new_src_buffer, BufferRegion(src_buffer, read_region)),
                MatchBufferRegion(new_tgt_buffer, BufferRegion(tgt_buffer, write_region)),
            },
            /*annotations=*/{}));

  // Step 3.4. wrap outer loops
  for (int i = n - 3; i >= 0; i--) {
    mma_body = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent, loops[i]->kind,
                   std::move(mma_body), loops[i]->thread_binding, loops[i]->annotations);
  }
  return mma_body;
}

class MmaToGlobalRewriter : public StmtExprMutator {
 public:
  MmaToGlobalRewriter(const SeqStmtNode* tgt_stmt, const ConstraintSet& constraints)
      : tgt_stmt_(tgt_stmt), constraints_(constraints) {}

 private:
  Stmt VisitStmt_(const SeqStmtNode* op) final {
    if (op == tgt_stmt_) {
      ICHECK_EQ(op->seq.size(), 2);
      // Rewrite for local to shared.dyn
      // In this rewrite, we store local matrixC buffer to corresponding place in shared memory
      Stmt mma_to_shared = RewriteMmaStore(op->seq[0]);
      // Coalesce access for shared.dyn to global
      Stmt shared_to_global = CoalescedAccess().Rewrite(op->seq[1], constraints_, nullptr);
      return SeqStmt({mma_to_shared, shared_to_global});
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  const SeqStmtNode* tgt_stmt_;
  const ConstraintSet& constraints_;
};

Stmt MmaToGlobal::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                          OutputSet* output) const {
  Stmt body{nullptr};
  Optional<For> compute_location{nullptr};
  std::tie(body, compute_location) = TileMmaToGlobalBlock(stmt);
  SeqStmt seq{nullptr};
  Buffer cache_buffer;
  // Step 1. add a shared memory cache
  std::tie(body, seq) = InsertCacheStage(std::move(body), true, "shared.dyn", compute_location,
                                         constraints.outer_loops, &cache_buffer);
  output->alloc_buffer.push_back(cache_buffer);
  output->padding_min.Set(cache_buffer, 8);
  // Step 2. do coalesced rewrite and tensor core rewrite respectively for 2 parts
  MmaToGlobalRewriter rewriter(seq.get(), constraints);
  return rewriter(body);
}

}  // namespace tir
}  // namespace tvm
