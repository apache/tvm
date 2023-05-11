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
#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Check whether the loop has any annotation
 * \param sref The sref of loop
 * \return Whether the loop has any annotation
 */
inline bool HasAnnOrBinding(const ForNode* loop) {
  return loop->kind == ForKind::kThreadBinding || !loop->annotations.empty();
}

/*! \brief The visitor for extracting the stride of a var in a PrimExpr. */
class StrideExtractor : public ExprVisitor {
 public:
  /*!
   * \brief Extracting the stride of a var in a PrimExpr.
   *        e.g the stride of `x` in `(x * 2 + 1) * 3 + 1` is 6
   * \param expr The given PrimExpr.
   * \param var The target var.
   * \return The stride of the var.
   */
  static int64_t Extract(const PrimExpr& expr, const Var& var) {
    StrideExtractor extractor(var);
    extractor.VisitExpr(expr);
    return extractor.strides_[expr.get()];
  }

 private:
  explicit StrideExtractor(const Var& var) : var_(var) {}

  void VisitExpr_(const MulNode* node) final {
    ExprVisitor::VisitExpr_(node);

    if (const auto* a = node->a.as<IntImmNode>()) {
      if (strides_.count(node->b.get())) {
        strides_[node] = strides_[node->b.get()] * a->value;
      }
    } else if (const auto* b = node->b.as<IntImmNode>()) {
      if (strides_.count(node->a.get())) {
        strides_[node] = strides_[node->a.get()] * b->value;
      }
    }
  }

  void VisitExpr_(const AddNode* node) final {
    ExprVisitor::VisitExpr_(node);
    int64_t stride_a, stride_b;
    if (strides_.count(node->a.get())) {
      stride_a = strides_[node->a.get()];
    } else {
      stride_a = INT64_MAX;
    }
    if (strides_.count(node->b.get())) {
      stride_b = strides_[node->b.get()];
    } else {
      stride_b = INT64_MAX;
    }
    if (stride_a != INT64_MAX || stride_b != INT64_MAX) {
      strides_[node] = std::min(stride_a, stride_b);
    }
  }

  void VisitExpr_(const VarNode* node) final {
    if (node == var_.get()) {
      strides_[node] = 1;
    }
  }

  const Var& var_;
  std::unordered_map<const PrimExprNode*, int64_t> strides_;
};

struct ParsedAnnotation {
  int max_parallel_extent;
  int max_vectorize_extent;
  int unroll_explicit;
  int unroll_implicit;
  int num_parallel_loops;
  int num_vectorize_loops;
};

bool ParseAnnotation(const Block& block, ParsedAnnotation* parsed) {
  bool found = false;
  *parsed = ParsedAnnotation{-1, -1, -1, -1, -1, -1};
  for (const auto& ann : block->annotations) {
    if (ann.first == attr::meta_schedule_parallel) {
      found = true;
      if (const auto* imm = ann.second.as<tir::IntImmNode>()) {
        parsed->max_parallel_extent = imm->value;
      }
    } else if (ann.first == attr::meta_schedule_vectorize) {
      found = true;
      if (const auto* imm = ann.second.as<tir::IntImmNode>()) {
        parsed->max_vectorize_extent = imm->value;
      }
    } else if (ann.first == attr::meta_schedule_unroll_explicit) {
      found = true;
      if (const auto* imm = ann.second.as<tir::IntImmNode>()) {
        parsed->unroll_explicit = imm->value;
      }
    } else if (ann.first == attr::meta_schedule_unroll_implicit) {
      found = true;
      if (const auto* imm = ann.second.as<tir::IntImmNode>()) {
        parsed->unroll_implicit = imm->value;
      }
    }
  }
  return found;
}

void RemoveParsedAnn(const Schedule& sch, const BlockRV& block_rv, const ParsedAnnotation& parsed) {
  if (parsed.max_parallel_extent != -1) {
    sch->Unannotate(block_rv, attr::meta_schedule_parallel);
  }
  if (parsed.max_vectorize_extent != -1) {
    sch->Unannotate(block_rv, attr::meta_schedule_vectorize);
  }
  if (parsed.unroll_explicit != -1) {
    sch->Unannotate(block_rv, attr::meta_schedule_unroll_explicit);
  }
  if (parsed.unroll_implicit != -1) {
    sch->Unannotate(block_rv, attr::meta_schedule_unroll_implicit);
  }
}

int CalculateNumRewritableLoops(const Array<StmtSRef>& loop_srefs,
                                const std::vector<int>& loop_types) {
  int rw_loops_num = 0;
  ICHECK_EQ(loop_srefs.size(), loop_types.size());
  for (size_t i = 0; i < loop_srefs.size(); ++i) {
    const StmtSRef& loop_sref = loop_srefs[i];
    const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
    if (HasAnnOrBinding(loop)) {
      continue;
    }
    // Cannot vectorize reduce axis
    if (loop_types[i] != IterVarType::kDataPar) {
      continue;
    }
    // Cannot fuse with a loop with multiple children
    if (!IsSingleStmt(loop->body)) {
      continue;
    }
    // Check if the loop extent is valid
    if (GetLoopIntExtent(loop_sref) == nullptr) {
      continue;
    }
    ++rw_loops_num;
  }
  return rw_loops_num;
}

void AdjustParallelVectorize(const Schedule& sch, const BlockRV& block_rv,
                             const Array<LoopRV>& loop_rvs, ParsedAnnotation* parsed) {
  StmtSRef block_sref = sch->GetSRef(block_rv);
  if (parsed->max_parallel_extent == -1 && parsed->max_vectorize_extent == -1) {
    return;
  }
  const int n_loops = loop_rvs.size();
  if (n_loops == 0) {
    parsed->max_parallel_extent = -1;
    parsed->max_vectorize_extent = -1;
    return;
  }
  // Extract loop_srefs, and calculate the iterator types
  Array<StmtSRef> loop_srefs;
  std::vector<int> loop_types;
  {
    loop_srefs.reserve(n_loops);
    loop_types.reserve(n_loops);
    for (const LoopRV& loop_rv : loop_rvs) {
      loop_srefs.push_back(sch->GetSRef(loop_rv));
      loop_types.push_back(GetLoopIterType(loop_srefs.back()));
    }
  }
  // check the maximal number of axes that are vectorizable (contiguous memory access)
  BlockRealize realize = GetBlockRealize(sch->state(), block_sref);
  Array<BufferRegion> buffer_access(realize->block->reads);
  buffer_access.insert(buffer_access.end(), realize->block->writes.begin(),
                       realize->block->writes.end());
  std::unordered_map<const VarNode*, PrimExpr> binding_map;
  for (size_t i = 0; i < realize->iter_values.size(); i++) {
    binding_map[realize->block->iter_vars[i]->var.get()] = realize->iter_values[i];
  }
  int max_fusible = INT32_MAX;
  // for each block read/write, get the strides of the loop vars and find the fusible
  // (vectorizable) axes
  for (const BufferRegion& access : buffer_access) {
    int fusible = 0;
    std::vector<int64_t> strides;
    // get strides for each loop var
    for (const StmtSRef& loop_sref : loop_srefs) {
      int64_t stride = 0, buffer_stride = 1;
      const auto* var = loop_sref->StmtAs<ForNode>();
      arith::Analyzer analyzer;
      for (int i = access->region.size() - 1; i >= 0; i--) {
        PrimExpr idx = analyzer.Simplify(Substitute(access->region[i]->min, binding_map));
        int64_t coef = StrideExtractor::Extract(idx, var->loop_var);
        if (coef != 0) {
          stride = coef * buffer_stride;
          break;
        }
        buffer_stride *= access->buffer->shape[i].as<IntImmNode>()->value;
      }
      strides.push_back(stride);
    }
    int prev_used_iter = -1;
    // check the number of fusible loops
    for (int i = strides.size() - 1; i >= 0; i--) {
      if (strides[i] == 0) {
        // not used in the buffer access, safe to fuse
        fusible++;
        continue;
      } else if (prev_used_iter == -1) {
        // the stride of last axis is not 1 means the memory access is not contiguous
        if (strides[i] != 1 && fusible != 0) {
          break;
        }
        fusible++;
        prev_used_iter = i;
      } else {
        // contiguous memory access
        const auto* prev_loop = loop_srefs[prev_used_iter]->StmtAs<ForNode>();
        int64_t prev_used_iter_extent = prev_loop->extent.as<IntImmNode>()->value;
        if (strides[i] == strides[prev_used_iter] * prev_used_iter_extent) {
          fusible++;
          prev_used_iter = i;
        } else {
          break;
        }
      }
    }
    max_fusible = std::min(max_fusible, fusible);
  }

  // Calculate how many loops are rewritable, i.e. valid for vectorization and parallelization.
  int max_rw_loops = CalculateNumRewritableLoops(loop_srefs, loop_types);

  // Calculate the parallelize extent
  if (parsed->max_parallel_extent != -1) {
    int max_extent = parsed->max_parallel_extent;
    int& num_fusible = parsed->num_parallel_loops = 0;
    int64_t prod_extent = 1;
    for (int i = 0; i < n_loops && loop_types[i] == IterVarType::kDataPar; ++i) {
      const StmtSRef& loop_sref = loop_srefs[i];
      const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
      if (HasAnnOrBinding(loop)) {
        break;
      }
      // Check if the loop extent is valid
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (extent == nullptr) {
        break;
      }
      // Then we can fuse it in
      ++num_fusible;
      // Check if we need to break
      prod_extent *= *extent;
      if (prod_extent > max_extent || !IsSingleStmt(loop->body)) {
        break;
      }
    }
    if (prod_extent == 1) {
      num_fusible = -1;
    }
  }
  // Calculate the vectorize extent
  if (parsed->max_vectorize_extent != -1) {
    int max_extent = parsed->max_vectorize_extent;
    int& num_fusible = parsed->num_vectorize_loops = 0;
    int64_t prod_extent = 1;
    for (int i = n_loops - 1;
         i >= 0 && loop_types[i] == IterVarType::kDataPar && num_fusible < max_fusible; --i) {
      const StmtSRef& loop_sref = loop_srefs[i];
      const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
      if (HasAnnOrBinding(loop)) {
        break;
      }
      // Cannot vectorize reduce axis
      if (GetLoopIterType(loop_sref) != IterVarType::kDataPar) {
        break;
      }
      // Cannot fuse with a loop with multiple children
      if (!IsSingleStmt(loop->body)) {
        break;
      }
      // Check if the loop extent is valid
      const int64_t* extent = GetLoopIntExtent(loop_sref);
      if (extent == nullptr) {
        break;
      }
      // Check if the extent is still in a good range
      prod_extent *= *extent;
      if (prod_extent > max_extent) {
        break;
      }
      ++num_fusible;
    }
    if (prod_extent == 1) {
      num_fusible = -1;
    }
  }

  if (parsed->num_parallel_loops != -1 && parsed->num_vectorize_loops != -1) {
    if (max_rw_loops == n_loops && max_fusible == n_loops) {
      // All loops can be fused, parallelized and vectorized
      parsed->num_parallel_loops = n_loops;
      parsed->num_vectorize_loops = n_loops;
    } else {
      // Prefer num_vectorize to num_parallel
      parsed->num_parallel_loops =
          std::min(parsed->num_parallel_loops, n_loops - parsed->num_vectorize_loops);
    }
  }
}

bool FindAnnotatedRootBlock(const Schedule& sch, ParsedAnnotation* parsed, BlockRV* root_rv) {
  IRModule mod = sch->mod();
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* prim_func = base_func.as<PrimFuncNode>()) {
      const BlockRealizeNode* block_realize = prim_func->body.as<BlockRealizeNode>();
      if (block_realize != nullptr) {
        Block block = block_realize->block;
        if (ParseAnnotation(block, parsed)) {
          *root_rv = sch->GetBlock(block->name_hint, g_var->name_hint);
          RemoveParsedAnn(sch, *root_rv, *parsed);
          return true;
        }
      }
    }
  }
  return false;
}

void RewriteFuseSplitParallelVectorize(const Schedule& sch, Array<LoopRV>* loop_rvs, int vec_len) {
  size_t n_loops = loop_rvs->size();
  LoopRV fused = sch->Fuse({loop_rvs->begin(), loop_rvs->end()});
  Array<LoopRV> split = sch->Split(fused, {NullOpt, Integer(vec_len)});
  ICHECK_EQ(split.size(), 2);
  const LoopRV& outer = split[0];
  const LoopRV& inner = split[1];
  sch->Parallel(outer);
  sch->Vectorize(inner);
  for (size_t i = 0; i < n_loops - 1; ++i) {
    loop_rvs->Set(i, outer);
  }
  loop_rvs->Set(n_loops - 1, inner);
}

void RewriteParallel(const Schedule& sch, size_t n, Array<LoopRV>* loop_rvs) {
  ICHECK_LE(n, loop_rvs->size());
  LoopRV fused = sch->Fuse({loop_rvs->begin(), loop_rvs->begin() + n});
  sch->Parallel(fused);
  for (size_t i = 0; i < n; ++i) {
    loop_rvs->Set(i, fused);
  }
}

void RewriteVectorize(const Schedule& sch, size_t n, Array<LoopRV>* loop_rvs) {
  size_t n_loops = loop_rvs->size();
  ICHECK_LE(n, n_loops);
  LoopRV fused = sch->Fuse({loop_rvs->end() - n, loop_rvs->end()});
  sch->Vectorize(fused);
  for (size_t i = n_loops - n; i < n_loops; ++i) {
    loop_rvs->Set(i, fused);
  }
}

void RewriteUnroll(const Schedule& sch, int unroll_explicit, int max_step, const BlockRV& block,
                   const LoopRV& loop) {
  // Do not unroll for pure spatial block.
  if (max_step <= 0 || IsSpatial(sch->GetSRef(block))) {
    return;
  }

  sch->Annotate(loop, attr::pragma_auto_unroll_max_step, IntImm(DataType::Int(32), max_step));
  sch->Annotate(loop, attr::pragma_unroll_explicit, IntImm(DataType::Int(32), unroll_explicit));
}

}  // namespace tir

namespace meta_schedule {

using tir::Schedule;

class RewriteParallelVectorizeUnrollNode : public PostprocNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const Schedule& sch) final {
    tir::ParsedAnnotation parsed_root;
    tir::BlockRV root_rv{nullptr};
    while (tir::FindAnnotatedRootBlock(sch, &parsed_root, &root_rv)) {
      for (tir::BlockRV block_rv : sch->GetChildBlocks(root_rv)) {
        Array<tir::LoopRV> loop_rvs = sch->GetLoops(block_rv);
        if (loop_rvs.empty()) {
          continue;
        }
        tir::ParsedAnnotation parsed = parsed_root;
        tir::AdjustParallelVectorize(sch, block_rv, loop_rvs, &parsed);
        const int loops_num = loop_rvs.size();
        if (parsed.num_parallel_loops == loops_num && parsed.num_vectorize_loops == loops_num) {
          // Fuse, split, vectorize and parallelize
          tir::RewriteFuseSplitParallelVectorize(sch, &loop_rvs, parsed.max_vectorize_extent);
        } else {
          // Parallel
          if (parsed.num_parallel_loops > 0) {
            tir::RewriteParallel(sch, parsed.num_parallel_loops, &loop_rvs);
          }
          // Vectorize
          if (parsed.num_vectorize_loops > 0) {
            tir::RewriteVectorize(sch, parsed.num_vectorize_loops, &loop_rvs);
          }
        }
        // AutoUnroll
        if (parsed.unroll_explicit != -1 || parsed.unroll_implicit != -1) {
          ICHECK(parsed.unroll_explicit == -1 || parsed.unroll_implicit == -1);
          int unroll_explicit = parsed.unroll_explicit != -1;
          int max_step = parsed.unroll_explicit + parsed.unroll_implicit + 1;
          tir::RewriteUnroll(sch, unroll_explicit, max_step, block_rv, loop_rvs[0]);
        }
      }
    }
    return true;
  }

  Postproc Clone() const {
    ObjectPtr<RewriteParallelVectorizeUnrollNode> n =
        make_object<RewriteParallelVectorizeUnrollNode>(*this);
    return Postproc(n);
  }

  static constexpr const char* _type_key = "meta_schedule.RewriteParallelVectorizeUnroll";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteParallelVectorizeUnrollNode, PostprocNode);
};

Postproc Postproc::RewriteParallelVectorizeUnroll() {
  ObjectPtr<RewriteParallelVectorizeUnrollNode> n =
      make_object<RewriteParallelVectorizeUnrollNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteParallelVectorizeUnrollNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteParallelVectorizeUnroll")
    .set_body_typed(Postproc::RewriteParallelVectorizeUnroll);

}  // namespace meta_schedule
}  // namespace tvm
