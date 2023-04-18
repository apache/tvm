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
#include "../../runtime/thread_storage_scope.h"
#include "./memhammer_rewrite_rule.h"

namespace tvm {
namespace tir {

/*!
 * \brief Fuse consecutive loops
 * \param body the outer-most loop
 * \return the fused loop
 */
Stmt FuseNestLoops(Stmt body) {
  std::vector<const ForNode*> loops;
  while (const ForNode* loop = body.as<ForNode>()) {
    loops.push_back(loop);
    body = loop->body;
  }
  std::string suffix;
  int n = loops.size();
  for (int i = 1; i < n; i++) {
    suffix += "_" + loops[i]->loop_var->name_hint;
  }
  suffix += "_fused";
  Var fused_var = loops[0]->loop_var.copy_with_suffix(suffix);
  Map<Var, PrimExpr> subst_map;
  PrimExpr tot = fused_var;
  for (int i = n - 1; i >= 0; i--) {
    subst_map.Set(loops[i]->loop_var, floormod(tot, loops[i]->extent));
    tot = floordiv(tot, loops[i]->extent);
  }
  auto f_substitute = [&](const Var& v) -> Optional<PrimExpr> {
    return subst_map.Get(v).value_or(v);
  };
  PrimExpr fused_extent = 1;
  for (int i = 0; i < n; i++) {
    fused_extent *= loops[i]->extent;
  }
  return For(fused_var, 0, fused_extent, ForKind::kSerial,
             Substitute(std::move(body), f_substitute));
}

/*!
 * \brief a combination of split, bind, vectorize,
 *        a helper function to perform coalesced load/store
 * \param stmt the stmt to do transformation
 * \param constraints The constraints, including thread extents, vector bytes, and data bits.
 * \return The stmt after transformation
 */
Stmt SplitBindVectorize(const Stmt& stmt, const ConstraintSet& constraints) {
  const ForNode* loop = TVM_TYPE_AS(stmt, ForNode);
  int loop_extent = Downcast<Integer>(loop->extent)->value;
  int vector_bytes = constraints.vector_bytes;
  int data_bits = constraints.data_bits;
  int vector_len = std::max(1, vector_bytes * 8 / data_bits);
  int tot_threads = 1;
  // generate thread binding loops
  std::vector<int> factors{-1};
  std::vector<std::string> thread_axis;
  if (Optional<Integer> o_t = constraints.thread_extent.Get("threadIdx.z")) {
    int t = o_t.value()->value;
    tot_threads *= t;
    factors.push_back(t);
    thread_axis.push_back("threadIdx.z");
  }
  if (Optional<Integer> o_t = constraints.thread_extent.Get("threadIdx.y")) {
    int t = o_t.value()->value;
    tot_threads *= t;
    factors.push_back(t);
    thread_axis.push_back("threadIdx.y");
  }
  if (Optional<Integer> o_t = constraints.thread_extent.Get("threadIdx.x")) {
    int t = o_t.value()->value;
    tot_threads *= t;
    factors.push_back(t);
    thread_axis.push_back("threadIdx.x");
  }
  // generate vectorized loop
  factors.push_back(vector_len);
  // generate outer loop
  factors[0] = (loop_extent + tot_threads * vector_len - 1) / (tot_threads * vector_len);
  // create new loop vars
  int n = factors.size();
  std::vector<Var> new_loop_vars;
  new_loop_vars.reserve(n);
  arith::Analyzer analyzer;
  for (int i = 0; i < n; i++) {
    const PrimExpr& factor = factors[i];
    Var var = loop->loop_var.copy_with_suffix("_" + std::to_string(i));
    analyzer.Bind(var, Range::FromMinExtent(0, factor));
    new_loop_vars.push_back(var);
  }
  // substitute fused loop var with new loop vars
  PrimExpr substitute_value = 0;
  for (int i = 0; i < n; i++) {
    substitute_value *= factors[i];
    substitute_value += new_loop_vars[i];
  }
  // Construct the new loop nest
  Stmt body = Substitute(loop->body, [&](const Var& v) -> Optional<PrimExpr> {
    if (v.same_as(loop->loop_var)) {
      return substitute_value;
    } else {
      return NullOpt;
    }
  });
  PrimExpr predicate = substitute_value < loop->extent;
  if (!analyzer.CanProve(predicate)) {
    body = IfThenElse(predicate, body);
  }
  body = For(new_loop_vars.back(), 0, vector_len, ForKind::kVectorized, std::move(body));
  for (int i = n - 2; i >= 1; i--) {
    body = For(new_loop_vars[i], 0, factors[i], ForKind::kThreadBinding, std::move(body),
               IterVar(Range(nullptr), Var(thread_axis[i - 1]), kThreadIndex, thread_axis[i - 1]));
  }
  return For(new_loop_vars[0], 0, factors[0], ForKind::kSerial, std::move(body));
}

Stmt CoalescedAccess::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                              OutputSet* output) const {
  Stmt after_fuse = FuseNestLoops(stmt);
  Stmt after_split = SplitBindVectorize(std::move(after_fuse), constraints);
  return after_split;
}

/*!
 * \brief Get the index mapping of a specific stmt.
 *        The stmt is like:
 *        for i0:
 *          ...
 *          for in:
 *            A[f(i0, ..., in])] = B[i0, ..., in],
 *        where f is the index mapping we want to get.
 * \param constraints The constraints, including the write region that is required to calculate
 * the index mapping
 * \return The mapping in the form of j0, ..., jm, where j0, ... jm = f(i0, ..., in)
 */
Array<PrimExpr> GetMapping(const Stmt& stmt, const ConstraintSet& constraints) {
  Stmt body = stmt;
  while (const ForNode* loop = body.as<ForNode>()) {
    body = loop->body;
  }
  const BufferStoreNode* buf_store = TVM_TYPE_AS(body, BufferStoreNode);
  BufferRegion write_region = constraints.write_region;
  const Array<PrimExpr>& write_index = buf_store->indices;
  ICHECK(write_region->region.size() == write_index.size() &&
         write_region->buffer.same_as(buf_store->buffer));
  Array<PrimExpr> result;
  arith::Analyzer analyzer;
  for (int i = 0; i < static_cast<int>(write_region->region.size()); i++) {
    PrimExpr pattern = analyzer.Simplify(write_index[i] - write_region->region[i]->min);
    if (!is_zero(pattern)) {
      result.push_back(pattern);
    }
  }
  return result;
}

Stmt InverseMapping::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                             OutputSet* output) const {
  Stmt body = stmt;
  Map<Var, Range> var_range;
  Array<PrimExpr> loop_vars;
  // Step 1. Get index mapping
  Array<PrimExpr> mapping_pattern = GetMapping(stmt, constraints);
  while (const ForNode* loop = body.as<ForNode>()) {
    var_range.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    loop_vars.push_back(loop->loop_var);
    body = loop->body;
  }
  // Step 2. Get Inverse mapping
  arith::Analyzer analyzer;
  DiagnosticContext diag_ctx(DiagnosticContext::Default(IRModule()));
  auto iter_map =
      arith::DetectIterMap(mapping_pattern, var_range, Bool(true), arith::Bijective, &analyzer);
  CHECK_EQ(iter_map->indices.size(), loop_vars.size());
  Map<Var, PrimExpr> inverse_mapping = arith::InverseAffineIterMap(iter_map->indices, loop_vars);
  // Step 3. Generate new body
  BufferRegion read_region = constraints.read_region;
  BufferRegion write_region = constraints.write_region;
  Array<PrimExpr> write_index;
  Array<PrimExpr> read_index;
  Array<Var> new_loop_vars;
  Map<Var, PrimExpr> substitute_map;
  // Step 3.1 construct target buffer indices
  for (int i = 0, j = 0; i < static_cast<int>(write_region->region.size()); i++) {
    if (is_one(write_region->region[i]->extent)) {
      write_index.push_back(write_region->region[i]->min);
    } else {
      Var var = runtime::Downcast<Var>(loop_vars[j]).copy_with_suffix("_inverse");
      new_loop_vars.push_back(var);
      substitute_map.Set(runtime::Downcast<Var>(loop_vars[j++]), var);
      write_index.push_back(write_region->region[i]->min + var);
    }
  }
  // Step 3.2 construct source buffer indices
  for (int i = 0, j = 0; i < static_cast<int>(read_region->region.size()); i++) {
    if (is_one(read_region->region[i]->extent)) {
      read_index.push_back(read_region->region[i]->min);
    } else {
      read_index.push_back(
          read_region->region[i]->min +
          Substitute(inverse_mapping[Downcast<Var>(loop_vars[j++])], substitute_map));
    }
  }
  BufferLoad new_buf_load = BufferLoad(read_region->buffer, read_index);
  BufferStore new_buf_store = BufferStore(write_region->buffer, new_buf_load, write_index);
  Stmt ret = new_buf_store;
  // Step 3.3 construct loop body
  for (int i = static_cast<int>(new_loop_vars.size()) - 1; i >= 0; i--) {
    PrimExpr extent = write_region->region[i]->extent;
    ret = For(new_loop_vars[i], 0, extent, ForKind::kSerial, std::move(ret));
  }
  return ret;
}
}  // namespace tir
}  // namespace tvm
