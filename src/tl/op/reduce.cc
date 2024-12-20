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
 * \file tl/op/reduce.cc
 *
 * Define reduce operator.
 */

#include "reduce.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../layout/utils.h"
#include "../transform/loop_partition.h"

namespace tvm {
namespace tl {

using namespace tir;

ReduceOp::ReduceOp(Array<PrimExpr> args, BufferMap vmap) {
  src = vmap[GetVarFromAccessPtr(args[0])];
  dst = vmap[GetVarFromAccessPtr(args[1])];
  String reduce_type = args[2].as<StringImm>().value()->value;
  dim = args[3].as<IntImm>().value()->value;
  if (reduce_type == "sum")
    type = ReduceType::kSum;
  else if (reduce_type == "abssum")
    type = ReduceType::kAbsSum;
  else if (reduce_type == "max")
    type = ReduceType::kMax;
  else if (reduce_type == "min")
    type = ReduceType::kMin;
  else
    ICHECK(0) << "Unknown reduce type: " << reduce_type;
  clear = args[4].as<Bool>().value();
}

PrimExpr ReduceOp::MakeInitValue() const {
  switch (type) {
    case ReduceType::kSum:
      return make_zero(dst->dtype);
    case ReduceType::kAbsSum:
      return make_zero(dst->dtype);
    case ReduceType::kMax:
      return make_const(dst->dtype, -INFINITY);
    case ReduceType::kMin:
      return make_const(dst->dtype, INFINITY);
    default:
      ICHECK(0);
  }
}

PrimExpr ReduceOp::MakeReduce(const PrimExpr& a, const PrimExpr& b) const {
  PrimExpr lhs = a, rhs = b;
  if (lhs->dtype != rhs->dtype) {
    rhs = Cast(lhs->dtype, rhs);
  }
  switch (type) {
    case ReduceType::kSum:
      return lhs + rhs;
    case ReduceType::kAbsSum:
      return lhs + Max(rhs, -rhs);
    case ReduceType::kMax:
      return Max(lhs, rhs);
    case ReduceType::kMin:
      return Min(lhs, rhs);
    default:
      ICHECK(0);
      return PrimExpr(0);
  }
}

std::string ReduceOp::MakeCodegenReducer() const {
  switch (type) {
    case ReduceType::kSum:
      return "tl::SumOp";
    case ReduceType::kAbsSum:
      return "tl::SumOp";
    case ReduceType::kMax:
      return "tl::MaxOp";
    case ReduceType::kMin:
      return "tl::MinOp";
    default:
      ICHECK(0);
      return "";
  }
}

Stmt ReduceOp::Lower(const LowerArgs& T, arith::Analyzer* analyzer) const {
  ICHECK(this->src.scope() == "local.fragment" && this->dst.scope() == "local.fragment")
      << "Reduce for shared memory not implemented.";
  auto src_buffer = T.buffer_remap[this->src];
  auto dst_buffer = T.buffer_remap[this->dst];
  Fragment src_layout = T.layout_map[this->src].as<Fragment>().value();
  Fragment dst_layout = T.layout_map[this->dst].as<Fragment>().value();
  ICHECK(src_layout->InputDim() == dst_layout->InputDim() + 1);
  Array<IterVar> dst_vars;
  for (size_t i = 0; i < dst_layout->InputDim(); i++) {
    Var var = Var(std::string{char('i' + i)});
    dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var, IterVarType::kDataPar));
  }
  Array<IterVar> src_vars = dst_vars;
  src_vars.insert(src_vars.begin() + this->dim, {Range(0, src_layout->InputShape()[this->dim]),
                                                 Var("rv"), IterVarType::kDataPar});
  Array<PrimExpr> src_indices =
      src_layout->Forward(src_vars.Map([](const auto& iv) { return PrimExpr(iv->var); }));
  Array<PrimExpr> dst_indices =
      dst_layout->Forward(dst_vars.Map([](const auto& iv) { return PrimExpr(iv->var); }));

  Array<Stmt> stmts;

  // make reduce-init stmt
  if (this->clear) stmts.push_back(BufferStore(dst_buffer, this->MakeInitValue(), dst_indices));

  // make thread-local reduce
  Array<PrimExpr> src_indice_compressed;
  Array<IterVar> src_var_compressed;
  for (size_t i = 0; i < src_layout->OutputDim(); i++) {
    PrimExpr expr;
    IterVar var;
    std::tie(expr, var) =
        CompressIterator(src_indices[i], src_vars, src_vars[this->dim]->var, analyzer);
    src_indice_compressed.push_back(expr);
    src_var_compressed.push_back(var);
  }
  Stmt reduce_local = BufferStore(dst_buffer,
                                  this->MakeReduce(BufferLoad(dst_buffer, dst_indices),
                                                   BufferLoad(src_buffer, src_indice_compressed)),
                                  dst_indices);
  for (int i = src_layout->OutputDim() - 1; i >= 0; i--) {
    reduce_local =
        For(src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent, ForKind::kUnrolled,
            reduce_local, NullOpt, {{tir::attr::pragma_unroll_explicit, Bool(false)}});
  }
  stmts.push_back(reduce_local);

  // make inter-thread reduce
  PrimExpr src_thread =
      src_layout->ForwardThread(src_vars.Map([](const auto& iv) { return PrimExpr(iv->var); }), {});
  auto iter_sum = arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);
  for (const auto& iter_split : iter_sum->args) {
    auto mark = iter_split->source->source.as<Var>();
    ICHECK(mark.defined());
    if (mark.value().same_as(src_vars[this->dim]->var)) {
      auto scale = as_const_int(iter_split->scale);
      auto extent = as_const_int(iter_split->extent);
      ICHECK(scale != nullptr && extent != nullptr);
      if (*extent == 1) continue;
      int reducing_threads = (*extent) * (*scale);
      std::stringstream ss;
      ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", " << reducing_threads << ", "
         << (*scale) << ">::run";
      Array<PrimExpr> thread_reduce_args = {StringImm(ss.str()),
                                            BufferLoad(dst_buffer, dst_indices)};
      if (reducing_threads >= 32) {
        PrimExpr workspace = T.AddWorkspace(T.block_size, dst_buffer->dtype);
        thread_reduce_args.push_back(workspace);
      }
      auto call = Call(dst_buffer->dtype, builtin::call_extern(), thread_reduce_args);
      stmts.push_back(BufferStore(dst_buffer, call, dst_indices));
    }
  }
  Stmt reduce_interthread =
      BufferStore(dst_buffer, BufferLoad(dst_buffer, dst_indices), dst_indices);

  // make the outer spatial loop
  Stmt body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
  for (int i = dst_layout->InputDim() - 1; i >= 0; i--) {
    body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent, ForKind::kParallel, body);
  }

  body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer, dst_layout);
  return body;
}

LayoutMap ReduceOp::InferLayout(const LayoutInferArgs& T, InferLevel level) {
  if (level >= InferLevel::kStrict) return {};
  if (src.scope() == "local.fragment" && dst.scope() == "local.fragment" &&
      T.layout_map.count(src) && !T.layout_map.count(dst)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();

    PrimExpr indice_rep_extent = src->shape[dim];
    PrimExpr src_rep_extent = src_layout->ReplicateExtent();
    PrimExpr dest_buffer_rep_extent = indice_rep_extent * src_rep_extent;

    Array<PrimExpr> fwd;
    for (int i = 0; i < static_cast<int>(src->shape.size()); i++) {
      if (i == dim) {
        fwd.push_back(FloorMod(ReplicationPlaceholder(), indice_rep_extent));
      } else if (i < dim) {
        fwd.push_back(InputPlaceholder(i));
      } else if (i > dim) {
        fwd.push_back(InputPlaceholder(i - 1));
      }
    }
    auto thd =
        src_layout->ForwardThread(fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));
    Fragment dst_layout =
        Fragment(dst->shape, {}, thd, dest_buffer_rep_extent, NullOpt)->CondenseReplicateVar();
    return {{dst, dst_layout}};
  }
  return {};
}

TIR_REGISTER_TL_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace tl
}  // namespace tvm