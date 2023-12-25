/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_tile_op.cc
 * \brief Lower the tile op for further codegen.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../arith/ir_mutator_with_analyzer.h"
#include "arith.h"
#include "helper.h"
#include "layout.h"
#include "loop_partition.h"
#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class LowerTileOpPass : arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LowerTileOpPass substituter(&analyzer);
    for (const auto& [_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode* op) final {
    Map<Var, Layout> vmap;
    if (op->annotations.count(attr::kLayoutMap)) {
      vmap = op->annotations.at(attr::kLayoutMap).as<Map<Var, Layout>>().value();
    }
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
      if (vmap.count(buffer->data)) layout_map_.Set(buffer, vmap[buffer->data]);
    }
    auto ret = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    auto block = ret.as<Block>().value();
    if (!workspaces_.empty()) {
      auto block_ptr = block.CopyOnWrite();
      for (const auto& buffer : workspaces_) block_ptr->alloc_buffers.push_back(buffer);
      workspaces_.clear();
    }
    return block;
  }

  Stmt VisitStmt_(const EvaluateNode* node) final {
    if (auto call = node->value.as<CallNode>()) {
      if (call->op.same_as(tl::gemm())) {
        return LowerGemm(call->args);
      } else if (call->op.same_as(tl::reduce())) {
        return LowerReduce(call->args);
      }
    }
    return GetRef<Evaluate>(node);
  }

  PrimExpr GetWorkspace(int num_elem, DataType dtype) {
    // TODO: fix merge dyn shared memory pass
    auto workspace = decl_buffer({PrimExpr(num_elem)}, dtype, "workspace", "shared.dyn");
    workspaces_.push_back(workspace);
    return workspace.access_ptr(2);  // write
  }

  Stmt LowerReduce(const Array<PrimExpr>& call_args) {
    ReduceArgs args = ReduceArgs::Parse(call_args, buffer_data_to_buffer_);
    ICHECK(args.src.scope() == "local" && args.dst.scope() == "local")
        << "Reduce for shared memory not implemented.";
    Fragment src = layout_map_[args.src].as<Fragment>().value();
    Fragment dst = layout_map_[args.dst].as<Fragment>().value();
    ICHECK(src->InputDim() == dst->InputDim() + 1);
    Array<IterVar> dst_vars;
    for (size_t i = 0; i < dst->InputDim(); i++) {
      Var var = Var(std::string{char('i' + i)});
      dst_vars.push_back(IterVar(Range(0, dst->InputShape()[i]), var, IterVarType::kDataPar));
    }
    Array<IterVar> src_vars = dst_vars;
    src_vars.insert(src_vars.begin() + args.dim,
                    {Range(0, src->InputShape()[args.dim]), Var("rv"), IterVarType::kDataPar});
    Array<PrimExpr> src_indices =
        src->Forward(src_vars.Map([](const auto& iv) { return PrimExpr(iv->var); }));
    Array<PrimExpr> dst_indices =
        dst->Forward(dst_vars.Map([](const auto& iv) { return PrimExpr(iv->var); }));

    Array<Stmt> stmts;

    // make reduce-init stmt
    if (args.clear) stmts.push_back(BufferStore(args.dst, args.MakeInitValue(), dst_indices));

    // make thread-local reduce
    Array<PrimExpr> src_indice_compressed;
    Array<IterVar> src_var_compressed;
    for (size_t i = 0; i < src->OutputDim(); i++) {
      PrimExpr expr;
      IterVar var;
      std::tie(expr, var) =
          CompressIterator(src_indices[i], src_vars, src_vars[args.dim], analyzer_);
      src_indice_compressed.push_back(expr);
      src_var_compressed.push_back(var);
    }
    Stmt reduce_local = BufferStore(args.dst,
                                    args.MakeReduce(BufferLoad(args.dst, dst_indices),
                                                    BufferLoad(args.src, src_indice_compressed)),
                                    dst_indices);
    for (int i = src->OutputDim() - 1; i >= 0; i--) {
      reduce_local =
          For(src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent, ForKind::kUnrolled,
              reduce_local, NullOpt, {{tir::attr::pragma_unroll_explicit, Bool(false)}});
    }
    stmts.push_back(reduce_local);

    // make inter-thread reduce
    PrimExpr src_thread =
        src->ForwardThread(src_vars.Map([](const auto& iv) { return PrimExpr(iv->var); }), {});
    auto iter_sum = arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer_);
    for (const auto& iter_split : iter_sum->args) {
      auto mark = iter_split->source->source.as<Var>();
      ICHECK(mark.defined());
      if (mark.value().same_as(src_vars[args.dim]->var)) {
        auto scale = as_const_int(iter_split->scale);
        auto extent = as_const_int(iter_split->extent);
        ICHECK(scale != nullptr && extent != nullptr);
        if (*extent == 1) continue;
        int reducing_threads = (*extent) * (*scale);
        std::stringstream ss;
        ss << "tl::AllReduce<" << args.MakeCodegenReducer() << ", " << reducing_threads << ", "
           << (*scale) << ">::run";
        Array<PrimExpr> thread_reduce_args = {StringImm(ss.str()),
                                              BufferLoad(args.dst, dst_indices)};
        if (reducing_threads >= 32) {
          PrimExpr workspace = GetWorkspace(thread_block_size_, args.dst->dtype);
          thread_reduce_args.push_back(workspace);
        }
        auto call = Call(args.dst->dtype, builtin::call_extern(), thread_reduce_args);
        stmts.push_back(BufferStore(args.dst, call, dst_indices));
      }
    }
    Stmt reduce_interthread = BufferStore(args.dst, BufferLoad(args.dst, dst_indices), dst_indices);

    // make the outer spatial loop
    Stmt body = SeqStmt(stmts);
    for (int i = dst->InputDim() - 1; i >= 0; i--) {
      body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent, ForKind::kParallel, body);
    }

    body = PartitionLoop(body.as<ForNode>(), thread_var_, analyzer_, dst);
    return body;
  }

  Stmt LowerGemm(const Array<PrimExpr>& call_args) {
    GemmArgs args = GemmArgs::Parse(call_args, buffer_data_to_buffer_);
    ICHECK(thread_block_size_ % 32 == 0);
    auto [warp_m, warp_n] = args.ComputeWarpPartition(thread_block_size_ / 32);
    std::stringstream ss;
    std::string op_name = "tl::gemm_ss";
    if (args.A.scope() == "local") {
      ICHECK(args.B.scope() != "local");
      op_name = "tl::gemm_rs";
    } else if (args.B.scope() == "local") {
      op_name = "tl::gemm_sr";
    }
    ss << op_name << "<" << args.M << ", " << args.N << ", " << args.K << ", ";
    ss << warp_m << ", " << warp_n << ", ";
    ss << args.trans_A << ", " << args.trans_B << ">";

    Array<PrimExpr> new_args;
    new_args.push_back(StringImm(ss.str()));
    new_args.push_back(call_args[0]);
    new_args.push_back(call_args[1]);
    new_args.push_back(call_args[2]);
    auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
    return Evaluate(new_call);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv->var;
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_block_size_ = iv->dom->extent.as<IntImmNode>()->value;
      }
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Layout> layout_map_;
  Var thread_var_;
  size_t thread_block_size_ = 0;
  Array<Buffer> workspaces_;
};

namespace transform {

using namespace tir::transform;

tvm::transform::Pass LowerTileOp() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerTileOpPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerTileOp", {});
}

TVM_REGISTER_GLOBAL("tl.LowerTileOp").set_body_typed(LowerTileOp);
}  // namespace transform

}  // namespace tl
}  // namespace tvm
