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
 * \file src/relax/transform/split_tir_layout_rewrite.cc
 * \brief Use for rewriting the TIRs after meta_schedule layout rewrite post process.
 */
#include <tvm/ir/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cstddef>

namespace tvm {
namespace tir {
class SplitPrimFuncLayoutRewrite : public StmtMutator {
 public:
  explicit SplitPrimFuncLayoutRewrite(const PrimFunc& func) : original_func_(func) {}
  std::tuple<Optional<PrimFunc>, PrimFunc> Transform(const PrimFunc& func) {
    ICHECK(func->body.as<BlockRealizeNode>()) << "The body of the primfunc should be a root block.";
    const auto& block = func->body.as<BlockRealizeNode>()->block;
    visit_root_block(block.get());
    if (layout_rewrite_preproc_stmts_.size() > 0) {
      return std::make_tuple(create_layout_rewrite_preproc_func(), create_compute_func());
    } else {
      return std::make_tuple(NullOpt, func);
    }
  }

 private:
  void sort_rewrite_infos() {
    std::sort(
        rewrite_infos_.begin(), rewrite_infos_.end(),
        [](const RewriteInfo& a, const RewriteInfo& b) { return a.buffer_index < b.buffer_index; });
  }

  PrimFunc create_layout_rewrite_preproc_func() const {
    // Step 1: Check the number of pre_rewrite_buffers and post_rewrite_buffers
    ICHECK(rewrite_infos_.size() > 0) << "There should be at least one buffer rewrite.";

    // Step 2: Create the params for the new PrimFunc
    Array<Var> params;
    Map<Var, Buffer> buffer_map;

    for (const auto& info : rewrite_infos_) {
      params.push_back(Var(info.pre_rewrite_buffer->name, DataType::Handle()));
      buffer_map.Set(params.back(), info.pre_rewrite_buffer);
    }
    for (const auto& info : rewrite_infos_) {
      params.push_back(Var(info.post_rewrite_buffer->name, DataType::Handle()));
      buffer_map.Set(params.back(), info.post_rewrite_buffer);
    }

    // Step 3: Create the body for the new PrimFunc
    ICHECK(layout_rewrite_preproc_stmts_.size() > 0)
        << "There should be at least one layout rewrite preproc stmt.";
    Stmt body = layout_rewrite_preproc_stmts_.size() == 1 ? layout_rewrite_preproc_stmts_[0]
                                                          : SeqStmt(layout_rewrite_preproc_stmts_);
    body = BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/"root", body));

    PrimFunc func = PrimFunc(params, body, VoidType(), buffer_map);

    return RenewDefs(func);
  }

  PrimFunc create_compute_func() const {
    // Step 1: Create the params for the new PrimFunc
    Array<Var> params = original_func_->params;
    Map<Var, Buffer> buffer_map = original_func_->buffer_map;
    for (const auto& info : rewrite_infos_) {
      const Var& param = params[info.buffer_index];
      ICHECK(buffer_map[param] == info.pre_rewrite_buffer);
      buffer_map.Set(param, info.post_rewrite_buffer);
    }

    // Step 2: Create the body for the new PrimFunc
    Stmt body = compute_stmts_.size() == 1 ? compute_stmts_[0] : SeqStmt(compute_stmts_);
    Block original_block = original_func_->body.as<BlockRealizeNode>()->block;
    Array<Buffer> alloc_buffers;
    for (const auto& buffer : original_block->alloc_buffers) {
      auto it =
          std::find_if(rewrite_infos_.begin(), rewrite_infos_.end(),
                       [&](const RewriteInfo& info) { return info.post_rewrite_buffer == buffer; });
      if (it == rewrite_infos_.end()) {
        alloc_buffers.push_back(buffer);
      }
    }

    body = BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/"root", body,
              /*init=*/NullOpt,
              /*alloc_buffers=*/alloc_buffers));

    PrimFunc func = PrimFunc(original_func_->params, body, VoidType(), buffer_map);
    return RenewDefs(func);
  }

  void visit_root_block(const BlockNode* op) {
    Stmt body = op->body;
    if (const auto* seq_stmt = body.as<SeqStmtNode>()) {
      for (const auto& stmt : seq_stmt->seq) {
        current_subtree_ = 0;
        Stmt new_stmt = this->VisitStmt(stmt);
        ICHECK(current_subtree_ != 0) << "There should be at least a block in the subtree.";
        if (current_subtree_ == 1) {
          layout_rewrite_preproc_stmts_.push_back(new_stmt);
        } else {
          compute_stmts_.push_back(new_stmt);
        }
      }
    } else {
      current_subtree_ = 0;
      this->VisitStmt(body);
      ICHECK(current_subtree_ == -1)
          << "There should be a compute block if there is only one subtree under the root.";
    }
  }
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
    auto it = op->annotations.find(attr::meta_schedule_layout_rewrite_preproc);
    bool is_layout_rewrite_preproc =
        it != op->annotations.end() && is_one(Downcast<PrimExpr>((*it).second));

    if (current_subtree_ == 0) {
      current_subtree_ = is_layout_rewrite_preproc ? 1 : -1;
    } else if (current_subtree_ == 1) {
      CHECK(is_layout_rewrite_preproc)
          << "There is a layout rewrite block in the subtree, but meet a non-layout rewrite block.";
    } else {
      CHECK(!is_layout_rewrite_preproc)
          << "There is a non-layout rewrite block in the subtree, but meet a layout rewrite block.";
    }

    if (is_layout_rewrite_preproc) {
      ICHECK(op->reads.size() == 1) << "There should be only one read buffer in the layout rewrite";
      ICHECK(op->writes.size() == 1)
          << "There should be only one write buffer in the layout rewrite";
      ICHECK(op->alloc_buffers.empty()) << "There should be no alloc buffer in the layout rewrite";
      ICHECK(op->match_buffers.empty()) << "There should be no match buffer in the layout rewrite";
      const Buffer& preproc_buffer = op->reads[0]->buffer;
      int buffer_index = -1;
      for (size_t i = 0; i < original_func_->params.size(); ++i) {
        const Buffer& buffer = original_func_->buffer_map[original_func_->params[i]];
        if (buffer == preproc_buffer) {
          buffer_index = i;
          break;
        }
      }
      ICHECK(buffer_index != -1) << "The preproc buffer is not found in the original primfunc.";
      rewrite_infos_.push_back(
          RewriteInfo{buffer_index, op->reads[0]->buffer, op->writes[0]->buffer});

      auto new_annotations = op->annotations;
      new_annotations.erase(attr::meta_schedule_layout_rewrite_preproc);
      auto n = make_object<BlockNode>(*block.get());
      n->annotations = new_annotations;
      return Block(n);
    }
    return block;
  }

 public:
  struct RewriteInfo {
    int buffer_index;
    Buffer pre_rewrite_buffer;
    Buffer post_rewrite_buffer;
  };
  std::vector<RewriteInfo> rewrite_infos_;

 private:
  /*! \brief The stmts that are used for layout rewrite preproc*/
  Array<Stmt> layout_rewrite_preproc_stmts_;
  /*! \brief The stmts that are other than layout rewrite preproc*/
  Array<Stmt> compute_stmts_;
  /*!
   \brief Whether the current subtree is a layout rewrite preproc subtree.
          -1: visited a non-layout rewrite preproc block
           0: unsure, not visited any block
           1: visited a layout rewrite preproc block
  */
  int current_subtree_;
  /*! \brief The original primfunc*/
  PrimFunc original_func_;
};
}  // namespace tir

namespace relax {
class SplitLayoutRewritePreproc : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    SplitLayoutRewritePreproc mutator(mod);

    // Step 1: Split the primfunc into preproc and compute
    for (auto [gv, func] : mod->functions) {
      if (func->IsInstance<tir::PrimFuncNode>()) {
        tir::SplitPrimFuncLayoutRewrite tir_rewriter(Downcast<tir::PrimFunc>(func));
        auto [preproc_func, compute_func] = tir_rewriter.Transform(Downcast<tir::PrimFunc>(func));
        if (preproc_func.defined()) {
          mutator.split_funcs_.emplace(gv.get(),
                                       std::make_tuple(preproc_func.value(), compute_func));
          mutator.rewrite_infos_.emplace(gv.get(), tir_rewriter.rewrite_infos_);
        }
      }
    }

    for (auto [gv, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionNode>()) {
        auto relax_func = Downcast<relax::Function>(func);
        mutator.builder_->UpdateFunction(gv, Downcast<relax::Function>(mutator(relax_func)));
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit SplitLayoutRewritePreproc(const IRModule& mod) : ExprMutator(mod) {}
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* op) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(op));

    // Step 1: Skip call to other than `tir.call_tir`
    if (!call->op.same_as(call_tir_op)) {
      return call;
    }

    // Step 2: Skip if there is no preproc stage
    const GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    auto it = split_funcs_.find(gv.get());
    if (it == split_funcs_.end()) {
      return call;
    }

    // Step 3: Get the preproc and compute functions and update the module
    const auto& [preproc_func, compute_func] = it->second;
    GlobalVar preproc_gv = builder_->AddFunction(preproc_func, gv->name_hint + "_weight_prepack");
    GlobalVar compute_gv = builder_->AddFunction(compute_func, gv->name_hint + "_prepacked");
    // Step 4. Get rewrite infos
    auto rewrite_infos_it = rewrite_infos_.find(gv.get());
    ICHECK(rewrite_infos_it != rewrite_infos_.end())
        << "Rewrite infos are not found for " << gv->name_hint;
    const auto& rewrite_infos = rewrite_infos_it->second;

    // Step 5: Emit the preproc call
    Array<Expr> call_tir_args = Downcast<Tuple>(call->args[1])->fields;
    Array<Expr> preproc_args;
    Array<StructInfo> preproc_sinfo_list;
    for (const auto& info : rewrite_infos) {
      preproc_args.push_back(call_tir_args[info.buffer_index]);
      tir::Buffer rewritten_buffer = info.post_rewrite_buffer;
      for (const auto& shape_expr : rewritten_buffer->shape) {
        CHECK(shape_expr.as<tir::IntImmNode>()) << "Currently does not support rewrite buffer with "
                                                   "dynamic shape.";
      }
      preproc_sinfo_list.push_back(
          TensorStructInfo(ShapeExpr(rewritten_buffer->shape), rewritten_buffer->dtype));
    }
    StructInfo preproc_sinfo = preproc_sinfo_list.size() > 1              //
                                   ? TupleStructInfo(preproc_sinfo_list)  //
                                   : preproc_sinfo_list[0];

    // Step 6: Call the preproc function
    Expr preproc_call =
        builder_->Emit(Call(call_tir_op, {preproc_gv, Tuple(preproc_args)}, {}, {preproc_sinfo}));
    if (rewrite_infos.size() == 1) {
      call_tir_args.Set(rewrite_infos[0].buffer_index, preproc_call);
    } else {
      for (size_t i = 0; i < rewrite_infos.size(); ++i) {
        call_tir_args.Set(rewrite_infos[i].buffer_index, TupleGetItem(preproc_call, i));
      }
    }
    Expr main_call =
        builder_->Emit(Call(call_tir_op, {compute_gv, Tuple(call_tir_args)}, {}, call->sinfo_args));

    return main_call;
  }

 private:
  std::unordered_map<const GlobalVarNode*, std::tuple<tir::PrimFunc, tir::PrimFunc>> split_funcs_;
  std::unordered_map<const GlobalVarNode*,
                     std::vector<tir::SplitPrimFuncLayoutRewrite::RewriteInfo>>
      rewrite_infos_;
};

}  // namespace relax

namespace transform {
Pass SplitLayoutRewritePreproc() {
  auto pass_func = [](IRModule mod, PassContext pc) {
    return relax::SplitLayoutRewritePreproc::Transform(mod);
  };
  auto pass = CreateModulePass(pass_func, 0, "SplitLayoutRewritePreproc", {});
  return tvm::transform::Sequential({pass, relax::transform::DeadCodeElimination()},
                                    "SplitLayoutRewritePreproc");
}
TVM_REGISTER_GLOBAL("relax.transform.SplitLayoutRewritePreproc")
    .set_body_typed(SplitLayoutRewritePreproc);
}  // namespace transform
}  // namespace tvm
