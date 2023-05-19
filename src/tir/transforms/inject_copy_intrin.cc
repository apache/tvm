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
 * \brief Replace certain copy with copy intrinsics.
 * \file copy_intrin_rewrite.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/pattern_match.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::PackedFunc;

class CopyIntrinInjector : public StmtMutator {
 public:
  CopyIntrinInjector(const std::string& pragma_key, const PackedFunc& flower_copy_fromto)
      : pragma_key_(attr::pragma_scope_prefix + pragma_key),
        flower_copy_fromto_(flower_copy_fromto) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == pragma_key_) {
      Stmt ret;
      std::string error_info;
      ICHECK(MatchCopyPattern(op->body, &ret, &error_info))
          << "Cannot match copy pattern. The error is " << error_info << " The body is "
          << op->body;
      return ret;
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  bool MatchCopyPattern(Stmt stmt, Stmt* out, std::string* error_info) {
    using namespace arith;
    Stmt body = stmt;

    // strip the loops
    std::vector<const ForNode*> loops;
    while (const ForNode* op = body.as<ForNode>()) {
      if (!is_zero(op->min)) {
        *error_info = "the 'min' value of body 'Fonode' is 0.";
        return false;
      }
      loops.push_back(op);
      body = op->body;
    }
    auto store = body.as<BufferStoreNode>();
    if (store == nullptr) {
      *error_info = "the body is not a 'BufferStoreNode'";
      return false;
    }
    // Expr sel_cond, sel_true_value, sel_false_value;
    // match select or if
    PVar<PrimExpr> sel_cond, sel_true_value, sel_false_value;
    bool has_cond = if_then_else(sel_cond, sel_true_value, sel_false_value).Match(store->value) ||
                    select(sel_cond, sel_true_value, sel_false_value).Match(store->value);

    const CastNode* cast = store->value.as<CastNode>();
    auto load = store->value.as<BufferLoadNode>();
    if (0 == loops.size()) {
      ICHECK(!has_cond);
    }
    // for now only support true condition matching
    if (has_cond) {
      load = sel_true_value.Eval().as<BufferLoadNode>();
    }
    // cast can be part of the pattern
    if (cast != nullptr) {
      load = cast->value.as<BufferLoadNode>();
    }
    if (load == nullptr) {
      *error_info = "the 'BufferLoadNode' of body is a nullptr.";
      return false;
    }
    if (load->dtype.lanes() != 1) return false;
    Array<Var> loop_vars;
    for (const ForNode* op : loops) {
      loop_vars.push_back(op->loop_var);
    }
    // TODO(Lunderberg): Move this pass to be before
    // StorageFlatten/FlattenBuffer.  That will simplify the
    // implementation, since the pre-flattened indices/strides can be
    // used directly.
    ICHECK((store->indices.size() == 1) && (load->indices.size() == 1))
        << "InjectDoubleBuffer expects flat 1-d buffers.  "
        << "Has StorageFlatten (TE-based schedules) or "
        << "FlattenBuffer (TIR-based schedules) been run?";

    Array<PrimExpr> store_strides = arith::DetectLinearEquation(store->indices[0], loop_vars);
    Array<PrimExpr> load_strides = arith::DetectLinearEquation(load->indices[0], loop_vars);
    if (load_strides.size() == 0 || store_strides.size() == 0) return false;
    Array<PrimExpr> dst_shape;
    const size_t loop_var_size = loop_vars.size();
    if (loop_var_size == 0) {
      dst_shape.push_back(make_const(DataType::Int(32), 1));
    } else {
      for (const ForNode* op : loops) {
        dst_shape.push_back(op->extent);
      }
    }
    Array<PrimExpr> src_shape = dst_shape;
    Array<PrimExpr> pad_before, pad_after;
    PrimExpr pad_value;
    PrimExpr src_elem_offset = load_strides[loop_var_size];
    if (has_cond) {
      Array<PrimExpr> clip_bound = arith::DetectClipBound(sel_cond.Eval(), loop_vars);
      pad_value = sel_false_value.Eval();
      if (clip_bound.size() == 0) {
        *error_info = "the size of clip bound is 0.";
        return false;
      }
      ICHECK_EQ(src_shape.size(), loop_vars.size());
      ICHECK_EQ(clip_bound.size(), loop_vars.size() * 2);
      for (size_t i = 0; i < src_shape.size(); ++i) {
        PrimExpr min_value = clip_bound[2 * i];
        PrimExpr max_value = clip_bound[2 * i + 1];
        DataType t = loop_vars[i].dtype();
        PrimExpr svalue = src_shape[i];
        if (min_value.defined()) {
          PrimExpr pbefore = analyzer_.Simplify(Max(min_value, make_zero(t)));
          src_elem_offset = src_elem_offset + pbefore * load_strides[i];
          svalue = svalue - pbefore;
          pad_before.push_back(pbefore);
        } else {
          pad_before.push_back(make_zero(t));
        }
        if (max_value.defined()) {
          PrimExpr pafter = analyzer_.Simplify(
              max(loops[i]->extent - max_value - make_const(t, 1), make_zero(t)));
          svalue = svalue - pafter;
          pad_after.push_back(pafter);
        } else {
          pad_after.push_back(make_zero(t));
        }
        src_shape.Set(i, analyzer_.Simplify(svalue));
      }
      src_elem_offset = analyzer_.Simplify(src_elem_offset);
    }
    ICHECK_EQ(load_strides.size(), store_strides.size());
    ICHECK_EQ(load_strides.size(), loop_var_size + 1);
    Array<PrimExpr> src_strides(load_strides.begin(), load_strides.begin() + loop_var_size);
    Array<PrimExpr> dst_strides(store_strides.begin(), store_strides.begin() + loop_var_size);
    if (loop_var_size == 0) {
      src_strides.push_back(make_const(DataType::Int(32), 1));
      dst_strides.push_back(make_const(DataType::Int(32), 1));
    }
    Buffer dst = store->buffer;
    {
      auto writer = dst.CopyOnWrite();
      writer->shape = dst_shape;
      writer->strides = dst_strides;
      writer->elem_offset = store_strides[loop_var_size];
    }

    Buffer src = load->buffer;
    {
      auto writer = src.CopyOnWrite();
      writer->shape = src_shape;
      writer->strides = src_strides;
      writer->elem_offset = src_elem_offset;
    }
    *out = flower_copy_fromto_(src, dst, pad_before, pad_after, pad_value);
    if (!out->defined()) {
      *error_info = "flower function did not return correct stmt";
      return false;
    }
    return true;
  }

  // pragma key
  std::string pragma_key_;
  // function to lower copy intrinsics.
  const PackedFunc& flower_copy_fromto_;
  // arith analyzer
  arith::Analyzer analyzer_;
};

Stmt InjectCopyIntrin(Stmt stmt, const std::string& pragma_key,
                      const PackedFunc& flower_copy_fromto) {
  return CopyIntrinInjector(pragma_key, flower_copy_fromto)(std::move(stmt));
}

namespace transform {

Pass InjectCopyIntrin(String pragma_key, PackedFunc flower_copy_fromto) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = CopyIntrinInjector(pragma_key, flower_copy_fromto)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectCopyIntrin", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectCopyIntrin").set_body_typed(InjectCopyIntrin);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
