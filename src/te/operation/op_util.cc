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
 * \brief Utility to make loop nest.
 * \file op_util.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/operation.h>
#include <string>
#include "op_util.h"
#include "../schedule/message_passing.h"
#include "../../arith/compute_expr.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace te {

using namespace arith;
using namespace tir;

std::vector<std::vector<Stmt> >
MakeLoopNest(const Stage& stage,
             const std::unordered_map<IterVar, Range>& dom_map,
             size_t begin_iter_pos,
             bool new_loop_var,
             const std::unordered_set<IterVar>& skip_iter,
             std::unordered_map<IterVar, PrimExpr>* p_value_map,
             bool debug_keep_trivial_loop) {
  auto leaf_iter_vars = stage->leaf_iter_vars;
  Stmt no_op = EvaluateNode::make(0);
  // create the loop nest
  std::vector<std::vector<Stmt> > nest;
  nest.resize(leaf_iter_vars.size() + 1);
  std::unordered_map<IterVar, PrimExpr>& value_map = *p_value_map;

  for (size_t i = begin_iter_pos; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    if (skip_iter.count(iv) || iv->iter_type == kOpaque) {
      // skip this iteration.
      value_map[iv] = iv->var;
      continue;
    }
    // Bind iv could be another thread.
    IterVar bind_iv = iv;
    if (stage->iter_var_attrs.count(iv)) {
      IterVar bind_thread = stage->iter_var_attrs[iv]->bind_thread;
      if (bind_thread.defined()) bind_iv = bind_thread;
    }

    Range dom = dom_map.at(iv);

    // initialize the offset and loop_level
    Var var = bind_iv->var;

    // Mark the iter var in the IR, to remember the point
    if (bind_iv->thread_tag.length() == 0) {
      // Only generate new loop if we're not bound to a thread.
      if (new_loop_var) {
        var = Var(iv->var->name_hint + ".init", bind_iv->var.dtype());
      }

      ForType for_type = ForType::Serial;
      IterVarAttr it_attr;
      if (stage->iter_var_attrs.count(iv)) {
        it_attr = stage->iter_var_attrs[iv];
      }
      if (it_attr.defined()) {
        switch (it_attr->iter_type) {
          case kUnrolled: for_type = ForType::Unrolled; break;
          case kVectorized: for_type = ForType::Vectorized; break;
          case kParallelized: for_type = ForType::Parallel; break;
          case kDataPar: break;
          case kTensorized: break;
          default: LOG(FATAL) << "Unknown iter type"
                              << it_attr->iter_type
                              << " in the iter_var_attrs";
        }
        CHECK_EQ(it_attr->pragma_keys.size(), it_attr->pragma_values.size());
        for (size_t k = 0; k < it_attr->pragma_keys.size(); ++k) {
          const std::string& pkey = it_attr->pragma_keys[k].as<StringImmNode>()->value;
          PrimExpr pvalue = it_attr->pragma_values[k];
          if (!pvalue.defined()) {
            pvalue = make_const(DataType::Int(32), 1);
          }
          nest[i + 1].emplace_back(
              AttrStmtNode::make(iv, tir::attr::pragma_scope_prefix + pkey, pvalue, no_op));
        }
      }
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        nest[i + 1].emplace_back(
            LetStmtNode::make(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(
            ForNode::make(var, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        value_map[iv] = var;
      } else {
        Var idx(bind_iv->var->name_hint + ".idx", bind_iv->var.dtype());
        nest[i + 1].emplace_back(
            ForNode::make(idx, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        PrimExpr new_value = dom->min + idx;
        value_map[iv] = new_value;
        nest[i + 1].emplace_back(
            LetStmtNode::make(var, new_value, no_op));
      }
      if (it_attr.defined() && it_attr->prefetch_data.size() != 0) {
        CHECK(!is_one(dom->extent))
            << "Cannot prefetch on trivial loop with extent=1";
        CHECK_EQ(it_attr->prefetch_data.size(),
                 it_attr->prefetch_offset.size());
        for (size_t j = 0; j < it_attr->prefetch_data.size(); ++j) {
          nest[i + 1].emplace_back(
              AttrStmtNode::make(it_attr->prefetch_data[j],
                             tir::attr::prefetch_scope,
                             it_attr->prefetch_offset[j], no_op));
        }
      }
    } else if (bind_iv->thread_tag == "vthread" ||
               bind_iv->thread_tag == "cthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      CHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else if (bind_iv->thread_tag == "pipeline") {
      // pipeline marker.
      CHECK(is_zero(dom->min));
      CHECK(is_one(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::pipeline_exec_scope, dom->extent, no_op));
      value_map[iv] = dom->min;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmtNode::make(bind_iv, tir::attr::thread_extent, dom->extent, no_op));
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        runtime::ThreadScope ts = runtime::ThreadScope::make(bind_iv->thread_tag);
        if (stage->scope == "" || stage->scope == "warp" ||
            static_cast<int>(runtime::StorageScope::make(stage->scope).rank) <= ts.rank) {
          value_map[iv] = var;
        } else {
          value_map[iv] = dom->min;
        }
      }
    }
    // annotate the extent of the IterVar
    if (!new_loop_var) {
      nest[i + 1].emplace_back(
          AttrStmtNode::make(iv, tir::attr::loop_scope, iv->var, no_op));
    }
  }
  // message passing to get offset of root iter vars.
  te::PassUpIndex(stage, dom_map, &value_map);
  return nest;
}

std::vector<Stmt> MakeIfNest(const std::vector<PrimExpr>& predicates) {
  Stmt no_op = EvaluateNode::make(0);
  std::vector<Stmt> nest;
  for (const PrimExpr& cond : predicates) {
    nest.emplace_back(IfThenElseNode::make(cond, no_op));
  }
  return nest;
}

// replacer to replace tensors
class TensorReplacer : public tir::StmtExprMutator {
 public:
  explicit TensorReplacer(const std::unordered_map<Tensor, Tensor>& vmap)
      : vmap_(vmap) {}

  PrimExpr VisitExpr_(const tir::CallNode* op) final {
    if (op->call_type == tir::CallNode::Halide) {
      Tensor t = Downcast<Operation>(op->func).output(op->value_index);
      auto it = vmap_.find(t);
      if (it != vmap_.end()) {
        PrimExpr ret = tir::CallNode::make(
            op->dtype, it->second->op->name, op->args,
            op->call_type, it->second->op, it->second->value_index);
        found = true;
        return this->VisitExpr(ret);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor>& vmap_;
};

Stmt ReplaceTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  Stmt ret = repl(stmt);
  return repl.found ? ret : stmt;
}
PrimExpr ReplaceTensor(PrimExpr expr,
                       const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  PrimExpr ret = repl(expr);
  return repl.found ? ret : expr;
}


Stmt Substitute(Stmt s,
                const std::unordered_map<IterVar, PrimExpr>& value_map) {
  std::unordered_map<const VarNode*, PrimExpr> init;
  for (const auto& kv : value_map) {
    init[kv.first->var.get()] = kv.second;
  }
  return tir::Substitute(s, init);
}

IterVarType ForTypeToIterVarType(tir::ForType for_type) {
  switch (for_type) {
  case ForType::Serial:
    return kDataPar;
  case ForType::Parallel:
    return kParallelized;
  case ForType::Vectorized:
    return kVectorized;
  case ForType::Unrolled:
    return kUnrolled;
  default:
    return kDataPar;
  }
}

tir::ForType IterVarTypeToForType(IterVarType iter_type) {
  switch (iter_type) {
  case kDataPar:
    return ForType::Serial;
  case kParallelized:
    return ForType::Parallel;
  case kVectorized:
    return ForType::Vectorized;
  case kUnrolled:
    return ForType::Unrolled;
  default:
    return ForType::Serial;
  }
}

}  // namespace te
}  // namespace tvm
