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
 * \file op_utils.cc
 */
#include "op_utils.h"

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <string>

#include "../../runtime/thread_storage_scope.h"
#include "../schedule/message_passing.h"

namespace tvm {
namespace te {

using namespace arith;
using namespace tir;

std::vector<std::vector<Stmt>> MakeLoopNest(const Stage& stage,
                                            const std::unordered_map<IterVar, Range>& dom_map,
                                            size_t begin_iter_pos, bool new_loop_var,
                                            const std::unordered_set<IterVar>& skip_iter,
                                            std::unordered_map<IterVar, PrimExpr>* p_value_map,
                                            bool debug_keep_trivial_loop) {
  auto leaf_iter_vars = stage->leaf_iter_vars;
  Stmt no_op = Evaluate(0);
  // create the loop nest
  std::vector<std::vector<Stmt>> nest;
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

    ICHECK(iv->var.dtype() == dom->min.dtype() && iv->var.dtype() == dom->extent.dtype())
        << "iter_var type " << iv->var.dtype() << " and domain types (min:" << dom->min.dtype()
        << ", extent:" << dom->extent.dtype() << ") should all be the same";

    // This is a hack to ensure that the replacing expression has the same
    // dtype as the replacing expression. This happens when a thread/block
    // itervar is bound to another itervar. Because the thread/block itervar
    // has no way to know its correct dtype before it is bound, it defaults to
    // int32. Then the itervar it is bound to may have a different dtype. The
    // thread/block dtype really should be promoted to dtype of what it is
    // bound to (in `bind`) but that would require inplace modification of the
    // itervar.
    // XXX: we will get integer overflow if the bound itervar is greater than int32::max.
    auto promote_to_iv_dtype = [type = iv->var.dtype()](PrimExpr e) {
      return type != e.dtype() ? cast(type, e) : e;
    };

    // initialize the offset and loop_level
    Var var = bind_iv->var;

    // Mark the iter var in the IR, to remember the point
    if (bind_iv->thread_tag.length() == 0) {
      // Only generate new loop if we're not bound to a thread.
      if (new_loop_var) {
        var = Var(iv->var->name_hint + ".init", bind_iv->var.dtype());
      }

      ForKind kind = ForKind::kSerial;
      IterVarAttr it_attr;
      if (stage->iter_var_attrs.count(iv)) {
        it_attr = stage->iter_var_attrs[iv];
      }
      if (it_attr.defined()) {
        switch (it_attr->iter_type) {
          case kUnrolled:
            kind = ForKind::kUnrolled;
            break;
          case kVectorized:
            kind = ForKind::kVectorized;
            break;
          case kParallelized:
            kind = ForKind::kParallel;
            break;
          case kDataPar:
            break;
          case kTensorized:
            break;
          default:
            LOG(FATAL) << "Unknown iter type" << it_attr->iter_type << " in the iter_var_attrs";
        }
        ICHECK_EQ(it_attr->pragma_keys.size(), it_attr->pragma_values.size());
        for (size_t k = 0; k < it_attr->pragma_keys.size(); ++k) {
          const std::string& pkey = it_attr->pragma_keys[k].as<StringImmNode>()->value;
          PrimExpr pvalue = it_attr->pragma_values[k];
          if (!pvalue.defined()) {
            pvalue = make_const(DataType::Int(32), 1);
          }
          nest[i + 1].emplace_back(
              AttrStmt(iv, tir::attr::pragma_scope_prefix + pkey, pvalue, no_op));
        }
      }
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        nest[i + 1].emplace_back(LetStmt(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(For(var, 0, dom->extent, kind, no_op));
        value_map[iv] = promote_to_iv_dtype(var);
      } else {
        Var idx(bind_iv->var->name_hint + ".idx", iv->var.dtype());
        nest[i + 1].emplace_back(For(idx, 0, dom->extent, kind, no_op));
        PrimExpr new_value = dom->min + idx;
        value_map[iv] = new_value;
        nest[i + 1].emplace_back(LetStmt(var, new_value, no_op));
      }
      if (it_attr.defined() && it_attr->prefetch_data.size() != 0) {
        ICHECK(!is_one(dom->extent)) << "Cannot prefetch on trivial loop with extent=1";
        ICHECK_EQ(it_attr->prefetch_data.size(), it_attr->prefetch_offset.size());
        for (size_t j = 0; j < it_attr->prefetch_data.size(); ++j) {
          nest[i + 1].emplace_back(AttrStmt(it_attr->prefetch_data[j], tir::attr::prefetch_scope,
                                            it_attr->prefetch_offset[j], no_op));
        }
      }
    } else if (bind_iv->thread_tag == "vthread" || bind_iv->thread_tag == "cthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      ICHECK(is_zero(dom->min));
      ICHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(AttrStmt(bind_iv, tir::attr::virtual_thread,
                                        cast(bind_iv->var.dtype(), dom->extent), no_op));
      value_map[iv] = promote_to_iv_dtype(var);
    } else if (bind_iv->thread_tag == "pipeline") {
      // pipeline marker.
      ICHECK(is_zero(dom->min));
      ICHECK(is_one(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(AttrStmt(bind_iv, tir::attr::pipeline_exec_scope,
                                        cast(bind_iv->var.dtype(), dom->extent), no_op));
      value_map[iv] = dom->min;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      ICHECK(is_zero(dom->min)) << "Itervar " << iv << " must start at zero, but it starts at "
                                << dom->min;
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(AttrStmt(bind_iv, tir::attr::thread_extent,
                                        cast(bind_iv->var.dtype(), dom->extent), no_op));
      if (!debug_keep_trivial_loop && is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else if (stage->scope == "") {
        value_map[iv] = promote_to_iv_dtype(var);
      } else {
        runtime::ThreadScope ts = runtime::ThreadScope::Create(bind_iv->thread_tag);
        runtime::StorageScope ss = runtime::StorageScope::Create(stage->scope);
        if (static_cast<int>(ss.rank) <= ts.rank) {
          value_map[iv] = promote_to_iv_dtype(var);
        } else if (stage->scope == "warp" && ts.rank == 1) {
          // To determine whether a thread index is inside or outside a warp, we need
          // to know the thread extent. We leave a warning for now.
          if (ts.dim_index == 0) {
            value_map[iv] = promote_to_iv_dtype(var);
          } else {
            LOG(WARNING)
                << "WARNING: threadIdx.y or threadIdx.z accessing warp-scope memory detected. "
                << "TVM assumes only threadIdx.x indicates threads inside a warp, "
                << "while threadIdx.y and threadIdx.z indicates different warps.";
            value_map[iv] = dom->min;
          }
        } else {
          value_map[iv] = dom->min;
        }
      }
    }
    // annotate the extent of the IterVar
    if (!new_loop_var) {
      nest[i + 1].emplace_back(AttrStmt(iv, tir::attr::loop_scope, iv->var, no_op));
    }
  }
  // message passing to get offset of root iter vars.
  te::PassUpIndex(stage, dom_map, &value_map);
  return nest;
}

std::vector<Stmt> MakeIfNest(const std::vector<PrimExpr>& predicates) {
  Stmt no_op = Evaluate(0);
  std::vector<Stmt> nest;
  for (const PrimExpr& cond : predicates) {
    nest.emplace_back(IfThenElse(cond, no_op));
  }
  return nest;
}

// replacer to replace tensors
class TensorReplacer : public tir::StmtExprMutator {
 public:
  explicit TensorReplacer(const std::unordered_map<Tensor, Tensor>& vmap) : vmap_(vmap) {}

  PrimExpr VisitExpr_(const tir::ProducerLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<tir::ProducerLoadNode>();
    ICHECK(op != nullptr);

    Tensor t = Downcast<Tensor>(op->producer);
    auto it = vmap_.find(t);
    if (it != vmap_.end()) {
      found = true;
      return tir::ProducerLoad(it->second, op->indices);
    } else {
      return expr;
    }
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor>& vmap_;
};

Stmt ReplaceTensor(Stmt stmt, const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  Stmt ret = repl(stmt);
  return repl.found ? ret : stmt;
}
PrimExpr ReplaceTensor(PrimExpr expr, const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  PrimExpr ret = repl(expr);
  return repl.found ? ret : expr;
}

IterVarType ForKindToIterVarType(tir::ForKind kind) {
  switch (kind) {
    case ForKind::kSerial:
      return kDataPar;
    case ForKind::kParallel:
      return kParallelized;
    case ForKind::kVectorized:
      return kVectorized;
    case ForKind::kUnrolled:
      return kUnrolled;
    default:
      return kDataPar;
  }
}

tir::ForKind IterVarTypeToForKind(IterVarType iter_type) {
  switch (iter_type) {
    case kDataPar:
      return ForKind::kSerial;
    case kParallelized:
      return ForKind::kParallel;
    case kVectorized:
      return ForKind::kVectorized;
    case kUnrolled:
      return ForKind::kUnrolled;
    default:
      return ForKind::kSerial;
  }
}

}  // namespace te
}  // namespace tvm
