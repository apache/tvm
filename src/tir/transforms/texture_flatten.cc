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
 * \file storage_flatten.cc
 * \brief Flattens storage from multi-dimensional array to 1D buffer access
 */
// The pass definition originates from Halide pipeline.

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <stack>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {
namespace {
  using IndexMod = tir::FloorModNode;
  using IndexDiv = tir::FloorDivNode;

// Split the given expression w.r.t the add operator
inline std::vector<const PrimExpr*> ExprSplitAddition(const PrimExpr& expr) {
  using namespace tir;
  std::vector<const PrimExpr*> ret;
  std::stack<const PrimExpr*> split_buffer;
  split_buffer.push(&expr);
  while (!split_buffer.empty()) {
    const PrimExpr* top_ele = split_buffer.top();
    split_buffer.pop();
    auto expr_add_match = top_ele->as<AddNode>();
    if (expr_add_match) {
      split_buffer.push(&expr_add_match->b);
      split_buffer.push(&expr_add_match->a);
    } else {
      ret.emplace_back(top_ele);
    }
  }
  return ret;
}

// Searches for the following types of expr:
//   mult_expr = (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   mod_l_expr = c
//   mod_r_expr = k1 * k2 * ... * ki
// If it can be optimized, returns (true, (a1 + a2 + ... + aj) * kt * ... * ki + c)
// Currently the we will not search the add/mult combinations exhaustively
//   as it will take too much computation.
inline std::pair<bool, PrimExpr> MergeMulModInner(const PrimExpr& mult_expr,
                                                  const PrimExpr& mod_l_expr,
                                                  const PrimExpr& mod_r_expr) {
  using namespace tir;
  const MulNode* mult_ptr = mult_expr.as<MulNode>();
  if (!mult_ptr) return std::make_pair(false, PrimExpr());
  PrimExpr mult_outer = mult_ptr->b;
  const PrimExpr* inner = &(mult_ptr->a);
  // 1. Calculate the outer multiplier
  while (true) {
    mult_ptr = inner->as<MulNode>();
    if (mult_ptr) {
      inner = &(mult_ptr->a);
      mult_outer = mult_ptr->b * mult_outer;
    } else {
      break;
    }
  }
  // 2. Search for the pattern c / (...) * (...) + c % (...)
  // We match the search element with Add, Mul and Div.
  //   If Add is found, we need to continue our search for the rhs
  //   If Mult is found, we will expand the inner multiplication factor
  //   If Div is found, we will go on testing whether lhs matches the lhs of mod expr
  //      and returns the optimization result.
  const PrimExpr* search_ptr = inner;
  PrimExpr mult_inner;  // The inner multiplication factor
  PrimExpr no_opt_sum;  // Sum of the exprs that cannot be optimized
  tir::ExprDeepEqual expr_equal;

  while (true) {
    auto inner_div_ptr = search_ptr->as<IndexDiv>();
    auto inner_mult_ptr = search_ptr->as<MulNode>();
    auto inner_add_ptr = search_ptr->as<AddNode>();
    if (!inner_div_ptr && !inner_mult_ptr && !inner_add_ptr) {
      return std::make_pair(false, PrimExpr());
    } else if (inner_div_ptr) {
      PrimExpr overall_mult = mult_inner.get() ? mult_inner * mult_outer : mult_outer;
      if (expr_equal(overall_mult, inner_div_ptr->b) && expr_equal(overall_mult, mod_r_expr) &&
          expr_equal(inner_div_ptr->a, mod_l_expr)) {
        // Found!
        PrimExpr ret = no_opt_sum.get() ? no_opt_sum * mult_outer + mod_l_expr : mod_l_expr;
        return std::make_pair(true, ret);
      } else {
        return std::make_pair(false, PrimExpr());
      }
    } else if (inner_mult_ptr) {
      mult_inner = mult_inner.get() ? inner_mult_ptr->b * mult_inner : inner_mult_ptr->b;
      search_ptr = &(inner_mult_ptr->a);
    } else if (inner_add_ptr) {
      if (mult_inner.get()) {
        return std::make_pair(false, PrimExpr());
      }
      no_opt_sum = no_opt_sum.get() ? no_opt_sum + inner_add_ptr->a : inner_add_ptr->a;
      search_ptr = &(inner_add_ptr->b);
    } else {
      LOG(FATAL) << "Unexpected search result!";
      break;
    }
  }
  return std::make_pair(false, PrimExpr());
}

// Insert the elements into the corresponding mult_exprs and mod_exprs.
// If the element is found to match Mul, it will be pushed to the mult_exprs.
// If the element it found to match Mod, it will be pused to the mod_exprs.
// Otherwise, the elements will be added to the no_opt_sum variable
inline void MergeMulModInsertElements(const std::vector<const PrimExpr*>& eles,
                                      std::list<PrimExpr>* mult_exprs,
                                      std::list<std::pair<PrimExpr, PrimExpr> >* mod_exprs,
                                      PrimExpr* no_opt_sum, bool* has_mult, bool* has_mod) {
  using namespace tir;
  *has_mult = false;
  *has_mod = false;
  for (const PrimExpr* ele : eles) {
    auto mod_ptr = ele->as<IndexMod>();
    auto mult_ptr = ele->as<MulNode>();
    if (mod_ptr) {
      *has_mod = true;
      mod_exprs->emplace_back(std::make_pair(std::move(mod_ptr->a), std::move(mod_ptr->b)));
    } else if (mult_ptr) {
      *has_mult = true;
      mult_exprs->emplace_back(*ele);
    } else {
      *no_opt_sum = no_opt_sum->get() ? *no_opt_sum + *ele : *ele;
    }
  }
}

// Searches for this types of expr:
//   (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   + c % (k1 * k2 * ... * ki)
// and simplifies to (a1 + a2 + ... + aj) * kt * ... * ki + c
// The search will be performed repeatively until no pattern is found.
// Return: a pair with (false, Expr()) if cannot be optimized.
//         a pair with (true, optimized_expr) if can be optimized
inline PrimExpr MergeMulMod(arith::Analyzer* analyzer, const PrimExpr& base) {
  using namespace tir;
  // 1. Prepare the lists.
  // We store two lists, a list that contain all the elements that match Mul and
  //                     a list that contain all the elements that match Mod.
  // The elements in the Mod will be used to match against the elements in Mul.
  // The result will then be split and pushed back to these two lists.
  PrimExpr simplified_base = analyzer->Simplify(base);
  std::vector<const PrimExpr*> eles = ExprSplitAddition(simplified_base);
  std::list<PrimExpr> mult_exprs;
  std::list<std::pair<PrimExpr, PrimExpr> > mod_exprs;
  PrimExpr no_opt_sum;
  bool has_mult;
  bool has_mod;
  MergeMulModInsertElements(eles, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult, &has_mod);
  bool find_opt = false;
  std::list<std::pair<PrimExpr, PrimExpr> >::iterator search_mod_it = mod_exprs.begin();
  // 2. Exhaustive Search
  while (search_mod_it != mod_exprs.end()) {
    std::list<PrimExpr>::iterator mult_it = mult_exprs.begin();
    bool inner_find_opt = false;
    while (mult_it != mult_exprs.end()) {
      std::pair<bool, PrimExpr> ret =
          MergeMulModInner(*mult_it, search_mod_it->first, search_mod_it->second);
      if (ret.first) {
        inner_find_opt = true;
        auto temp_mod_it = search_mod_it;
        ++search_mod_it;
        mod_exprs.erase(temp_mod_it);
        mult_exprs.erase(mult_it);
        std::vector<const PrimExpr*> ret_eles = ExprSplitAddition(ret.second);
        MergeMulModInsertElements(ret_eles, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult,
                                  &has_mod);
        if (has_mult) {
          search_mod_it = mod_exprs.begin();
        } else if (has_mod && search_mod_it == mod_exprs.end()) {
          search_mod_it--;
        }
        break;
      } else {
        ++mult_it;
      }
    }
    find_opt = find_opt || inner_find_opt;
    if (!inner_find_opt) {
      ++search_mod_it;
    }
  }
  if (!find_opt) {
    return simplified_base;
  }
  for (std::list<PrimExpr>::iterator it = mult_exprs.begin(); it != mult_exprs.end(); ++it) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + *it : *it;
  }
  for (std::list<std::pair<PrimExpr, PrimExpr> >::iterator it = mod_exprs.begin();
       it != mod_exprs.end(); ++it) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + indexmod(it->first, it->second)
                                  : indexmod(it->first, it->second);
  }
  return no_opt_sum;
}

inline PrimExpr SimplifyOffset(const Array<PrimExpr>& shape, const Array<PrimExpr>& index) {
  PrimExpr base = make_const(DataType::Int(32), 0); //IntImm(DataType::Int(32), 0);
  ICHECK_EQ(shape.size(), index.size());
  arith::Analyzer ana;
  if (index.size() > 0) {
    PrimExpr offset = index[0];
    for (size_t i = 1; i < index.size(); ++i) {
      offset = MergeMulMod(&ana, offset * shape[i] + index[i]);
    }
    base = base + offset;
  }
  return base;
}

size_t GetAxisSeparator(size_t shape_rank) {
  // Convention is that shape is packed with the last axis
  // as RGBA (length 4) and the second to last axis
  // will be the packed texure columns. All other
  // axes are packed into rows.
  //
  // e.g. [N,C,H,W,c] -> TextureFlattening -> [N*C*H, W, c]
  //

  return shape_rank - 2;
}
}

class TextureFlattener : public StmtExprMutator {
 public:
  explicit TextureFlattener() {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImmNode>()->value;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    Var buffer_var(op->buffer->data->name_hint, TextureType(op->buffer->dtype));
    let_binding_.insert({op->buffer->data, buffer_var});

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferRealizeNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      Stmt body = this->VisitStmt(op->body);
      ICHECK(op->bounds.size() >= 3) << "Only 2d RGBA texture is currently supported";
      ICHECK_EQ(static_cast<int>(op->bounds.back()->extent.as<IntImmNode>()->value), 4) << "FCD of texture must be vector of length 4 (RGBA)";

      Array<PrimExpr> shape;
      auto width = IntImm(DataType::Int(32), 1);
      auto height = IntImm(DataType::Int(32), 1);
      // TODO(csulivan): We do not currently handle the case where
      // the last dimension isn't previously set to a vector(4)
      for (size_t i = 0; i < op->bounds.size()-1; i++) {
        if (i < GetAxisSeparator(op->bounds.size())) {
          width *= op->bounds[i]->extent;
        } else {
          height *= op->bounds[i]->extent;
        }
      }

      Array<PrimExpr> args = {width, height};
      stmt = LetStmt(buffer_var, Call(buffer_var.dtype(), builtin::text2d_alloca(), args), body);
    }

    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      Array<PrimExpr> args;
      if (let_binding_.count(op->buffer->data))
      {
        args.push_back(let_binding_[op->buffer->data]);
      }
      else
      {
        args.push_back(op->buffer->data);
      }

      Array<PrimExpr> row_dims, row_indices, col_dims, col_indices;
      for (size_t i = 0; i < op->buffer->shape.size()-1; i++)
      {
        if (i < GetAxisSeparator(op->buffer->shape.size())) {
          row_dims.push_back(op->buffer->shape[i]);
          row_indices.push_back(op->indices[i]);
        } else {
          col_dims.push_back(op->buffer->shape[i]);
          col_indices.push_back(op->indices[i]);
        }
      }

      PrimExpr row_offset = SimplifyOffset(row_dims, row_indices);
      PrimExpr col_offset = SimplifyOffset(col_dims, col_indices);

      args.push_back(row_offset);
      args.push_back(col_offset);
      args.push_back(op->value);

      stmt = Evaluate(Call(args[0]->dtype, builtin::text2d_store(), args));
    }

    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      Array<PrimExpr> args;
      if (let_binding_.count(op->buffer->data))
      {
        args.push_back(let_binding_[op->buffer->data]);
      }
      else
      {
        args.push_back(op->buffer->data);
      }


      Array<PrimExpr> row_dims, row_indices, col_dims, col_indices;
      for (size_t i = 0; i < op->buffer->shape.size()-1; i++)
      {
        if (i < GetAxisSeparator(op->buffer->shape.size())) {
          row_dims.push_back(op->buffer->shape[i]);
          row_indices.push_back(op->indices[i]);
        } else {
          col_dims.push_back(op->buffer->shape[i]);
          col_indices.push_back(op->indices[i]);
        }
      }

      PrimExpr row_offset = SimplifyOffset(row_dims, row_indices);
      PrimExpr col_offset = SimplifyOffset(col_dims, col_indices);
      args.push_back(row_offset);
      args.push_back(col_offset);
      args.push_back(op->indices.back());
      expr = Call(op->buffer->dtype, builtin::text2d_load(), args);
    }

    return expr;
  }

 private:
  // Storage scope
  std::unordered_map<const Object*, std::string> storage_scope_;
  // Let binding
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> let_binding_;
};

PrimFunc TextureFlatten(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = TextureFlattener()(std::move(fptr->body));
  return func;
}

namespace transform {

Pass TextureFlatten() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TextureFlatten(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.TextureFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.TextureFlatten").set_body_typed(TextureFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
