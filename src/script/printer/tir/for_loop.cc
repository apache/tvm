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
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::For>("", [](tir::For loop, ObjectPath p, IRDocsifier d) -> Doc {
      // Step 1. Check syntactic sugar: `T.grid`
      std::vector<const tir::ForNode*> grid;
      std::unordered_set<const tir::VarNode*> grid_loop_vars;
      auto f_var_dep = [&grid_loop_vars](const PrimExpr& e) -> bool {
        return tir::UsesVar(e, [&grid_loop_vars](const tir::VarNode* v) -> bool {  //
          return grid_loop_vars.count(v);
        });
      };
      for (const tir::ForNode* l = loop.get(); l != nullptr; l = l->body.as<tir::ForNode>()) {
        ICHECK(l->loop_var->dtype == l->min->dtype);
        ICHECK(l->loop_var->dtype == l->extent->dtype);
        if (l->kind != tir::ForKind::kSerial ||  //
            !tir::is_zero(l->min) ||             //
            !l->annotations.empty() ||           //
            f_var_dep(l->extent)) {
          break;
        }
        grid.push_back(l);
        grid_loop_vars.insert(l->loop_var.get());
      }
      With<TIRFrame> f(d, loop);
      // Step 2. Construct `T.grid`
      if (grid.size() > 1) {
        int n = grid.size();
        Array<ExprDoc> lhs;
        Array<ExprDoc> rhs;
        lhs.reserve(n);
        rhs.reserve(n);
        for (int i = 0; i < n; ++i) {
          const tir::ForNode* loop = grid[i];
          lhs.push_back(DefineVar(loop->loop_var, *f, d));
          rhs.push_back(d->AsDoc<ExprDoc>(loop->extent, p->Attr("extent")));
          p = p->Attr("body");
        }
        AsDocBody(grid.back()->body, p, (*f).get(), d);
        return ForDoc(TupleDoc(lhs), TIR(d)->Attr("grid")->Call(rhs), (*f)->stmts);
      }
      // Step 3. If not `T.grid`, print loop kind accordingly
      IdDoc lhs = DefineVar(loop->loop_var, *f, d);
      Optional<ExprDoc> min = NullOpt;
      Optional<ExprDoc> max = NullOpt;
      Optional<ExprDoc> annotations = NullOpt;
      Optional<ExprDoc> thread = NullOpt;
      if (tir::is_zero(loop->min)) {
        max = d->AsDoc<ExprDoc>(loop->extent, p->Attr("extent"));
      } else {
        min = d->AsDoc<ExprDoc>(loop->min, p->Attr("min"));
        max = d->AsDoc<ExprDoc>(loop->min + loop->extent, p->Attr("extent"));
      }
      if (!loop->annotations.empty()) {
        annotations = d->AsDoc<ExprDoc>(loop->annotations, p->Attr("annotations"));
      }
      ExprDoc prefix = TIR(d);
      if (loop->kind == tir::ForKind::kSerial) {
        if (loop->annotations.empty()) {
          prefix = IdDoc("range");
        } else {
          prefix = prefix->Attr("serial");
        }
      } else if (loop->kind == tir::ForKind::kParallel) {
        prefix = prefix->Attr("parallel");
      } else if (loop->kind == tir::ForKind::kUnrolled) {
        prefix = prefix->Attr("unroll");
      } else if (loop->kind == tir::ForKind::kVectorized) {
        prefix = prefix->Attr("vectorized");
      } else if (loop->kind == tir::ForKind::kThreadBinding) {
        prefix = prefix->Attr("thread_binding");
        thread = LiteralDoc::Str(loop->thread_binding.value()->thread_tag);
      } else {
        LOG(FATAL) << "ValueError: Unknown ForKind: " << tir::ForKind2String(loop->kind);
      }
      Array<ExprDoc> args;
      Array<String> kwargs_keys;
      Array<ExprDoc> kwargs_values;
      if (min.defined()) {
        args.push_back(min.value());
      }
      if (max.defined()) {
        args.push_back(max.value());
      }
      if (thread.defined()) {
        kwargs_keys.push_back("thread");
        kwargs_values.push_back(thread.value());
      }
      if (annotations.defined()) {
        kwargs_keys.push_back("annotations");
        kwargs_values.push_back(annotations.value());
      }
      ExprDoc rhs = prefix->Call(args, kwargs_keys, kwargs_values);
      AsDocBody(loop->body, p, (*f).get(), d);
      return ForDoc(lhs, rhs, (*f)->stmts);
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
