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

Doc PrintBlock(IRDocsifier d, tir::Block block, ObjectPath block_p,  //
               Optional<tir::BlockRealize> opt_realize, Optional<ObjectPath> opt_realize_p) {
  With<TIRFrame> frame(d, block);
  ICHECK_EQ(opt_realize.defined(), opt_realize_p.defined());
  const tir::BlockRealizeNode* realize =
      opt_realize.defined() ? opt_realize.value().get() : nullptr;
  const ObjectPathNode* realize_p = opt_realize_p.defined() ? opt_realize_p.get() : nullptr;
  // Step 1. Handle block var and block bindings
  // Step 1.1. Obtain all loop var defined along path
  std::unordered_map<const tir::VarNode*, tir::For> loop_vars;
  for (Frame f : d->frames) {
    if (const auto* tir_f = f.as<TIRFrameNode>()) {
      if (auto for_loop = tir_f->tir.as<tir::For>()) {
        for (Optional<tir::For> loop = for_loop; loop; loop = loop.value()->body.as<tir::For>()) {
          loop_vars.insert(std::make_pair(loop.value()->loop_var.get(), loop.value()));
        }
      }
    }
  }

  std::vector<int> remap_vars_indices;
  auto add_remapped_iter_var = [&](int i) -> bool {
    if (realize && d->cfg->syntax_sugar) {
      tir::ExprDeepEqual expr_equal;
      tir::IterVar iter_var = block->iter_vars[i];
      PrimExpr value = realize->iter_values[i];
      if (iter_var->iter_type == tir::IterVarType::kDataPar ||
          iter_var->iter_type == tir::IterVarType::kCommReduce) {
        if (const auto* var = value.as<tir::VarNode>()) {
          if (loop_vars.count(var)) {
            tir::For for_loop = loop_vars.at(var);
            if (expr_equal(for_loop->min, iter_var->dom->min) &&
                expr_equal(for_loop->extent, iter_var->dom->extent)) {
              remap_vars_indices.push_back(i);
              return true;
            }
          }
        }
      }
    }
    return false;
  };

  auto print_single_iter_var = [&](int i) {
    tir::IterVar iter_var = block->iter_vars[i];
    ObjectPath iter_var_p = block_p->Attr("iter_var")->ArrayIndex(i);
    ExprDoc rhs = TIR(d, "axis");
    if (iter_var->iter_type == tir::IterVarType::kDataPar) {
      rhs = rhs->Attr("spatial");
    } else if (iter_var->iter_type == tir::IterVarType::kCommReduce) {
      rhs = rhs->Attr("reduce");
    } else if (iter_var->iter_type == tir::IterVarType::kOrdered) {
      rhs = rhs->Attr("scan");
    } else if (iter_var->iter_type == tir::IterVarType::kOpaque) {
      rhs = rhs->Attr("opaque");
    } else {
      LOG(FATAL) << "ValueError: Unknown IterVarType in block signature: "
                 << tir::IterVarType2String(iter_var->iter_type);
    }
    ExprDoc dom{nullptr};
    if (tir::is_zero(iter_var->dom->min)) {
      ExprDoc extent = d->AsDoc<ExprDoc>(iter_var->dom->extent,  //
                                         iter_var_p->Attr("dom")->Attr("extent"));
      dom = extent;
    } else {
      ExprDoc min = d->AsDoc<ExprDoc>(iter_var->dom->min, iter_var_p->Attr("dom")->Attr("min"));
      ExprDoc max = d->AsDoc<ExprDoc>(iter_var->dom->min + iter_var->dom->extent,
                                      iter_var_p->Attr("dom")->Attr("extent"));
      dom = TupleDoc({min, max});
    }
    if (realize) {
      ExprDoc binding = d->AsDoc<ExprDoc>(realize->iter_values[i],  //
                                          realize_p->Attr("iter_values")->ArrayIndex(i));
      rhs = rhs->Call({dom, binding});
    } else {
      rhs = rhs->Call({dom});
    }
    (*frame)->stmts.push_back(AssignDoc(DefineVar(iter_var->var, *frame, d), rhs, NullOpt));
  };

  auto print_remapped_iter_var = [&]() {
    if (remap_vars_indices.size()) {
      int m = remap_vars_indices.size();
      if (!m) {
        return;
      }
      if (m == 1) {
        print_single_iter_var(remap_vars_indices[0]);
        remap_vars_indices.clear();
        return;
      }
      Array<ExprDoc> lhs;
      Array<ExprDoc> loop_var_doc;
      lhs.reserve(m);
      loop_var_doc.reserve(m);
      std::string binding_type = "";
      Array<ObjectPath> binding_paths;
      for (int i : remap_vars_indices) {
        tir::IterVar iter_var = block->iter_vars[i];
        ObjectPath iter_var_p = block_p->Attr("iter_vars")->ArrayIndex(i);
        lhs.push_back(DefineVar(iter_var->var, *frame, d));
        loop_var_doc.push_back(d->AsDoc<ExprDoc>(realize->iter_values[i],
                                                 realize_p->Attr("iter_values")->ArrayIndex(i)));
        binding_paths.push_back(iter_var_p->Attr("iter_type"));
        binding_type += iter_var->iter_type == tir::IterVarType::kDataPar ? "S" : "R";
      }
      ExprDoc rhs = TIR(d, "axis")->Attr("remap");
      ExprDoc binding_str = LiteralDoc::Str(binding_type, NullOpt);
      binding_str->source_paths = std::move(binding_paths);
      rhs = rhs->Call({binding_str, ListDoc(loop_var_doc)});
      (*frame)->stmts.push_back(AssignDoc(TupleDoc(lhs), rhs, NullOpt));
      remap_vars_indices.clear();
    }
  };

  // Step 1.2. Construct all block var bindings
  int n_vars = block->iter_vars.size();
  for (int i = 0; i < n_vars; ++i) {
    if (!add_remapped_iter_var(i)) {
      print_remapped_iter_var();
      print_single_iter_var(i);
    }
  }
  print_remapped_iter_var();

  // Step 2. Handle block predicate
  if (realize) {
    ICHECK(realize->predicate.defined() && realize->predicate->dtype.is_bool());
    if (!tir::is_one(realize->predicate)) {
      (*frame)->stmts.push_back(ExprStmtDoc(
          TIR(d, "where")
              ->Call({d->AsDoc<ExprDoc>(realize->predicate, realize_p->Attr("predicate"))})));
    }
  }
  // Step 3. Handle block read/write regions
  {
    Array<ExprDoc> reads;
    for (int i = 0, n = block->reads.size(); i < n; ++i) {
      reads.push_back(d->AsDoc<ExprDoc>(block->reads[i], block_p->Attr("reads")->ArrayIndex(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(TIR(d, "reads")->Call(reads)));
    Array<ExprDoc> writes;
    for (int i = 0, n = block->writes.size(); i < n; ++i) {
      writes.push_back(d->AsDoc<ExprDoc>(block->writes[i], block_p->Attr("writes")->ArrayIndex(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(TIR(d, "writes")->Call(writes)));
  }
  // Step 4. Handle block attributes
  if (!block->annotations.empty()) {
    (*frame)->stmts.push_back(ExprStmtDoc(
        TIR(d, "block_attr")
            ->Call({d->AsDoc<ExprDoc>(block->annotations, block_p->Attr("annotations"))})));
  }
  // Step 5. Handle `alloc_buffer`
  for (int i = 0, n = block->alloc_buffers.size(); i < n; ++i) {
    tir::Buffer buffer = block->alloc_buffers[i];
    ObjectPath buffer_p = block_p->Attr("alloc_buffers")->ArrayIndex(i);
    IdDoc lhs = DefineBuffer(buffer, *frame, d);
    ExprDoc rhs = BufferDecl(buffer, "alloc_buffer", {}, buffer_p, *frame, d,
                             BufferVarDefinition::DataPointer);
    (*frame)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
  }
  // Step 6. Handle `match_buffer`
  for (int i = 0, n = block->match_buffers.size(); i < n; ++i) {
    tir::MatchBufferRegion buffer_region = block->match_buffers[i];
    ObjectPath buffer_region_p = block_p->Attr("match_buffers")->ArrayIndex(i);
    StmtDoc doc = d->AsDoc<StmtDoc>(buffer_region, buffer_region_p);
    (*frame)->stmts.push_back(doc);
  }
  // Step 7. Handle init block
  if (block->init.defined()) {
    tir::Stmt init = block->init.value();
    With<TIRFrame> init_frame(d, init);
    AsDocBody(init, block_p->Attr("init"), init_frame->get(), d);
    (*frame)->stmts.push_back(ScopeDoc(NullOpt, TIR(d, "init")->Call({}), (*init_frame)->stmts));
  }
  // Step 8. Handle block body
  AsDocBody(block->body, block_p->Attr("body"), frame->get(), d);
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  if (!realize) {
    kwargs_keys.push_back("no_realize");
    kwargs_values.push_back(LiteralDoc::Boolean(true, NullOpt));
  }
  return ScopeDoc(NullOpt,
                  TIR(d, "block")  //
                      ->Call({LiteralDoc::Str(block->name_hint, block_p->Attr("name_hint"))},
                             kwargs_keys, kwargs_values),
                  (*frame)->stmts);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BlockRealize>(
        "", [](tir::BlockRealize realize, ObjectPath p, IRDocsifier d) -> Doc {
          Doc doc = PrintBlock(d, realize->block, p->Attr("block"), realize, p);
          // since we do not have d->AsDoc for realize->block,
          // we should add possible doc decoration manually.
          AddDocDecoration<ScopeDoc>(doc, realize->block, p->Attr("block"), d->cfg);
          return doc;
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Block>("", [](tir::Block block, ObjectPath p, IRDocsifier d) -> Doc {
      return PrintBlock(d, block, p, NullOpt, NullOpt);
    });

TVM_SCRIPT_REPR(tir::BlockNode, ReprPrintTIR);
TVM_SCRIPT_REPR(tir::BlockRealizeNode, ReprPrintTIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm
