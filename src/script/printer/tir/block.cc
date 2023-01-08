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
  const tir::BlockRealizeNode* realize = opt_realize.value().get();
  const ObjectPathNode* realize_p = opt_realize_p.get();
  // Step 1. Handle block var and block bindings
  int n_vars = block->iter_vars.size();
  for (int i = 0; i < n_vars; ++i) {
    tir::IterVar iter_var = block->iter_vars[i];
    ObjectPath iter_var_p = block_p->Attr("iter_var")->ArrayIndex(i);
    ExprDoc rhs = TIR(d)->Attr("axis");
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
  }
  // Step 2. Handle block predicate
  if (realize) {
    ICHECK(realize->predicate.defined() && realize->predicate->dtype.is_bool());
    if (!tir::is_one(realize->predicate)) {
      (*frame)->stmts.push_back(ExprStmtDoc(TIR(d)->Attr("where")->Call(
          {d->AsDoc<ExprDoc>(realize->predicate, realize_p->Attr("predicate"))})));
    }
  }
  // Step 3. Handle block read/write regions
  {
    Array<ExprDoc> reads;
    for (int i = 0, n = block->reads.size(); i < n; ++i) {
      reads.push_back(d->AsDoc<ExprDoc>(block->reads[i], block_p->Attr("reads")->ArrayIndex(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(TIR(d)->Attr("reads")->Call(reads)));
    Array<ExprDoc> writes;
    for (int i = 0, n = block->writes.size(); i < n; ++i) {
      writes.push_back(d->AsDoc<ExprDoc>(block->writes[i], block_p->Attr("writes")->ArrayIndex(i)));
    }
    (*frame)->stmts.push_back(ExprStmtDoc(TIR(d)->Attr("writes")->Call(writes)));
  }
  // Step 4. Handle block attributes
  if (!block->annotations.empty()) {
    (*frame)->stmts.push_back(ExprStmtDoc(
        TIR(d)
            ->Attr("block_attr")
            ->Call({d->AsDoc<ExprDoc>(block->annotations, block_p->Attr("annotations"))})));
  }
  // Step 5. Handle `alloc_buffer`
  for (int i = 0, n = block->alloc_buffers.size(); i < n; ++i) {
    tir::Buffer buffer = block->alloc_buffers[i];
    ObjectPath buffer_p = block_p->Attr("alloc_buffers")->ArrayIndex(i);
    IdDoc lhs = DefineBuffer(buffer, *frame, d);
    ExprDoc rhs = BufferDecl(buffer, "alloc_buffer", {}, buffer_p, *frame, d);
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
    (*frame)->stmts.push_back(
        ScopeDoc(NullOpt, TIR(d)->Attr("init")->Call({}), (*init_frame)->stmts));
  }
  // Step 8. Handle block body
  AsDocBody(block->body, block_p->Attr("body"), frame->get(), d);
  return ScopeDoc(NullOpt, TIR(d)->Attr("block")->Call({LiteralDoc::Str(block->name_hint)}),
                  (*frame)->stmts);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BlockRealize>(
        "", [](tir::BlockRealize realize, ObjectPath p, IRDocsifier d) -> Doc {
          return PrintBlock(d, realize->block, p->Attr("block"), realize, p);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Block>("", [](tir::Block block, ObjectPath p, IRDocsifier d) -> Doc {
      return PrintBlock(d, block, p, NullOpt, NullOpt);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::MatchBufferRegion>(
        "", [](tir::MatchBufferRegion stmt, ObjectPath p, IRDocsifier d) -> Doc {
          Frame frame = d->frames.back();
          ExprDoc lhs = DefineBuffer(stmt->buffer, frame, d);
          ExprDoc src_buffer = d->AsDoc<ExprDoc>(stmt->source, p->Attr("source"));
          ExprDoc rhs = BufferDecl(stmt->buffer, "match_buffer", {src_buffer}, p->Attr("buffer"),
                                   d->frames.back(), d);
          return AssignDoc(lhs, rhs, NullOpt);
        });

}  // namespace printer
}  // namespace script
}  // namespace tvm
