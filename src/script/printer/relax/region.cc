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

Array<StmtDoc> PrintSeqExpr(const relax::SeqExpr& n, const ObjectPath& n_p, const IRDocsifier& d,
                            bool use_ret) {
  With<RelaxFrame> f(d);
  const Array<relax::BindingBlock>& blocks = n->blocks;
  ObjectPath blocks_p = n_p->Attr("blocks");
  Array<StmtDoc>* stmts = &(*f)->stmts;
  for (int i = 0, l = blocks.size(); i < l; ++i) {
    Doc block = d->AsDoc(blocks[i], blocks_p->ArrayIndex(i));
    if (const auto* stmt_block = block.as<StmtBlockDocNode>()) {
      stmts->insert(stmts->end(), stmt_block->stmts.begin(), stmt_block->stmts.end());
    } else if (const auto* stmt = block.as<StmtDocNode>()) {
      stmts->push_back(GetRef<StmtDoc>(stmt));
    } else {
      LOG(FATAL) << "TypeError: Unknown type: " << block->GetTypeKey();
    }
  }
  ExprDoc ret = d->AsDoc<ExprDoc>(n->body, n_p->Attr("body"));
  if (use_ret) {
    stmts->push_back(ReturnDoc(ret));
  } else {
    stmts->push_back(ExprStmtDoc(ret));
  }
  return *stmts;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::SeqExpr>("", [](relax::SeqExpr n, ObjectPath n_p, IRDocsifier d) -> Doc {
      return StmtBlockDoc(PrintSeqExpr(n, n_p, d, false));
    });

Array<StmtDoc> PrintBindingBlock(const relax::BindingBlock& n, const ObjectPath& n_p,
                                 const IRDocsifier& d, Array<ExprDoc>* non_dataflow_vars) {
  const Array<relax::Binding>& bindings = n->bindings;
  ObjectPath bindings_p = n_p->Attr("bindings");
  Array<StmtDoc> stmts;
  for (int i = 0, l = bindings.size(); i < l; ++i) {
    const relax::Binding& binding = bindings[i];
    ObjectPath binding_p = bindings_p->ArrayIndex(i);
    ICHECK(binding->var.defined());
    Doc binding_doc = d->AsDoc(binding, binding_p);
    if (const auto* stmt = binding_doc.as<StmtDocNode>()) {
      stmts.push_back(GetRef<StmtDoc>(stmt));
    } else if (const auto* stmt_block = binding_doc.as<StmtBlockDocNode>()) {
      stmts.insert(stmts.end(), stmt_block->stmts.begin(), stmt_block->stmts.end());
    } else {
      LOG(FATAL) << "TypeError: Unknown type: " << binding_doc->GetTypeKey();
    }
    if (non_dataflow_vars != nullptr && !binding->var->IsInstance<relax::DataflowVarNode>()) {
      non_dataflow_vars->push_back(d->AsDoc<ExprDoc>(binding->var, binding_p->Attr("var")));
    }
  }
  return stmts;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::BindingBlock>(  //
        "", [](relax::BindingBlock n, ObjectPath n_p, IRDocsifier d) -> Doc {
          return StmtBlockDoc(PrintBindingBlock(n, n_p, d, nullptr));
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::DataflowBlock>(  //
        "", [](relax::DataflowBlock n, ObjectPath n_p, IRDocsifier d) -> Doc {
          Array<ExprDoc> non_dataflow_vars;
          Array<StmtDoc> stmts = PrintBindingBlock(n, n_p, d, &non_dataflow_vars);
          stmts.push_back(ExprStmtDoc(Relax(d, "output")->Call(non_dataflow_vars)));
          return ScopeDoc(NullOpt, Relax(d, "dataflow")->Call({}), stmts);
        });

TVM_SCRIPT_REPR(relax::SeqExprNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::BindingBlockNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::DataflowBlockNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
