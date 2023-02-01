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
#ifndef TVM_SCRIPT_PRINTER_UTILS_H_
#define TVM_SCRIPT_PRINTER_UTILS_H_

#include <tvm/script/printer/ir_docsifier.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

#define TVM_SCRIPT_REPR(ObjectType, Method)                   \
  TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)                  \
      .set_dispatch<ObjectType>(RedirectedReprPrinterMethod); \
  TVM_STATIC_IR_FUNCTOR(TVMScriptPrinter, vtable).set_dispatch<ObjectType>(Method);

inline void RedirectedReprPrinterMethod(const ObjectRef& obj, ReprPrinter* p) {
  try {
    p->stream << TVMScriptPrinter::Script(obj, NullOpt);
  } catch (const tvm::Error& e) {
    LOG(WARNING) << "TVMScript printer falls back to the legacy ReprPrinter with the error:\n"
                 << e.what();
    p->stream << AsLegacyRepr(obj);
  }
}

inline std::string Docsify(const ObjectRef& obj, const IRDocsifier& d, const Frame& f,
                           const PrinterConfig& cfg) {
  Doc doc = d->AsDoc(obj, ObjectPath::Root());
  if (const auto* expr_doc = doc.as<ExprDocNode>()) {
    if (!cfg->verbose_expr) {
      f->stmts.clear();
    }
    f->stmts.push_back(ExprStmtDoc(GetRef<ExprDoc>(expr_doc)));
  } else if (const auto* stmt_doc = doc.as<StmtDocNode>()) {
    f->stmts.push_back(GetRef<StmtDoc>(stmt_doc));
  } else if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
    for (const StmtDoc& d : stmt_block->stmts) {
      f->stmts.push_back(d);
    }
  } else {
    LOG(FATAL) << "TypeError: Unexpected doc type: " << doc->GetTypeKey();
  }
  return DocToPythonScript(StmtBlockDoc(f->stmts), cfg);
}

inline std::string DType2Str(const runtime::DataType& dtype) {
  return dtype.is_void() ? "void" : runtime::DLDataType2String(dtype);
}

/*! \brief Add headers as comments to doc if needed */
inline Doc HeaderWrapper(const IRDocsifier& d, const Doc& doc) {
  if (d->ir_usage.size()) {
    Array<StmtDoc> stmts;
    if (d->ir_usage.count("ir")) {
      stmts.push_back(CommentDoc("from tvm.script import ir as " + d->cfg->ir_prefix));
    }
    if (d->ir_usage.count("tir")) {
      stmts.push_back(CommentDoc("from tvm.script import tir as " + d->cfg->tir_prefix));
    }
    if (d->ir_usage.count("relax")) {
      stmts.push_back(CommentDoc("from tvm.script import relax as " + d->cfg->relax_prefix));
    }
    stmts.push_back(CommentDoc(""));
    stmts.push_back(Downcast<StmtDoc>(doc));
    return StmtBlockDoc(stmts);
  }
  return doc;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_UTILS_H_
