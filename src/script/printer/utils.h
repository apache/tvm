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

#include <tvm/node/serialization.h>
#include <tvm/script/printer/ir_docsifier.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/str_escape.h"

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
    if (ReprLegacyPrinter::CanDispatch(obj)) {
      LOG(WARNING) << "TVMScript printer falls back to the legacy ReprPrinter with the error:\n"
                   << e.what();
      try {
        p->stream << AsLegacyRepr(obj);
      } catch (const tvm::Error& e) {
        LOG(WARNING) << "AsLegacyRepr fails. Falling back to the basic address printer";
      }
    } else {
      LOG(WARNING) << "TVMScript printer falls back to the basic address printer with the error:\n"
                   << e.what();
    }
    p->stream << obj->GetTypeKey() << '(' << obj.get() << ')';
  }
}

inline std::string Docsify(const ObjectRef& obj, const IRDocsifier& d, const Frame& f,
                           const PrinterConfig& cfg) {
  Doc doc = d->AsDoc(obj, ObjectPath::Root());
  bool move_source_paths = false;
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
    move_source_paths = true;
  } else {
    LOG(FATAL) << "TypeError: Unexpected doc type: " << doc->GetTypeKey();
  }
  std::ostringstream os;
  if (!d->metadata.empty()) {
    if (d->cfg->show_meta) {
      os << "metadata = tvm.ir.load_json(\"\"\""
         << support::StrEscape(
                SaveJSON(Map<String, ObjectRef>(d->metadata.begin(), d->metadata.end())), false,
                false)
         << "\"\"\")\n";
    } else {
      f->stmts.push_back(
          CommentDoc("Metadata omitted. Use show_meta=True in script() method to show it."));
    }
  }
  if (move_source_paths) {
    StmtBlockDoc new_doc(f->stmts);
    new_doc->source_paths = std::move(doc->source_paths);
    os << DocToPythonScript(new_doc, cfg);
  } else {
    os << DocToPythonScript(StmtBlockDoc(f->stmts), cfg);
  }
  return os.str();
}

/*! \brief Creates the IR common prefix, which is by default `I` */
inline ExprDoc IR(const IRDocsifier& d, const String& attr) {
  d->ir_usage.insert("ir");
  return IdDoc(d->cfg->ir_prefix)->Attr(attr);
}

/*! \brief Creates the TIR common prefix, which is by default `T` */
inline ExprDoc TIR(const IRDocsifier& d, const String& attr) {
  d->ir_usage.insert("tir");
  return IdDoc(d->cfg->tir_prefix)->Attr(attr);
}

/*! \brief Creates the Relax common prefix, which is by default `R` */
inline ExprDoc Relax(const IRDocsifier& d, const String& attr) {
  d->ir_usage.insert("relax");
  return IdDoc(d->cfg->relax_prefix)->Attr(attr);
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

/*! \brief Check if a string has multiple lines. */
inline bool HasMultipleLines(const std::string& str) {
  return str.find_first_of('\n') != std::string::npos;
}

inline Optional<String> GetBindingName(const IRDocsifier& d) {
  return d->cfg->binding_names.empty() ? Optional<String>(NullOpt) : d->cfg->binding_names.back();
}

inline Optional<String> FindFunctionName(const IRDocsifier& d, const BaseFunc& f) {
  if (Optional<String> name = GetBindingName(d)) {
    return name.value();
  }
  if (Optional<String> sym = f->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    return sym.value();
  }
  return NullOpt;
}

inline String GenerateUniqueName(std::string name_hint,
                                 const std::unordered_set<String>& defined_names) {
  for (char& c : name_hint) {
    if (c != '_' && !std::isalnum(c)) {
      c = '_';
    }
  }
  std::string name = name_hint;
  for (int i = 1; defined_names.count(name) > 0; ++i) {
    name = name_hint + "_" + std::to_string(i);
  }
  return name;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_UTILS_H_
