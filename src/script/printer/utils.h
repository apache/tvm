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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/printer.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/str_escape.h"

namespace tvm {
namespace script {
namespace printer {

// Note: the `TVM_SCRIPT_REPR` macro is intentionally duplicated in each
// dialect-local `src/<dialect>/script/printer/utils.h`. Keeping a single
// definition here would force the dialect headers to depend on this shared
// header, which the per-dialect restructure aims to avoid for cross-directory
// references. See each `<dialect>/script/printer/utils.h` for the macro.

inline std::string Docsify(const ffi::ObjectRef& obj, const IRDocsifier& d, const Frame& f,
                           const PrinterConfig& cfg) {
  Doc doc = d->AsDoc(obj, AccessPath::Root());
  bool move_source_paths = false;
  if (const auto* expr_doc = doc.as<ExprDocNode>()) {
    if (!cfg->verbose_expr) {
      f->stmts.clear();
    }
    f->stmts.push_back(ExprStmtDoc(ffi::GetRef<ExprDoc>(expr_doc)));
  } else if (const auto* stmt_doc = doc.as<StmtDocNode>()) {
    f->stmts.push_back(ffi::GetRef<StmtDoc>(stmt_doc));
  } else if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
    for (const StmtDoc& d : stmt_block->stmts) {
      f->stmts.push_back(d);
    }
    move_source_paths = true;
  } else {
    TVM_FFI_THROW(TypeError) << "Unexpected doc type: " << doc->GetTypeKey();
  }
  std::ostringstream os;
  if (d->ir_usage.count("future_annotations")) {
    os << "from __future__ import annotations\n\n";
  }
  if (!d->metadata.empty()) {
    if (d->cfg->show_meta) {
      os << "metadata = tvm.ir.load_json(\"\"\""
         << support::StrEscape(
                ffi::json::Stringify(
                    ffi::ToJSONGraph(
                        ffi::Map<ffi::String, ffi::Any>(d->metadata.begin(), d->metadata.end()),
                        ffi::json::Object{{"tvm_version", TVM_VERSION}}),
                    2),
                false, false)
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
inline ExprDoc IR(const IRDocsifier& d, const ffi::String& attr) {
  d->ir_usage.insert("ir");
  return IdDoc(d->cfg->ir_prefix)->Attr(attr);
}

/*! \brief Creates the TIR common prefix, which is by default `T` */
inline ExprDoc TIR(const IRDocsifier& d, const ffi::String& attr) {
  d->ir_usage.insert("tirx");
  return IdDoc(d->cfg->GetExtraConfig<ffi::String>("tirx.prefix", "T"))->Attr(attr);
}

/*! \brief Alias for TIR — historical TIRx name used by tirx printer code */
inline ExprDoc TIRx(const IRDocsifier& d, const ffi::String& attr) { return TIR(d, attr); }

/*! \brief Creates the Relax common prefix, which is by default `R` */
inline ExprDoc Relax(const IRDocsifier& d, const ffi::String& attr) {
  d->ir_usage.insert("relax");
  return IdDoc(d->cfg->GetExtraConfig<ffi::String>("relax.prefix", "R"))->Attr(attr);
}

inline std::string DType2Str(DLDataType dtype) {
  return (((dtype).code == kDLOpaqueHandle) && ((dtype).bits == 0) && ((dtype).lanes == 0))
             ? "void"
             : ffi::DLDataTypeToString(dtype);
}

/*! \brief Add headers as comments to doc if needed */
inline Doc HeaderWrapper(const IRDocsifier& d, const Doc& doc) {
  if (d->ir_usage.size()) {
    ffi::Array<StmtDoc> stmts;
    if (d->ir_usage.count("type_var")) {
      stmts.push_back(CommentDoc("from typing import TypeVar"));
    }
    if (d->ir_usage.count("ir")) {
      stmts.push_back(CommentDoc("from tvm.script import ir as " + d->cfg->ir_prefix));
    }
    if (d->ir_usage.count("tirx")) {
      stmts.push_back(CommentDoc("from tvm.script import tirx as " +
                                 d->cfg->GetExtraConfig<ffi::String>("tirx.prefix", "T")));
      // Layout sugar like `4 @ Axis.laneid` references registered axes via the
      // `Axis` class attribute. Mirror the `Axis` injection in `_default_globals`
      // so readers see the dependency. Decorative only.
      stmts.push_back(CommentDoc("from tvm.tirx.layout import Axis"));
    }
    if (d->ir_usage.count("relax")) {
      stmts.push_back(CommentDoc("from tvm.script import relax as " +
                                 d->cfg->GetExtraConfig<ffi::String>("relax.prefix", "R")));
    }
    stmts.push_back(CommentDoc(""));
    if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
      stmts.insert(stmts.end(), stmt_block->stmts.begin(), stmt_block->stmts.end());
    } else {
      stmts.push_back(doc.as_or_throw<StmtDoc>());
    }
    return StmtBlockDoc(stmts);
  }
  return doc;
}

inline std::vector<std::pair<std::string, ExprDoc>> DefineTypeVarDocs(
    const std::unordered_set<const VarNode*>& type_vars, const Frame& frame, const IRDocsifier& d) {
  std::vector<std::pair<std::string, ExprDoc>> type_var_docs;
  type_var_docs.reserve(type_vars.size());
  for (const VarNode* var_node : type_vars) {
    Var var = ffi::GetRef<Var>(var_node);
    ffi::Optional<ExprDoc> existing_doc = d->GetVarDoc(var);
    ExprDoc var_doc = existing_doc.has_value()
                          ? existing_doc.value()
                          : d->Define(var, frame, var->name.empty() ? "v" : var->name);
    const auto* id_doc = var_doc.as<IdDocNode>();
    TVM_FFI_ICHECK(id_doc != nullptr);
    type_var_docs.emplace_back(id_doc->name, var_doc);
  }
  std::sort(type_var_docs.begin(), type_var_docs.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  return type_var_docs;
}

inline bool UsePEP695TypeVars(const IRDocsifier& d) {
  return d->cfg->GetExtraConfig<bool>("script.use_pep695",
                                      d->cfg->GetExtraConfig<bool>("relax.use_pep695", false));
}

inline Doc WrapFunctionDocWithTypeVars(
    const IRDocsifier& d, FunctionDoc function_doc,
    const std::vector<std::pair<std::string, ExprDoc>>& type_var_docs) {
  if (type_var_docs.empty()) {
    return HeaderWrapper(d, function_doc);
  }
  if (UsePEP695TypeVars(d)) {
    d->ir_usage.insert("future_annotations");
    for (const auto& type_var_doc : type_var_docs) {
      function_doc->type_params.push_back(type_var_doc.second);
    }
    return HeaderWrapper(d, function_doc);
  }

  d->ir_usage.insert("type_var");
  ffi::Array<StmtDoc> stmts;
  for (const auto& [name, var_doc] : type_var_docs) {
    stmts.push_back(AssignDoc(
        var_doc, IdDoc("TypeVar")->Call({LiteralDoc::Str(name, ffi::Optional<AccessPath>())}),
        std::nullopt));
  }
  stmts.push_back(function_doc);
  return HeaderWrapper(d, StmtBlockDoc(stmts));
}

/*! \brief Check if a string has multiple lines. */
inline bool HasMultipleLines(const std::string& str) {
  return str.find_first_of('\n') != std::string::npos;
}

inline ffi::Optional<ffi::String> GetBindingName(const IRDocsifier& d) {
  return d->cfg->binding_names.empty() ? ffi::Optional<ffi::String>(std::nullopt)
                                       : d->cfg->binding_names.back();
}

inline ffi::Optional<ffi::String> FindFunctionName(const IRDocsifier& d, const BaseFunc& f) {
  if (ffi::Optional<ffi::String> name = GetBindingName(d)) {
    return name.value();
  }
  if (ffi::Optional<ffi::String> sym = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
    return sym.value();
  }
  return std::nullopt;
}

inline ffi::String GenerateUniqueName(std::string name_hint,
                                      const std::unordered_set<ffi::String>& defined_names) {
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
