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
 * \file src/contrib/msc/core/printer/print_utils.cc
 */
#include "print_utils.h"

#include <string>

namespace tvm {
namespace contrib {
namespace msc {

const ExprDoc DocUtils::ToDoc(int64_t val) { return LiteralDoc::Int(val, NullOpt); }

const ExprDoc DocUtils::ToDoc(int val) { return ToDoc(static_cast<int64_t>(val)); }

const ExprDoc DocUtils::ToDoc(size_t val) { return ToDoc(static_cast<int64_t>(val)); }

const ExprDoc DocUtils::ToDoc(const IntImm& val) { return ToDoc(val->value); }

const ExprDoc DocUtils::ToDoc(const Integer& val) { return ToDoc(val->value); }

const ExprDoc DocUtils::ToDoc(double val) { return LiteralDoc::Float(val, NullOpt); }

const ExprDoc DocUtils::ToDoc(float val) { return ToDoc(static_cast<double>(val)); }

const ExprDoc DocUtils::ToDoc(const FloatImm& val) { return ToDoc(val->value); }

const ExprDoc DocUtils::ToDoc(const char* val) { return IdDoc(std::string(val)); }

const ExprDoc DocUtils::ToDoc(const String& val) { return IdDoc(val); }

const ExprDoc DocUtils::ToDoc(bool val) { return LiteralDoc::Boolean(val, NullOpt); }

const ExprDoc DocUtils::ToDoc(const ExprDoc& val) { return val; }

const ExprDoc DocUtils::ToStrDoc(const String& val) { return LiteralDoc::Str(val, NullOpt); }

const PointerDoc DocUtils::ToPtrDoc(const String& val) { return PointerDoc(val); }

const DeclareDoc DocUtils::ToDeclareDoc(const String& type, const String& variable, size_t len,
                                        bool use_constructor) {
  Optional<ExprDoc> type_doc;
  if (type.size() == 0) {
    type_doc = NullOpt;
  } else {
    type_doc = IdDoc(type);
  }
  if (len == 0) {
    return DeclareDoc(type_doc, IdDoc(variable), Array<ExprDoc>(), use_constructor);
  }
  Array<Doc> doc_indices{DocUtils::ToDoc(len)};
  return DeclareDoc(type_doc, IndexDoc(IdDoc(variable), doc_indices), Array<ExprDoc>(),
                    use_constructor);
}

const AttrAccessDoc DocUtils::ToAttrAccessDoc(const String& value, const String& name) {
  return AttrAccessDoc(IdDoc(value), name);
}

const Array<StmtDoc> DocUtils::ToStmts(const Array<Doc>& docs) {
  Array<StmtDoc> stmts;
  for (const auto& d : docs) {
    if (d->IsInstance<StmtDocNode>()) {
      stmts.push_back(Downcast<StmtDoc>(d));
    } else if (d->IsInstance<ExprDocNode>()) {
      stmts.push_back(ExprStmtDoc(Downcast<ExprDoc>(d)));
    } else {
      LOG(FATAL) << "Unecpected doc type " << d->GetTypeKey();
    }
  }
  return stmts;
}

const StmtBlockDoc DocUtils::ToStmtBlock(const Array<Doc>& docs) {
  return StmtBlockDoc(ToStmts(docs));
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
