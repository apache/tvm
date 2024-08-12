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
 * \file src/contrib/msc/core/printer/prototxt_printer.cc
 */

#include "prototxt_printer.h"

#include <utility>
#include <vector>

namespace tvm {
namespace contrib {
namespace msc {

LiteralDoc PrototxtPrinter::ToLiteralDoc(const ObjectRef& obj) {
  if (obj.as<StringObj>()) {
    return LiteralDoc::Str(Downcast<String>(obj), NullOpt);
  } else if (obj.as<IntImmNode>()) {
    return LiteralDoc::Int(Downcast<IntImm>(obj)->value, NullOpt);
  } else if (obj.as<FloatImmNode>()) {
    return LiteralDoc::Float(Downcast<FloatImm>(obj)->value, NullOpt);
  }
  std::ostringstream obj_des;
  obj_des << obj;
  return LiteralDoc::Str(obj_des.str(), NullOpt);
}

DictDoc PrototxtPrinter::ToDictDoc(const Map<String, ObjectRef>& dict) {
  Array<ExprDoc> keys;
  Array<ExprDoc> values;
  for (const auto& pair : dict) {
    keys.push_back(IdDoc(pair.first));
    if (pair.second.as<DictDocNode>()) {
      values.push_back(Downcast<DictDoc>(pair.second));
    } else {
      values.push_back(ToLiteralDoc(pair.second));
    }
  }
  return DictDoc(keys, values);
}

DictDoc PrototxtPrinter::ToDictDoc(const std::vector<std::pair<String, ObjectRef>>& dict) {
  Array<ExprDoc> keys;
  Array<ExprDoc> values;
  for (const auto& pair : dict) {
    keys.push_back(IdDoc(pair.first));
    if (pair.second.as<DictDocNode>()) {
      values.push_back(Downcast<DictDoc>(pair.second));
    } else {
      values.push_back(ToLiteralDoc(pair.second));
    }
  }
  return DictDoc(keys, values);
}

void PrototxtPrinter::Append(const Map<String, ObjectRef>& dict) {
  DictDoc doc = ToDictDoc(dict);
  PrintDoc(doc, false);
}

void PrototxtPrinter::Append(const std::vector<std::pair<String, ObjectRef>>& dict) {
  DictDoc doc = ToDictDoc(dict);
  PrintDoc(doc, false);
}

void PrototxtPrinter::AppendPair(const String& key, const ObjectRef& value) {
  Map<String, ObjectRef> dict;
  dict.Set(key, value);
  return Append(dict);
}

void PrototxtPrinter::PrintTypedDoc(const DictDoc& doc) {
  ICHECK_EQ(doc->keys.size(), doc->values.size())
      << "DictDoc should have equal number of elements in keys and values.";
  for (size_t i = 0; i < doc->keys.size(); i++) {
    ICHECK(doc->keys[i].as<IdDocNode>())
        << "Prototxt key should be IdDoc, get " << doc->keys[i]->GetTypeKey();
    PrintDoc(doc->keys[i]);
    if (doc->values[i].as<DictDocNode>()) {
      output_ << " {";
      IncreaseIndent();
      PrintDoc(doc->values[i], false);
      DecreaseIndent();
      NewLine() << "}";
    } else {
      output_ << ": ";
      PrintDoc(doc->values[i], false);
    }
  }
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
