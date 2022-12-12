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

#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier.h>

#include <utility>

namespace tvm {
namespace script {
namespace printer {

template <typename DocType, typename NodeType>
Array<DocType> AsDocArray(const TracedArray<NodeType>& refs, const IRDocsifier& ir_docsifier) {
  Array<DocType> result;
  for (auto ref : refs) {
    result.push_back(ir_docsifier->AsExprDoc(ref));
  }
  return result;
}

template <typename DocType, typename NodeType>
Array<DocType> AsDocArray(std::initializer_list<NodeType>&& refs, const IRDocsifier& ir_docsifier) {
  Array<DocType> result;
  for (auto& ref : refs) {
    result.push_back(ir_docsifier->AsExprDoc(ref));
  }
  return result;
}

template <typename RefType>
Array<ExprDoc> AsExprDocArray(const TracedArray<RefType>& refs, const IRDocsifier& ir_docsifier) {
  return AsDocArray<ExprDoc>(refs, ir_docsifier);
}

template <typename RefType>
Array<ExprDoc> AsExprDocArray(std::initializer_list<RefType>&& refs,
                              const IRDocsifier& ir_docsifier) {
  return AsDocArray<ExprDoc>(std::move(refs), ir_docsifier);
}

inline DictDoc AsDictDoc(const TracedMap<String, ObjectRef>& dict,
                         const IRDocsifier& ir_docsifier) {
  Array<ExprDoc> keys;
  Array<ExprDoc> values;

  for (auto p : dict) {
    keys.push_back(LiteralDoc::Str(p.first));
    values.push_back(ir_docsifier->AsExprDoc(p.second));
  }

  auto doc = DictDoc(keys, values);
  doc->source_paths.push_back(dict.GetPath());
  return doc;
}

template <typename T>
inline ListDoc AsListDoc(const TracedArray<T>& arr, const IRDocsifier& ir_docsifier) {
  auto ret = ListDoc(AsExprDocArray(arr, ir_docsifier));
  ret->source_paths.push_back(arr.GetPath());
  return ret;
}

template <typename T>
inline TupleDoc AsTupleDoc(const TracedArray<T>& arr, const IRDocsifier& ir_docsifier) {
  auto ret = TupleDoc(AsExprDocArray(arr, ir_docsifier));
  ret->source_paths.push_back(arr.GetPath());
  return ret;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_UTILS_H_
