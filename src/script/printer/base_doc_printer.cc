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

#include "./base_doc_printer.h"

namespace tvm {
namespace script {
namespace printer {

DocPrinter::DocPrinter(int indent_spaces) : indent_spaces_(indent_spaces) {}

void DocPrinter::Append(const Doc& doc) { PrintDoc(doc); }

String DocPrinter::GetString() const {
  std::string text = output_.str();
  if (!text.empty() && text.back() != '\n') {
    text.push_back('\n');
  }
  return text;
}

void DocPrinter::PrintDoc(const Doc& doc) {
  if (const auto* doc_node = doc.as<LiteralDocNode>()) {
    PrintTypedDoc(GetRef<LiteralDoc>(doc_node));
  } else {
    LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    throw;
  }
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
