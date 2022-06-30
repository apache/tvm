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
#ifndef TVM_SCRIPT_PRINTER_DOC_PRINTER_H_
#define TVM_SCRIPT_PRINTER_DOC_PRINTER_H_

#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

struct DocPrinterOptions {
  int indent_spaces = 4;
};

class DocPrinter {
 public:
  explicit DocPrinter(const DocPrinterOptions& options);
  virtual ~DocPrinter() = default;

  void Append(const Doc& doc);
  String GetString() const;

 protected:
  void PrintDoc(const Doc& doc);

  virtual void PrintTypedDoc(const LiteralDoc& doc) = 0;

  using OutputStream = std::ostringstream;

  void IncreaseIndent() { indent_ += options_.indent_spaces; }

  void DecreaseIndent() { indent_ -= options_.indent_spaces; }

  OutputStream& NewLine() {
    output_ << "\n";
    output_ << std::string(indent_, ' ');
    return output_;
  }

  OutputStream output_;

 private:
  DocPrinterOptions options_;
  int indent_ = 0;
};

std::unique_ptr<DocPrinter> GetPythonDocPrinter(const DocPrinterOptions& options);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif
