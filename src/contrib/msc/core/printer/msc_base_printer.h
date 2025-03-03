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
 * \file src/contrib/msc/core/printer/msc_base_printer.h
 * \brief Base Printer for all MSC printers.
 */
#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_MSC_BASE_PRINTER_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_MSC_BASE_PRINTER_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include <string>

#include "../../../../../src/support/str_escape.h"
#include "msc_doc.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief MSCPrinterConfig is base for config class in MSC
 * \sa Doc
 */
struct MSCPrinterConfig {
  size_t indent{0};
  size_t float_precision{6};
  std::string indent_space{"  "};
  std::string separator{", "};
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "indent") {
        reader->Read(&indent);
      } else if (key == "float_precision") {
        reader->Read(&float_precision);
      } else if (key == "indent_space") {
        reader->Read(&indent_space);
      } else if (key == "separator") {
        reader->Read(&separator);
      } else {
        LOG(FATAL) << "Do not support config " << key << " in printer";
      }
    }
  }
};

/*!
 * \brief MSCBasePrinter is responsible for printing Doc tree into text format
 * \sa Doc
 */
class MSCBasePrinter {
 public:
  /*!
   * \brief The constructor of MSCBasePrinter
   * \param options the options for printer.
   */
  explicit MSCBasePrinter(const std::string& options = "") {
    if (options.size() > 0) {
      std::istringstream is(options);
      dmlc::JSONReader reader(&is);
      reader.Read(&config_);
    }
    indent_ = config_.indent;
  }

  virtual ~MSCBasePrinter() = default;

  /*!
   * \brief Append a doc into the final content
   * \sa GetString
   */
  void Append(const Doc& doc, bool new_line = true) { PrintDoc(doc, new_line); }

  /*!
   * \brief Get the printed string of all Doc appended
   * \sa Append
   */
  String GetString() const { return output_.str(); }

 protected:
  /*! \brief Print doc*/
  void PrintDoc(const Doc& doc, bool new_line = true);

  /*! \brief Virtual method to print a LiteralDoc*/
  virtual void PrintTypedDoc(const LiteralDoc& doc);

  /*! \brief Virtual method to print an IdDoc*/
  virtual void PrintTypedDoc(const IdDoc& doc);

  /*! \brief Virtual method to print a ListDoc*/
  virtual void PrintTypedDoc(const ListDoc& doc);

  /*! \brief Virtual method to print a TupleDoc*/
  virtual void PrintTypedDoc(const TupleDoc& doc);

  /*! \brief Virtual method to print a ReturnDoc*/
  virtual void PrintTypedDoc(const ReturnDoc& doc);

  /*! \brief Virtual method to print a StmtBlockDoc*/
  virtual void PrintTypedDoc(const StmtBlockDoc& doc);

  /*! \brief Virtual method to print a ExprStmtDoc*/
  virtual void PrintTypedDoc(const ExprStmtDoc& doc);

  /*! \brief Virtual method to print an IndexDoc*/
  virtual void PrintTypedDoc(const IndexDoc& doc) { LOG(FATAL) << "Index is not implemented"; }

  /*! \brief Virtual method to print a CallDoc*/
  virtual void PrintTypedDoc(const CallDoc& doc) { LOG(FATAL) << "Call is not implemented"; }

  /*! \brief Virtual method to print an AttrAccessDoc*/
  virtual void PrintTypedDoc(const AttrAccessDoc& doc) {
    LOG(FATAL) << "AttrAccess is not implemented";
  }

  /*! \brief Virtual method to print a DictDoc*/
  virtual void PrintTypedDoc(const DictDoc& doc) { LOG(FATAL) << "Dict is not implemented"; }

  /*! \brief Virtual method to print a SliceDoc*/
  virtual void PrintTypedDoc(const SliceDoc& doc) { LOG(FATAL) << "Slice is not implemented"; }

  /*! \brief Virtual method to print an AssignDoc*/
  virtual void PrintTypedDoc(const AssignDoc& doc) { LOG(FATAL) << "Assign is not implemented"; }

  /*! \brief Virtual method to print an IfDoc*/
  virtual void PrintTypedDoc(const IfDoc& doc) { LOG(FATAL) << "If is not implemented"; }

  /*! \brief Virtual method to print a WhileDoc*/
  virtual void PrintTypedDoc(const WhileDoc& doc) { LOG(FATAL) << "While is not implemented"; }

  /*! \brief Virtual method to print a ForDoc*/
  virtual void PrintTypedDoc(const ForDoc& doc) { LOG(FATAL) << "For is not implemented"; }

  /*! \brief Virtual method to print a ScopeDoc*/
  virtual void PrintTypedDoc(const ScopeDoc& doc) { LOG(FATAL) << "Scope is not implemented"; }

  /*! \brief Virtual method to print an AssertDoc*/
  virtual void PrintTypedDoc(const AssertDoc& doc) { LOG(FATAL) << "Assert is not implemented"; }

  /*! \brief Virtual method to print a FunctionDoc*/
  virtual void PrintTypedDoc(const FunctionDoc& doc) {
    LOG(FATAL) << "Function is not implemented";
  }

  /*! \brief Virtual method to print a ClassDoc*/
  virtual void PrintTypedDoc(const ClassDoc& doc) { LOG(FATAL) << "Class is not implemented"; }

  /*! \brief Virtual method to print a CommentDoc*/
  virtual void PrintTypedDoc(const CommentDoc& doc) { LOG(FATAL) << "Comment is not implemented"; }

  /*! \brief Virtual method to print a DeclareDoc*/
  virtual void PrintTypedDoc(const DeclareDoc& doc) { LOG(FATAL) << "Declare is not implemented"; }

  /*! \brief Virtual method to print a StrictListDoc*/
  virtual void PrintTypedDoc(const StrictListDoc& doc) {
    LOG(FATAL) << "StrictList is not implemented";
  }

  /*! \brief Virtual method to print a PointerDoc*/
  virtual void PrintTypedDoc(const PointerDoc& doc) {
    LOG(FATAL) << "PointerDoc is not implemented";
  }

  /*! \brief Virtual method to print a StructDoc*/
  virtual void PrintTypedDoc(const StructDoc& doc) { LOG(FATAL) << "StructDoc is not implemented"; }

  /*! \brief Virtual method to print a ConstructorDoc*/
  virtual void PrintTypedDoc(const ConstructorDoc& doc) {
    LOG(FATAL) << "ConstructorDoc is not implemented";
  }

  /*! \brief Virtual method to print a SwitchDoc*/
  virtual void PrintTypedDoc(const SwitchDoc& doc) { LOG(FATAL) << "SwitchDoc is not implemented"; }

  /*! \brief Virtual method to print a LambdaDoc*/
  virtual void PrintTypedDoc(const LambdaDoc& doc) { LOG(FATAL) << "LambdaDoc is not implemented"; }

  /*! \brief Print docs to joined doc */
  template <typename DocType>
  void PrintJoinedDocs(const Array<DocType>& docs, const String& separator = ", ") {
    for (size_t i = 0; i < docs.size(); i++) {
      PrintDoc(docs[i], false);
      output_ << (i == docs.size() - 1 ? "" : separator);
    }
  }

  /*! \brief Print comment for stmt*/
  virtual void MaybePrintComment(const StmtDoc& stmt, bool multi_lines = false);

  /*!
   * \brief Start line into the output stream.
   * \sa output_
   */
  std::ostream& NewLine(bool with_indent = true) {
    if (lines_ > 0) {
      output_ << "\n";
    }
    if (with_indent) {
      for (size_t i = 0; i < indent_; i++) {
        output_ << config_.indent_space;
      }
    }
    return output_;
  }

  /*! \brief Increase the indent level*/
  void IncreaseIndent() { indent_ += 1; }

  /*! \brief Decrease the indent level*/
  void DecreaseIndent() {
    if (indent_ >= 1) {
      indent_ -= 1;
    }
  }

  /*! \brief Get the output stream*/
  const MSCPrinterConfig config() { return config_; }

  /*! \brief The output stream of printer*/
  std::ostringstream output_;

 private:
  /*! \brief The current level of indent */
  size_t indent_ = 0;

  /*! \brief The lines num */
  size_t lines_ = 0;

  /*! \brief The config for printer */
  MSCPrinterConfig config_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_MSC_BASE_PRINTER_H_
