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
#ifndef TVM_SCRIPT_PRINTER_DOC_PRINTER_BASE_DOC_PRINTER_H_
#define TVM_SCRIPT_PRINTER_DOC_PRINTER_BASE_DOC_PRINTER_H_

#include <tvm/script/printer/doc.h>

#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

/*! \brief Range of byte offsets in a string */
using ByteSpan = std::pair<size_t, size_t>;

/*!
 * \brief DocPrinter is responsible for printing Doc tree into text format
 * \details This is the base class for translating Doc into string.
 *          Each target language needs to have its subclass of DocPrinter
 *          to define the actual logic of printing Doc.
 *
 * \sa Doc
 */
class DocPrinter {
 public:
  /*!
   * \brief The constructor of DocPrinter
   *
   * \param options the option for printer
   */
  explicit DocPrinter(const PrinterConfig& options);

  virtual ~DocPrinter() = default;

  /*!
   * \brief Append a doc into the final content
   *
   * \param doc the Doc to be printed
   *
   * \sa GetString
   */
  void Append(const Doc& doc);

  /*!
   * \brief Append a doc to the final content
   *
   * \param doc  Doc to be printed
   * \param path_to_underline  Object path to be underlined
   *
   * \sa GetString
   */
  void Append(const Doc& doc, const PrinterConfig& cfg);

  /*!
   * \brief Get the printed string of all Doc appended
   *
   * The content of each Doc in the returned string will
   * appear in the same order as they are appended.
   *
   * \sa Append
   */
  String GetString() const;

 protected:
  /*!
   * \brief Get the printed string
   *
   * It will dispatch to the PrintTypedDoc method based on
   * the actual type of Doc.
   *
   * \sa PrintTypedDoc
   */
  void PrintDoc(const Doc& doc);

  /*!
   * \brief Virtual method to print a LiteralDoc
   */
  virtual void PrintTypedDoc(const LiteralDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an IdDoc
   */
  virtual void PrintTypedDoc(const IdDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an AttrAccessDoc
   */
  virtual void PrintTypedDoc(const AttrAccessDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an IndexDoc
   */
  virtual void PrintTypedDoc(const IndexDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an OperationDoc
   */
  virtual void PrintTypedDoc(const OperationDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a CallDoc
   */
  virtual void PrintTypedDoc(const CallDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a LambdaDoc
   */
  virtual void PrintTypedDoc(const LambdaDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ListDoc
   */
  virtual void PrintTypedDoc(const ListDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a TupleDoc
   */
  virtual void PrintTypedDoc(const TupleDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a DictDoc
   */
  virtual void PrintTypedDoc(const DictDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a SliceDoc
   */
  virtual void PrintTypedDoc(const SliceDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a StmtBlockDoc
   */
  virtual void PrintTypedDoc(const StmtBlockDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an AssignDoc
   */
  virtual void PrintTypedDoc(const AssignDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an IfDoc
   */
  virtual void PrintTypedDoc(const IfDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a WhileDoc
   */
  virtual void PrintTypedDoc(const WhileDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ForDoc
   */
  virtual void PrintTypedDoc(const ForDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ScopeDoc
   */
  virtual void PrintTypedDoc(const ScopeDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an ExprStmtDoc
   */
  virtual void PrintTypedDoc(const ExprStmtDoc& doc) = 0;

  /*!
   * \brief Virtual method to print an AssertDoc
   */
  virtual void PrintTypedDoc(const AssertDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ReturnDoc
   */
  virtual void PrintTypedDoc(const ReturnDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a FunctionDoc
   */
  virtual void PrintTypedDoc(const FunctionDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a ClassDoc
   */
  virtual void PrintTypedDoc(const ClassDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a CommentDoc
   */
  virtual void PrintTypedDoc(const CommentDoc& doc) = 0;

  /*!
   * \brief Virtual method to print a DocStringDoc
   */
  virtual void PrintTypedDoc(const DocStringDoc& doc) = 0;

  /*!
   * \brief Increase the indent level of any content to be
   *        printed after this call
   */
  void IncreaseIndent() { indent_ += options_->indent_spaces; }

  /*!
   * \brief Decrease the indent level of any content to be
   *        printed after this call
   */
  void DecreaseIndent() { indent_ -= options_->indent_spaces; }

  /*!
   * \brief Add a new line into the output stream
   *
   * \sa output_
   */
  std::ostream& NewLine() {
    size_t start_pos = output_.tellp();
    output_ << "\n";
    line_starts_.push_back(output_.tellp());
    output_ << std::string(indent_, ' ');
    size_t end_pos = output_.tellp();
    underlines_exempted_.push_back({start_pos, end_pos});
    return output_;
  }

  /*!
   * \brief The output stream of printer
   *
   * All printed content will be stored in this stream and returned
   * when GetString is called.
   *
   * \sa GetString
   */
  std::ostringstream output_;

  /*! \brief Spans that we have already committed to underline exemption. */
  std::vector<ByteSpan> underlines_exempted_;

 private:
  void MarkSpan(const ByteSpan& span, const ObjectPath& path);

  /*! \brief Options to customize certain aspects of the output */
  PrinterConfig options_;

  /*! \brief the current level of indent */
  int indent_ = 0;

  /*! \brief For each line in the output_, byte offset of its first character */
  std::vector<size_t> line_starts_;

  /*! \brief Path of the object that we would like to underline */
  Array<ObjectPath> path_to_underline_;

  /*!
   * \brief Candidate spans to be underlined, until we find a better match.
   * (A better match is an object with a longer path that is still a prefix of path_to_underline_.)
   */
  std::vector<std::vector<ByteSpan>> current_underline_candidates_;

  /*! \brief Path length of the objects that are current candidates for underlining. */
  std::vector<int> current_max_path_length_;

  /*! \brief Spans that we have already committed to underline. */
  std::vector<ByteSpan> underlines_;
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_PRINTER_BASE_DOC_PRINTER_H_
