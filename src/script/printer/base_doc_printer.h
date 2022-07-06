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
#ifndef TVM_SCRIPT_PRINTER_BASE_DOC_PRINTER_H_
#define TVM_SCRIPT_PRINTER_BASE_DOC_PRINTER_H_

#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/doc_printer.h>

#include <memory>
#include <ostream>
#include <string>

namespace tvm {
namespace script {
namespace printer {

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
  explicit DocPrinter(int indent_spaces = 4);
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
   * \brief Increase the indent level of any content to be
   *        printed after this call
   */
  void IncreaseIndent() { indent_ += indent_spaces_; }

  /*!
   * \brief Decrease the indent level of any content to be
   *        printed after this call
   */
  void DecreaseIndent() { indent_ -= indent_spaces_; }

  /*!
   * \brief Add a new line into the output stream
   *
   * \sa output_
   */
  std::ostream& NewLine() {
    output_ << "\n";
    output_ << std::string(indent_, ' ');
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

 private:
  /*! \brief the number of spaces for one level of indentation */
  int indent_spaces_ = 4;

  /*! \brief the current level of indent */
  int indent_ = 0;
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_BASE_DOC_PRINTER_H_
