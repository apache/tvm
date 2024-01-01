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
 * \file src/contrib/msc/core/printer/python_printer.h
 * \brief Python Printer.
 */

#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_PYTHON_PRINTER_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_PYTHON_PRINTER_H_

#include <string>

#include "msc_base_printer.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief PythonPrinter change list of docs to python format
 * \sa Doc
 */
class PythonPrinter : public MSCBasePrinter {
 public:
  /*!
   * \brief The constructor of PythonPrinter
   * \param options the options for printer.
   */
  explicit PythonPrinter(const std::string& options = "") : MSCBasePrinter(options) {}

 protected:
  /*! * \brief Print a LiteralDoc to python format*/
  void PrintTypedDoc(const LiteralDoc& doc) final;

  /*! * \brief Print a AttrAccessDoc to python format*/
  void PrintTypedDoc(const AttrAccessDoc& doc) final;

  /*! * \brief Print a IndexDoc to python format*/
  void PrintTypedDoc(const IndexDoc& doc) final;

  /*! * \brief Print a CallDoc to python format*/
  void PrintTypedDoc(const CallDoc& doc) final;

  /*! * \brief Print a AssignDoc to python format*/
  void PrintTypedDoc(const AssignDoc& doc) final;

  /*! * \brief Print a IfDoc to python format*/
  void PrintTypedDoc(const IfDoc& doc) final;

  /*! * \brief Print a ForDoc to python format*/
  void PrintTypedDoc(const ForDoc& doc) final;

  /*! * \brief Print a ScopeDoc to python format*/
  void PrintTypedDoc(const ScopeDoc& doc) final;

  /*! * \brief Print a FunctionDoc to python format*/
  void PrintTypedDoc(const FunctionDoc& doc) final;

  /*! * \brief Print a ClassDoc to python format*/
  void PrintTypedDoc(const ClassDoc& doc) final;

  /*! * \brief Print a CommentDoc to python format*/
  void PrintTypedDoc(const CommentDoc& doc) final;

  /*! * \brief Print a StrictListDoc to python format*/
  void PrintTypedDoc(const StrictListDoc& doc) final;

  /*! * \brief Print a SwitchDoc to python format*/
  void PrintTypedDoc(const SwitchDoc& doc) final;

  /*! \brief Print comment for stmt in python format*/
  void MaybePrintComment(const StmtDoc& stmt, bool multi_lines = false) final;

 private:
  /*! \brief Print block with indent*/
  void PrintIndentedBlock(const Array<StmtDoc>& docs);

  /*! \brief Print decorators for function and class*/
  void PrintDecorators(const Array<ExprDoc>& decorators);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_PYTHON_PRINTER_H_
