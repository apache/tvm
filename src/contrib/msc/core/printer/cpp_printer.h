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
 * \file src/contrib/msc/core/printer/cpp_printer.h
 * \brief Cpp Printer.
 */

#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_CPP_PRINTER_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_CPP_PRINTER_H_

#include <string>
#include <vector>

#include "../utils.h"
#include "msc_base_printer.h"
#include "print_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief CppPrinter change list of docs to cpp format
 * \sa Doc
 */
class CppPrinter : public MSCBasePrinter {
 public:
  /*!
   * \brief The constructor of PythonPrinter
   * \param options the options for printer.
   */
  explicit CppPrinter(const std::string& options = "") : MSCBasePrinter(options) {
    endlines_.push_back(true);
  }

 protected:
  /*! * \brief Print a LiteralDoc to cpp format*/
  void PrintTypedDoc(const LiteralDoc& doc) final;

  /*! * \brief Print a IndexDoc to cpp format*/
  void PrintTypedDoc(const IndexDoc& doc) final;

  /*! * \brief Print a AttrAccessDoc to cpp format*/
  void PrintTypedDoc(const AttrAccessDoc& doc) final;

  /*! * \brief Print a CallDoc to cpp format*/
  void PrintTypedDoc(const CallDoc& doc) final;

  /*! * \brief Print a AssignDoc to cpp format*/
  void PrintTypedDoc(const AssignDoc& doc) final;

  /*! * \brief Print a IfDoc to cpp format*/
  void PrintTypedDoc(const IfDoc& doc) final;

  /*! * \brief Print a WhileDoc to cpp format*/
  void PrintTypedDoc(const WhileDoc& doc) final;

  /*! * \brief Print a ForDoc to cpp format*/
  void PrintTypedDoc(const ForDoc& doc) final;

  /*! * \brief Print a ScopeDoc to cpp format*/
  void PrintTypedDoc(const ScopeDoc& doc) final;

  /*! * \brief Print a FunctionDoc to cpp format*/
  void PrintTypedDoc(const FunctionDoc& doc) final;

  /*! * \brief Print a ClassDoc to cpp format*/
  void PrintTypedDoc(const ClassDoc& doc) final;

  /*! * \brief Print a CommentDoc to cpp format*/
  void PrintTypedDoc(const CommentDoc& doc) final;

  /*! * \brief Print a DeclareDoc to cpp format*/
  void PrintTypedDoc(const DeclareDoc& doc) final;

  /*! * \brief Print a PointerDoc to cpp format*/
  void PrintTypedDoc(const PointerDoc& doc) final;

  /*! * \brief Print a StrictListDoc to cpp format*/
  void PrintTypedDoc(const StrictListDoc& doc) final;

  /*! * \brief Print a StructDoc to cpp format*/
  void PrintTypedDoc(const StructDoc& doc) final;

  /*! * \brief Print a ConstructorDoc to cpp format*/
  void PrintTypedDoc(const ConstructorDoc& doc) final;

  /*! * \brief Print a LambdaDoc to cpp format*/
  void PrintTypedDoc(const LambdaDoc& doc) final;

  /*! * \brief Print a SwitchDoc to cpp format*/
  void PrintTypedDoc(const SwitchDoc& doc) final;

 private:
  /*! \brief endline scopes*/
  std::vector<bool> endlines_;

  /*! \brief Enter a endline scope*/
  void EnterEndlineScope(bool endline = false) { endlines_.push_back(endline); }

  /*! \brief Exit a endline scope*/
  void ExitEndlineScope() {
    ICHECK(endlines_.size() > 1) << "No endline scope found";
    endlines_.pop_back();
  }

  /*! \brief enable enbline*/
  void EnableEndline() {
    ICHECK(endlines_.size() > 0) << "No endline scope found";
    endlines_[endlines_.size() - 1] = true;
  }

  /*! \brief disable enbline*/
  void DisableEndline() {
    ICHECK(endlines_.size() > 0) << "No endline scope found";
    endlines_[endlines_.size() - 1] = false;
  }

  /*! \brief Print endline*/
  void Endline() {
    ICHECK(endlines_.size() > 0) << "No endline scope found";
    if (endlines_[endlines_.size() - 1]) {
      output_ << ";";
    }
  }

  /*! \brief Check if the doc is empty doc*/
  bool IsEmptyDoc(const ExprDoc& doc);

  /*! \brief Print block with indent*/
  void PrintIndentedBlock(const Array<StmtDoc>& docs);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_CPP_PRINTER_H_
