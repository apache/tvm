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
 * \file src/contrib/msc/core/printer/print_utils.h
 * \brief Common utilities for print.
 */
#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_PRINT_UTILS_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_PRINT_UTILS_H_

#include <tvm/script/printer/doc.h>

#include <string>
#include <vector>

#include "msc_doc.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief Symbols for Doc.
 */

class DocSymbol {
 public:
  /*! * \brief The empty symbol*/
  TVM_DLL static const String Empty();

  /*! * \brief The next line symbol*/
  TVM_DLL static const String NextLine();
};

/*!
 * \brief Utils for Doc.
 */
class DocUtils {
 public:
  /*!
   * \brief Change object to Doc.
   * \return The Doc.
   */
  TVM_DLL static const ExprDoc ToDoc(int val);
  TVM_DLL static const ExprDoc ToDoc(int64_t val);
  TVM_DLL static const ExprDoc ToDoc(size_t val);
  TVM_DLL static const ExprDoc ToDoc(const IntImm& val);
  TVM_DLL static const ExprDoc ToDoc(const Integer& val);
  TVM_DLL static const ExprDoc ToDoc(float val);
  TVM_DLL static const ExprDoc ToDoc(double val);
  TVM_DLL static const ExprDoc ToDoc(const FloatImm& val);
  TVM_DLL static const ExprDoc ToDoc(const char* val);
  TVM_DLL static const ExprDoc ToDoc(const String& val);
  TVM_DLL static const ExprDoc ToDoc(bool val);
  TVM_DLL static const ExprDoc ToDoc(const ExprDoc& val);
  TVM_DLL static const ExprDoc ToStr(const String& val);
  TVM_DLL static const PointerDoc ToPtr(const String& val);

  /*!
   * \brief Change object to DeclareDoc.
   * \return The DeclareDoc.
   */
  template <typename T>
  TVM_DLL static const DeclareDoc ToDeclare(const String& type, const T& variable, size_t len = 0,
                                            bool use_constructor = true) {
    Optional<ExprDoc> type_doc;
    if (type.size() == 0) {
      type_doc = NullOpt;
    } else {
      type_doc = IdDoc(type);
    }
    if (len == 0) {
      return DeclareDoc(type_doc, ToDoc(variable), Array<ExprDoc>(), use_constructor);
    }
    Array<Doc> doc_indices{DocUtils::ToDoc(len)};
    return DeclareDoc(type_doc, IndexDoc(ToDoc(variable), doc_indices), Array<ExprDoc>(),
                      use_constructor);
  }

  /*!
   * \brief Change object to AssignDoc.
   * \return The AssignDoc.
   */
  template <typename LT, typename RT>
  TVM_DLL static const AssignDoc ToAssign(const LT& lhs, const RT& rhs,
                                          const String& annotation = "") {
    if (annotation.size() == 0) {
      return AssignDoc(ToDoc(lhs), ToDoc(rhs), NullOpt);
    }
    return AssignDoc(ToDoc(lhs), ToDoc(rhs), IdDoc(annotation));
  }
  template <typename T>
  TVM_DLL static const AssignDoc ToAssign(const T& lhs, const String& rhs,
                                          const String& annotation = "") {
    Optional<ExprDoc> rhs_doc;
    if (rhs.size() > 0) {
      rhs_doc = IdDoc(rhs);
    } else {
      rhs_doc = NullOpt;
    }
    Optional<ExprDoc> annotation_doc;
    if (annotation.size() > 0) {
      annotation_doc = IdDoc(annotation);
    } else {
      annotation_doc = NullOpt;
    }
    return AssignDoc(ToDoc(lhs), rhs_doc, annotation_doc);
  }

  /*!
   * \brief Change object to AttrAccessDoc.
   * \return The AttrAccessDoc.
   */
  template <typename T>
  TVM_DLL static const AttrAccessDoc ToAttrAccess(const T& value, const String& name) {
    return AttrAccessDoc(ToDoc(value), name);
  }

  /*!
   * \brief Change object to List of Docs.
   * \return The List of Docs.
   */
  template <typename T>
  TVM_DLL static const Array<ExprDoc> ToDocList(const std::vector<T>& values) {
    Array<ExprDoc> elements;
    for (const auto& v : values) {
      elements.push_back(ToDoc(v));
    }
    return elements;
  }
  template <typename T>
  TVM_DLL static const Array<ExprDoc> ToDocList(const Array<T>& values) {
    std::vector<T> v_values;
    for (const auto& v : values) {
      v_values.push_back(v);
    }
    return ToDocList(v_values);
  }

  /*!
   * \brief Change object to ListDoc.
   * \return The ListDoc.
   */
  template <typename T>
  TVM_DLL static const StrictListDoc ToList(const std::vector<T>& values,
                                            bool allow_empty = false) {
    if (values.size() > 0 || allow_empty) {
      return StrictListDoc(ListDoc(ToDocList(values)), allow_empty);
    }
    return StrictListDoc(ListDoc(), false);
  }
  template <typename T>
  TVM_DLL static const StrictListDoc ToList(const Array<T>& values, bool allow_empty = false) {
    std::vector<T> v_values;
    for (const auto& v : values) {
      v_values.push_back(v);
    }
    return ToList(v_values, allow_empty);
  }

  /*!
   * \brief Change object to ListDoc for string elemenets.
   * \return The ListDoc.
   */
  TVM_DLL static const StrictListDoc ToStrList(const std::vector<std::string>& values,
                                               bool allow_empty = false);
  TVM_DLL static const StrictListDoc ToStrList(const std::vector<String>& values,
                                               bool allow_empty = false);
  TVM_DLL static const StrictListDoc ToStrList(const Array<String>& values,
                                               bool allow_empty = false);

  /*!
   * \brief Change object to IndexDoc.
   * \return The IndexDoc.
   */
  template <typename VT, typename IT>
  TVM_DLL static const IndexDoc ToIndex(const VT& value, const IT& index) {
    Array<Doc> doc_indices;
    doc_indices.push_back(ToDoc(index));
    return IndexDoc(ToDoc(value), doc_indices);
  }
  template <typename VT, typename IT>
  TVM_DLL static const IndexDoc ToIndices(const VT& value, const std::vector<IT>& indices) {
    Array<Doc> doc_indices;
    for (const auto& i : indices) {
      doc_indices.push_back(ToDoc(i));
    }
    return IndexDoc(ToDoc(value), doc_indices);
  }
  template <typename VT, typename IT>
  TVM_DLL static const IndexDoc ToIndices(const VT& value, const Array<IT>& indices) {
    Array<Doc> doc_indices;
    for (const auto& i : indices) {
      doc_indices.push_back(ToDoc(i));
    }
    return IndexDoc(ToDoc(value), doc_indices);
  }

  /*!
   * \brief Convert the docs to Stmts.
   * \return The Stmts.
   */
  TVM_DLL static const Array<StmtDoc> ToStmts(const Array<Doc>& docs);

  /*!
   * \brief Convert the docs to StmtBlock.
   * \return The StmtBlockDoc.
   */
  TVM_DLL static const StmtBlockDoc ToStmtBlock(const Array<Doc>& docs);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_PRINT_UTILS_H_
