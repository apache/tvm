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
 * \file tvm/printer/doc.h
 * \brief Doc ADT used for pretty printing.
 *
 *  Reference: Philip Wadler. A Prettier Printer. Journal of Functional Programming'98
 */
#ifndef TVM_RELAY_PRINTER_DOC_H_
#define TVM_RELAY_PRINTER_DOC_H_

#include <tvm/node/node.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>

#include <string>
#include <type_traits>
#include <vector>

namespace tvm {
namespace relay {

/*!
 * \brief Doc atom node for the ADT.
 * \sa DocAtom
 */
class DocAtomNode : public Object {
 public:
  static constexpr const char* _type_key = "printer.DocAtom";
  TVM_DECLARE_BASE_OBJECT_INFO(DocAtomNode, Object);
};

/*!
 * \brief Managed reference to DocAtomNode.
 * \sa DocAtomNode.
 */
class DocAtom : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DocAtom, ObjectRef, DocAtomNode);
};

/*!
 * \brief Stream-like interface for Doc DSL.
 *
 * The Doc DSL de-couples the layout decision from the printing decision.
 *
 * The layout(code formating) decisions include:
 * - Change indentation.
 * - Break single line into multiple ones(subjected to future improvements).
 */
class Doc {
 public:
  /*! \brief default constructor */
  Doc() {}
  /*!
   * \brief Append right to the end of the current doc stream.
   * \param right The doc to be appended.
   * \return reference to self.
   */
  Doc& operator<<(const Doc& right);
  /*!
   * \brief Append right to the end of the current doc stream.
   * \param right The doc to be appended.
   * \return reference to self.
   * \note pass by value to allow copy elison optimization.
   */
  Doc& operator<<(std::string right);
  /*!
   * \brief Append right to the end of the current doc stream.
   * \param right The doc to be appended.
   * \return reference to self.
   */
  Doc& operator<<(const DocAtom& right);
  /*!
   * \brief Convert value to string via std::ostreamstream
   *        the append to the current doc stream.
   * \param right The doc to be appended.
   * \tparam T the type of the value.
   * \return reference to self.
   */
  template <typename T, typename = typename std::enable_if<!std::is_class<T>::value>::type>
  Doc& operator<<(const T& value) {
    std::ostringstream os;
    os << value;
    return *this << os.str();
  }
  /*!
   * \brief Convert the doc stream into string.
   * \return The string representation.
   */
  std::string str();
  /*!
   * \brief Create a doc that represents text content.
   * \return The created doc.
   */
  static Doc Text(std::string value);
  /*!
   * \brief Create a doc that represents raw text(can have new lines)
   * \return The created doc.
   */
  static Doc RawText(std::string value);
  /*!
   * \brief Create a doc that represents a new line.
   * \return The created doc.
   */
  static Doc NewLine(int indent = 0);
  /*!
   * \brief Create a new doc that adds indentation to everyline of the doc.
   * \param indent The indent to be added.
   * \param doc The doc to be indented.
   * \return The created doc.
   * \note pass by value to allow copy elison optimization.
   */
  static Doc Indent(int indent, Doc doc);
  /*!
   * \brief Create a Doc that represents a string literal.
   * \param value The content of the string literal.
   * \param quote The quote in the literal.
   * \return The created doc.
   */
  static Doc StrLiteral(const std::string& value, std::string quote = "\"");
  /*!
   * \brief Create a Doc that represents a boolean literal in python syntax.
   * \param value The bool value.
   * \return The created doc.
   */
  static Doc PyBoolLiteral(bool value);
  /*!
   * \brief Enclose body by brace and add indent.
   * \param body The body
   * \param open The open brace.
   * \param close The close brace.
   * \param indent amount of indentation.
   * \return The created doc.
   */
  static Doc Brace(std::string open, const Doc& body, std::string close, int indent = 2);
  /*!
   * \brief Create a doc by concatenating  together with separator.
   * \param vec The docs to be concatenated.
   * \param sep The seperator.
   * \return The created doc.
   */
  static Doc Concat(const std::vector<Doc>& vec, const Doc& sep = Text(", "));

 private:
  /*! \brief Internal doc stream. */
  std::vector<DocAtom> stream_;
};
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PRINTER_DOC_H_
