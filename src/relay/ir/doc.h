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
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/doc.h
 * \brief Doc ADT used for pretty printing.
 *  Based on Section 1 of
 *  https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf, but with
 *  a vector instead of an implicitly linked list.
 */
#ifndef TVM_RELAY_IR_DOC_H_
#define TVM_RELAY_IR_DOC_H_

#include <tvm/relay/expr.h>
#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

// Doc Atom ADT
struct DocAtomNode {
  virtual ~DocAtomNode() = default;
};

using DocAtom = std::shared_ptr<DocAtomNode>;

struct TextNode : DocAtomNode {
  std::string str;

  explicit TextNode(const std::string& str) : str(str) {}
};

struct LineNode : DocAtomNode {
  int indent;

  explicit LineNode(int indent) : indent(indent) {}
};

// Doc is a stream-like interface
class Doc {
 public:
  Doc() {}
  explicit Doc(const std::string& str);
  template<typename T>
  explicit Doc(const T& str) {
    (*this) << str;
  }

  // Append right to this.
  Doc& operator<<(const Doc& right);
  // Like above.
  Doc& operator<<(const std::string& right);
  // Like above.
  Doc& operator<<(const DocAtom& right);
  // Like above, but converts right to a string first.
  template<typename T>
  Doc& operator<<(const T& right) {
    std::ostringstream os;
    os << right;
    return *this << os.str();
  }

  // Indent a doc stream.
  friend Doc Indent(int indent, const Doc& doc);

  // Wadler's `layout`
  std::string str();

 private:
  std::vector<DocAtom> stream_;
};

// DSL functions

// Render vectors of docs with a separator. e.g. PrintVec([1, 2, 3], f) -> 1f2f3
Doc PrintVec(const std::vector<Doc>& vec, const Doc& sep = Doc(", "));
// Print a constant bool value.
Doc PrintBool(bool value);
// Print a data type.
Doc PrintDType(DataType dtype);
// Print a string.
Doc PrintString(const std::string& value);
// Print a newline.
Doc PrintNewLine(int indent = 0);
/*!
 * \brief special method to print out const scalar
 * \param dtype The data type
 * \param data The pointer to hold the data.
 */
template<typename T>
Doc PrintConstScalar(DataType dtype, const T* data) {
  std::ostringstream os;
  if (dtype == Int(32)) {
    os << data[0];
  } else if (dtype == Float(32)) {
    os << data[0] << 'f';
  } else if (dtype == Bool()) {
    return PrintBool(data[0] != 0);
  } else {
    os << dtype << "(" << data[0] << ")";
  }
  return Doc(os.str());
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_IR_DOC_H_
