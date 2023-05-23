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
 * \file src/relay/doc.cc
 * \brief Doc ADT used for pretty printing.
 *
 *  Reference: Philip Wadler. A Prettier Printer. Journal of Functional Programming'98
 */
#include "doc.h"

#include <tvm/runtime/packed_func.h>

#include <sstream>
#include <vector>

#include "../../support/str_escape.h"

namespace tvm {
namespace relay {

/*!
 * \brief Represent a piece of text in the doc.
 */
class DocTextNode : public DocAtomNode {
 public:
  /*! \brief The str content in the text. */
  std::string str;

  explicit DocTextNode(std::string str_val) : str(str_val) {}

  static constexpr const char* _type_key = "printer.DocText";
  TVM_DECLARE_FINAL_OBJECT_INFO(DocTextNode, DocAtomNode);
};

TVM_REGISTER_OBJECT_TYPE(DocTextNode);

class DocText : public DocAtom {
 public:
  explicit DocText(std::string str) { data_ = runtime::make_object<DocTextNode>(str); }

  TVM_DEFINE_OBJECT_REF_METHODS(DocText, DocAtom, DocTextNode);
};

/*!
 * \brief Represent a line breaker in the doc.
 */
class DocLineNode : public DocAtomNode {
 public:
  /*! \brief The amount of indent in newline. */
  int indent;

  explicit DocLineNode(int indent) : indent(indent) {}

  static constexpr const char* _type_key = "printer.DocLine";
  TVM_DECLARE_FINAL_OBJECT_INFO(DocLineNode, DocAtomNode);
};

TVM_REGISTER_OBJECT_TYPE(DocLineNode);

class DocLine : public DocAtom {
 public:
  explicit DocLine(int indent) { data_ = runtime::make_object<DocLineNode>(indent); }

  TVM_DEFINE_OBJECT_REF_METHODS(DocLine, DocAtom, DocLineNode);
};

// DSL function implementations
Doc& Doc::operator<<(const Doc& right) {
  ICHECK(this != &right);
  this->stream_.insert(this->stream_.end(), right.stream_.begin(), right.stream_.end());
  return *this;
}

Doc& Doc::operator<<(std::string right) { return *this << DocText(right); }

Doc& Doc::operator<<(const DocAtom& right) {
  this->stream_.push_back(right);
  return *this;
}

std::string Doc::str() {
  std::ostringstream os;
  for (auto atom : this->stream_) {
    if (auto* text = atom.as<DocTextNode>()) {
      os << text->str;
    } else if (auto* line = atom.as<DocLineNode>()) {
      os << "\n" << std::string(line->indent, ' ');
    } else {
      LOG(FATAL) << "do not expect type " << atom->GetTypeKey();
    }
  }
  return os.str();
}

Doc Doc::NewLine(int indent) { return Doc() << DocLine(indent); }

Doc Doc::Text(std::string text) { return Doc() << DocText(text); }

Doc Doc::RawText(std::string text) {
  return Doc() << DocAtom(runtime::make_object<DocTextNode>(text));
}

Doc Doc::Indent(int indent, Doc doc) {
  for (size_t i = 0; i < doc.stream_.size(); ++i) {
    if (auto* line = doc.stream_[i].as<DocLineNode>()) {
      doc.stream_[i] = DocLine(indent + line->indent);
    }
  }
  return doc;
}

Doc Doc::StrLiteral(const std::string& value, std::string quote) {
  Doc doc;
  return doc << quote << support::StrEscape(value) << quote;
}

Doc Doc::PyBoolLiteral(bool value) {
  if (value) {
    return Doc::Text("True");
  } else {
    return Doc::Text("False");
  }
}

Doc Doc::Brace(std::string open, const Doc& body, std::string close, int indent) {
  Doc doc;
  doc << open;
  doc << Indent(indent, NewLine() << body) << NewLine();
  doc << close;
  return doc;
}

Doc Doc::Concat(const std::vector<Doc>& vec, const Doc& sep) {
  Doc seq;
  if (vec.size() != 0) {
    if (vec.size() == 1) return vec[0];
    seq << vec[0];
    for (size_t i = 1; i < vec.size(); ++i) {
      seq << sep << vec[i];
    }
  }
  return seq;
}
}  // namespace relay
}  // namespace tvm
