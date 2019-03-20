/*!
 *  Copyright (c) 2019 by Contributors
 * \file src/tvm/relay/doc.cc
 * \brief Doc ADT used for pretty printing.
 * Based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.
 */
#include <memory>
#include <vector>
#include "doc.h"

namespace tvm {
namespace relay {

// Text constructor
DocAtom Text(const std::string& str) {
  return std::make_shared<TextNode>(str);
}

// Line constructor
DocAtom Line(int indent = 0) {
  return std::make_shared<LineNode>(indent);
}

Doc::Doc(const std::string& str) {
  if (str == "\n") {
    this->stream_ = {Line()};
  } else {
    this->stream_ = {Text(str)};
  }
}

// DSL function implementations

Doc& Doc::operator<<(const Doc& right) {
  assert(this != &right);
  this->stream_.insert(this->stream_.end(), right.stream_.begin(), right.stream_.end());
  return *this;
}

Doc& Doc::operator<<(const std::string& right) {
  return *this << Doc(right);
}

Doc Indent(int indent, const Doc& doc) {
  Doc ret;
  for (auto atom : doc.stream_) {
    if (auto text = std::dynamic_pointer_cast<TextNode>(atom)) {
      ret.stream_.push_back(text);
    } else if (auto line = std::dynamic_pointer_cast<LineNode>(atom)) {
      ret.stream_.push_back(Line(indent + line->indent));
    } else {assert(false);}
  }
  return ret;
}

std::string Doc::str() {
  std::ostringstream os;
  for (auto atom : this->stream_) {
    if (auto text = std::dynamic_pointer_cast<TextNode>(atom)) {
      os << text->str;
    } else if (auto line = std::dynamic_pointer_cast<LineNode>(atom)) {
      os << "\n" << std::string(line->indent, ' ');
    } else {assert(false);}
  }
  return os.str();
}

Doc PrintVec(const std::vector<Doc>& vec, const Doc& sep) {
  Doc seq;
  if (vec.size() != 0) {
    seq = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
      seq << sep << vec[i];
    }
  }
  return seq;
}

Doc PrintBool(bool value) {
  if (value) {
    return Doc("True");
  } else {
    return Doc("False");
  }
}

Doc PrintDType(DataType dtype) {
  return Doc(runtime::TVMType2String(Type2TVMType(dtype)));
}

Doc PrintString(const std::string& value) {
  // TODO(M.K.): add escape.
  Doc doc;
  return doc << "\"" << value << "\"";
}

}  // namespace relay
}  // namespace tvm
