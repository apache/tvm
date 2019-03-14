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

// DSL function implementations

// text constructor
DocAtom Text(const std::string& str) {
  return std::make_shared<TextNode>(str);
}

// line constructor
DocAtom Line(int indent) {
  return std::make_shared<LineNode>(indent);
}

// sugar for Concat with result stored in left
Doc& Doc::operator<<(const Doc& right) {
  this->stream_.insert(this->stream_.end(), right.stream_.begin(), right.stream_.end());
  return *this;
}

// like above, but automatically lifts string to a doc
Doc& Doc::operator<<(const std::string& right) {
  if (right == "\n") {
    return *this << Line();
  } else {
    return *this << Text(right);
  }
}

// indent a doc
Doc Indent(int indent, const Doc& doc) {
  Doc ret;
  for (auto atom : doc.stream_) {
    if (auto text = std::dynamic_pointer_cast<TextNode>(atom)) {
      ret << atom;
    } else if (auto line = std::dynamic_pointer_cast<LineNode>(atom)) {
      ret << Line(indent + line->indent);
    } else {assert(false);}
  }
  return ret;
}

// render vectors of docs with a separator. e.g. [1, 2, 3], f -> 1f2f3
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

/*!
 * \brief Print constant bool value.
 * \param value The value to be printed.
 */
Doc PrintBool(bool value) {
  if (value) {
    return Text("True");
  } else {
    return Text("False");
  }
}

Doc PrintDType(DataType dtype) {
  return Text(runtime::TVMType2String(Type2TVMType(dtype)));
}

Doc PrintString(const std::string& value) { // NOLINT(*)
  // TODO(M.K.): add escape.
  Doc doc;
  return doc << "\"" << value << "\"";
}

}  // namespace relay
}  // namespace tvm
