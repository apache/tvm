/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/doc.h
 * \brief Doc ADT used for pretty printing.
 * Based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.
 */
#ifndef TVM_RELAY_DOC_H_
#define TVM_RELAY_DOC_H_

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

// ADT
struct DocNode {
  virtual ~DocNode() = default;
};

using Doc = std::shared_ptr<DocNode>;

struct NilNode : DocNode { };

struct TextNode : DocNode {
  std::string str;
  Doc doc;

  TextNode(const std::string& str, const Doc& doc) : str(str), doc(doc) {}
};

struct LineNode : DocNode {
  int indent;
  Doc doc;

  LineNode(int indent, const Doc& doc) : indent(indent), doc(doc) {}
};

/* template<typename T>
T Match(const Doc& doc,
  const T& case_nil,
  const std::function<T(const std::string&, const Doc&)>& case_text,
  const std::function<T(int, const Doc&)>& case_line) {
  if (auto nil = std::dynamic_pointer_cast<NilNode>(doc)) {
    return case_nil;
  } else if (auto text = std::dynamic_pointer_cast<TextNode>(doc)) {
    return case_text(text->str, text->doc);
  } else if (auto line = std::dynamic_pointer_cast<LineNode>(doc)) {
    return case_line(line->indent, line->doc);
  } else {assert(false);}
} */

// text constructor
Doc Text(const std::string& str, const Doc& doc);

// line constructor
Doc Line(int indent, const Doc& doc);

// DSL functions

// empty doc/nil constructor
Doc Nil();
// lift string to text
Doc Text(const std::string& str);
// new line
Doc Line();
// concat two docs
Doc Concat(const Doc& left, const Doc& right);
// sugar for Concat
Doc operator+(const Doc& left, const Doc& right);
// indent a doc
Doc Nest(int indent, const Doc& doc);
// convert doc to a string
std::string Layout(const Doc& doc);
// render array-like things: e.g. (1, 2, 3)
Doc PrintVec(const Doc& open, const std::vector<Doc>& arr, const Doc& sep, const Doc& close);
// Print constant bool value.
Doc PrintBool(bool value);
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
  return Text(os.str());
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_DOC_H_
