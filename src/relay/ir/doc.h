/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/doc.h
 * \brief Doc ADT used for pretty printing.
 * Based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.
 */
#ifndef TVM_RELAY_IR_DOC_H_
#define TVM_RELAY_IR_DOC_H_

#include <tvm/relay/expr.h>
#include <string>
#include <vector>

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
// sugar for Concat with result stored in left
Doc& operator<<(Doc& left, const Doc& right);
// like above, but automatically lifts string to a doc
Doc& operator<<(Doc& left, const std::string& right);
// like above, but converts right to a string
template<typename T>
Doc& operator<<(Doc& left, const T& right) {
  std::ostringstream os;
  os << right;
  return left << os.str();
}
// indent a doc
Doc Indent(int indent, const Doc& doc);
// convert doc to a string
std::string Layout(const Doc& doc);
// render vectors of docs with a separator. e.g. [1, 2, 3], f -> 1f2f3
Doc PrintVec(const std::vector<Doc>& vec, const Doc& sep = Text(", "));
// Print constant bool value.
Doc PrintBool(bool value);
Doc PrintDType(DataType dtype);
Doc PrintString(const std::string& value);
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
#endif  // TVM_RELAY_IR_DOC_H_
