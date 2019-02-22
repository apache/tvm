/*!
 *  Copyright (c) 2019 by Contributors
 * \file src/tvm/relay/doc.cc
 * \brief Doc ADT used for pretty printing.
 * Based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.
 */
#include <tvm/relay/doc.h>

namespace tvm {
namespace relay {

// DSL function implementations

// empty doc/nil constructor
Doc Nil() {
  return std::make_shared<NilNode>();
}

// text constructor
Doc Text(const std::string& str, const Doc& doc) {
  return std::make_shared<TextNode>(str, doc);
}

// lift string to text
Doc Text(const std::string& str) {
  return Text(str, Nil());
}

// line constructor
Doc Line(int indent, const Doc& doc) {
  return std::make_shared<LineNode>(indent, doc);
}

// new line
Doc Line() {
  return Line(0, Nil());
}

// concat two docs
Doc Concat(const Doc& left, const Doc& right) {
  if (auto nil = std::dynamic_pointer_cast<NilNode>(left)) {
    // throw away nil
    return right;
  } else if (auto text = std::dynamic_pointer_cast<TextNode>(left)) {
    // push right into text continuation
    return Text(text->str, Concat(text->doc, right));
  } else if (auto line = std::dynamic_pointer_cast<LineNode>(left)) {
    // push right into line continuation
    return Line(line->indent, Concat(line->doc, right));
  } else {assert(false);}
}

// sugar for Concat
Doc operator+(const Doc& left, const Doc& right) {
  return Concat(left, right);
}

// indent a doc
Doc Nest(int indent, const Doc& doc) {
  if (auto nil = std::dynamic_pointer_cast<NilNode>(doc)) {
    // absorb nest
    return nil;
  } else if (auto text = std::dynamic_pointer_cast<TextNode>(doc)) {
    // push nest through
    return Text(text->str, Nest(indent, text->doc));
  } else if (auto line = std::dynamic_pointer_cast<LineNode>(doc)) {
    // add indent to line and continue
    return Line(indent + line->indent, Nest(indent, line->doc));
  } else {assert(false);}
}

// convert a doc to a string
std::string Layout(const Doc& doc) {
  if (auto nil = std::dynamic_pointer_cast<NilNode>(doc)) {
    return "";
  } else if (auto text = std::dynamic_pointer_cast<TextNode>(doc)) {
    // add text and continue
    return text->str + Layout(text->doc);
  } else if (auto line = std::dynamic_pointer_cast<LineNode>(doc)) {
    // add a newline and indents, then continue
    return "\n" + std::string(line->indent, ' ') + Layout(line->doc);
  } else {assert(false);}
}

// render array-like things: e.g. (1, 2, 3)
Doc PrintVec(const Doc& open, const std::vector<Doc>& vec, const Doc& sep, const Doc& close) {
  Doc seq;
  if (vec.size() == 0) {
    seq = Nil();
  } else {
    seq = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
      seq = seq + sep + vec[i];
    }
  }

  return open + seq + close;
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

} // relay
} // tvm
