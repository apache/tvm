/*!
 *  Copyright (c) 2019 by Contributors
 * \file src/tvm/relay/doc.cc
 * \brief Doc ADT used for pretty printing.
 * Based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.
 */
#include <tvm/relay/doc.h>

namespace tvm {
namespace relay {

// Doc ADT implementation
Nil_ Nil_Node::make() {
  NodePtr<Nil_Node> n = make_node<Nil_Node>();
  return Nil_(n);
}

TVM_REGISTER_API("relay._make.Nil_")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = Nil_Node::make();
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<Nil_Node>([](const Nil_Node* node, tvm::IRPrinter* p) {
    p->stream << "Nil_Node()";
  });

Text_ Text_Node::make(std::string str, Doc doc) {
  NodePtr<Text_Node> n = make_node<Text_Node>();
  n->str = std::move(str);
  n->doc = std::move(doc);
  return Text_(n);
}

TVM_REGISTER_API("relay._make.Text_")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = Text_Node::make(args[0], args[1]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<Text_Node>([](const Text_Node* node, tvm::IRPrinter* p) {
    p->stream << "Text_Node(" << node->str << ", " << node->doc << ")";
  });

Line_ Line_Node::make(int indent, Doc doc) {
  NodePtr<Line_Node> n = make_node<Line_Node>();
  n->indent = indent;
  n->doc = std::move(doc);
  return Line_(n);
}

TVM_REGISTER_API("relay._make.Line_")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = Line_Node::make(args[0], args[1]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<Line_Node>([](const Line_Node* node, tvm::IRPrinter* p) {
    p->stream << "Line_Node(" << node->indent << ", " << node->doc << ")";
  });

// DSL functions

// empty doc
Doc Nil() {
  return Nil_Node::make();
}

// lift string to text
Doc Text(const std::string str) {
  return Text_Node::make(str, Nil());
}

// new line
Doc Line() {
  return Line_Node::make(0, Nil());
}

// concat two docs
Doc Concat(const Doc& left, const Doc& right) {
  if (const Text_Node* text = left.as<Text_Node>()) {
    // push right into text continuation
    return Text_Node::make(text->str, Concat(text->doc, right));
  } else if (const Line_Node* line = left.as<Line_Node>()) {
    // push right into line continuation
    return Line_Node::make(line->indent, Concat(line->doc, right));
  } else if (const Nil_Node* nil = left.as<Nil_Node>()) {
    // throwaway nils on the left
    return right;
  } else {assert(false);}
}

// sugar for Concat
Doc operator+(const Doc& left, const Doc& right) {
  return Concat(left, right);
}

// indent a doc
Doc Nest(int indent, const Doc& doc) {
  if (const Text_Node* text = doc.as<Text_Node>()) {
    // push nest through
    return Text_Node::make(text->str, Nest(indent, text->doc));
  } else if (const Line_Node* line = doc.as<Line_Node>()) {
    // add indent to lines and continue
    return Line_Node::make(indent + line->indent, Nest(indent, line->doc));
  } else if (const Nil_Node* nil = doc.as<Nil_Node>()) {
    // absorb it
    return Nil_();
  } else {assert(false);}
}

// convert a doc to a string
std::string Layout(const Doc& doc) {
  if (const Text_Node* text = doc.as<Text_Node>()) {
    // add text and continue
    return text->str + Layout(text->doc);
  } else if (const Line_Node* line = doc.as<Line_Node>()) {
    // add a newline and indents, then continue
    return "\n" + std::string(line->indent, ' ') + Layout(line->doc);
  } else if (const Nil_Node* nil = doc.as<Nil_Node>()) {
    // empty string
    return "";
  } else {assert(false);}
}

} // relay
} // tvm