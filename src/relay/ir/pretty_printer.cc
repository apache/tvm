/*!
 *  Copyright (c) 2019 by Contributors
 * \file pretty_printer.cc
 * \brief Pretty printer for Relay programs
 * Supports ANF and GNF formats and metadata.
 */
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

namespace doc {

// Doc model based on Section 1 of https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf.

enum DocType { NIL, TEXT, LINE };

struct Doc { 
  virtual DocType getType() { assert(false); };
};

struct Nil : Doc {
  DocType getType() { return NIL; }
};

struct Text : Doc {
  std::string str;
  Doc doc;

  Text(std::string str) : str(str), doc(Nil()) { }  
  Text(std::string str, Doc doc) : str(str), doc(doc) { }

  DocType getType() { return TEXT; }
};

struct Line : Doc {
  size_t indent;
  Doc doc;

  Line() : indent(0), doc(Nil()) { }
  Line(size_t indent, Doc doc) : indent(indent), doc(doc) { }

  DocType getType() { return LINE; }
};

// concatenate two documents
Doc Concat(Doc &left, Doc &right) {
  if (left.getType() == TEXT) {
    Text &text = static_cast<Text&>(left);
    return Text(text.str, Concat(text.doc, right));
  } else if (left.getType() == LINE) {
    Line &line = static_cast<Line&>(left);
    return Line(line.indent, Concat(line.doc, right));
  } else if (left.getType() == NIL) {
    return right;
  } else { assert(false); }
}

// overload + to concatenate documents
Doc operator+(Doc& left, Doc& right) {
  return Concat(left, right);
}

// add indentation to a document
Doc Nest(size_t indent, Doc doc) {
  if (doc.getType() == TEXT) {
    Text &text = static_cast<Text&>(doc);
    return Text(text.str, Nest(indent, text.doc));
  } else if (doc.getType() == LINE) {
    Line &line = static_cast<Line&>(doc);
    return Line(indent + line.indent, Nest(indent, line.doc));
  } else if (doc.getType() == NIL) {
    return Nil();
  } else { assert(false); }
}

// print a document to the given ostream
void Layout(Doc doc, std::ostream& os) {
  if (doc.getType() == TEXT) {
    Text &text = static_cast<Text&>(doc);
    os << text.str;
    Layout(text.doc, os);
  } else if (doc.getType() == LINE) {
    Line &line = static_cast<Line&>(doc);
    os << std::endl << std::string(line.indent, ' ');
    Layout(line.doc, os);
  } else if (doc.getType() == NIL) {
    // do nothing!
  } else { assert(false); }
}

} // doc

} // relay
} // tvm