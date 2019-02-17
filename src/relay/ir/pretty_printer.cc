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

// convert a document to a string
std::string Layout(Doc doc) {
  if (doc.getType() == TEXT) {
    Text &text = static_cast<Text&>(doc);
    return text.str + Layout(text.doc);
  } else if (doc.getType() == LINE) {
    Line &line = static_cast<Line&>(doc);
    return "\n" + std::string(line.indent, ' ') + Layout(line.doc);
  } else if (doc.getType() == NIL) {
    return "";
  } else { assert(false); }
}

} // doc

class PrettyPrinter :
    public ExprFunctor<doc::Doc(const Expr&)> {
  public:
    explicit PrettyPrinter() {}

    std::string Print(const NodeRef& node) {
      if (node.as_derived<ExprNode>()) {
        return doc::Layout(this->PrintExpr(Downcast<Expr>(node)));
      } else { assert(false); }
    }

    doc::Doc PrintExpr(const Expr& expr) {
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      doc::Doc val = this->VisitExpr(expr);
      memo_[expr] = val;
      return val;
    }

  private:
    /*! \brief Map from Expr to Doc */
    std::unordered_map<Expr, doc::Doc, NodeHash, NodeEqual> memo_;
};

std::string RelayPrettyPrint(const NodeRef& node) {
  return PrettyPrinter().Print(node);
}

TVM_REGISTER_API("relay._expr.RelayPrettyPrint")
.set_body_typed<std::string(const NodeRef&)>(RelayPrettyPrint);

} // relay
} // tvm