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
  virtual DocType getType() const = 0;
};

struct Nil : public Doc {
  DocType getType() const { return NIL; }
};

struct Text : public Doc {
  const std::string str;
  const Doc* doc;

  Text(const std::string str) : str(str), doc(new Nil()) { }
  Text(const std::string str, const Doc* doc) : str(str), doc(doc) { }

  DocType getType() const { return TEXT; }
};

struct Line : public Doc {
  const size_t indent;
  const Doc* doc;

  Line() : indent(0), doc(new Nil()) { }
  Line(const size_t indent, const Doc* doc) : indent(indent), doc(doc) { }

  DocType getType() const { return LINE; }
};

// concatenate two documents
const Doc* Concat(const Doc* left, const Doc* right) {
  if (left->getType() == TEXT) {
    const Text* text = static_cast<const Text*>(left);
    return new Text(text->str, Concat(text->doc, right));
  } else if (left->getType() == LINE) {
    const Line* line = static_cast<const Line*>(left);
    return new Line(line->indent, Concat(line->doc, right));
  } else if (left->getType() == NIL) {
    return right;
  } else { assert(false); }
}

// overload + to concatenate documents
const Doc& operator+(const Doc& left, const Doc& right) {
  return *Concat(&left, &right);
}

// add indentation to a document
const Doc* Nest(size_t indent, const Doc* doc) {
  if (doc->getType() == TEXT) {
    const Text* text = static_cast<const Text*>(doc);
    return new Text(text->str, Nest(indent, text->doc));
  } else if (doc->getType() == LINE) {
    const Line* line = static_cast<const Line*>(doc);
    return new Line(indent + line->indent, Nest(indent, line->doc));
  } else if (doc->getType() == NIL) {
    return new Nil();
  } else { assert(false); }
}

// convert a document to a string
std::string Layout(const Doc& doc) {
  if (doc.getType() == TEXT) {
    const Text& text = static_cast<const Text&>(doc);
    return text.str + Layout(*text.doc);
  } else if (doc.getType() == LINE) {
    const Line& line = static_cast<const Line&>(doc);
    return "\n" + std::string(line.indent, ' ') + Layout(*line.doc);
  } else if (doc.getType() == NIL) {
    return "";
  } else { assert(false); }
}

} // doc

using namespace doc;

class PrettyPrinter :
    public ExprFunctor<const Doc&(const Expr&)> {
  public:
    explicit PrettyPrinter() {}

    std::string Print(const NodeRef& node) {
      if (node.as_derived<ExprNode>()) {
        return Layout(this->PrintExpr(Downcast<Expr>(node)));
      } else { assert(false); }
    }

    const Doc& PrintExpr(const Expr& expr) {
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      const Doc& val = this->VisitExpr(expr);
      memo_[expr] = val;
      return val;
    }

    // render a tvm array with open and closing brackets and a separator
    // we use docs instead of strings for input to allow the caller to use
    // newlines where desired
    template<typename T>
    const Doc& PrintArray(const Doc& open, const tvm::Array<T>& arr, const Doc& sep, const Doc& close) {
      Doc seq;
      if (arr.size() == 0) {
        seq = Nil();
      } else {
        seq = Text(this->Print(arr[0]));
        for (size_t i = 1; i < arr.size(); i++) {
          seq = seq + sep + Text(this->Print(arr[i]));
        }
      }

      return open + seq + close;
    }

    /*!
    * \brief special method to print out const scalar
    * \param dtype The data type
    * \param data The pointer to hold the data.
    */
    template<typename T>
    const Doc& PrintConstScalar(DataType dtype, const T* data) {  // NOLINT(*)
      std::stringstream ss;
      if (dtype == Int(32)) {
        ss << data[0];
      } else if (dtype == Float(32)) {
        ss << data[0] << 'f';
      } else if (dtype == Bool()) {
        // ss << PrintBool(data[0] != 0);
        assert(false);
      } else {
        ss << dtype << "(" << data[0] << ")";
      }
      return new Text(ss.str());
    }

    const Doc& VisitExpr_(const ConstantNode* op) final {
      // Print out simple scalar directly.
      if (op->is_scalar()) {
        std::ostringstream os;
        DataType dtype = TVMType2Type(op->data->dtype);
        CHECK_EQ(op->data->ctx.device_type, kDLCPU);
        if (dtype == Int(32)) {
          return PrintConstScalar(dtype, static_cast<const int32_t*>(op->data->data));
        } else if (dtype == Int(64)) {
          return PrintConstScalar(dtype, static_cast<const int64_t*>(op->data->data));
        } else if (dtype == Float(32)) {
          return PrintConstScalar(dtype, static_cast<const float*>(op->data->data));
        } else if (dtype == Float(64)) {
          return PrintConstScalar(dtype, static_cast<const double*>(op->data->data));
        } else if (dtype == Bool()) {
          return PrintConstScalar(dtype, static_cast<const uint8_t*>(op->data->data));
        }
      }
      // TODO: handle complicated scalars
      assert(false);
    }

    const Doc& VisitExpr_(const TupleNode* op) final {
        return PrintArray(Text("("), op->fields, Text(", "), Text(")"));
    }

  private:
    /*! \brief Map from Expr to Doc */
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
};

std::string RelayPrettyPrint(const NodeRef& node) {
  return PrettyPrinter().Print(node);
}

TVM_REGISTER_API("relay._expr.pretty_print")
.set_body_typed<std::string(const NodeRef&)>(RelayPrettyPrint);

} // relay
} // tvm