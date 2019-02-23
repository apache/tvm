/*!
 *  Copyright (c) 2019 by Contributors
 * \file pretty_printer.cc
 * \brief Pretty printer for Relay programs
 * Supports ANF, GNF, and metadata.
 */
#include "doc.h"
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class PrettyPrinter :
    public ExprFunctor<Doc(const Expr&)> {
  public:
    explicit PrettyPrinter(const std::unordered_map<Expr, Doc, NodeHash, NodeEqual>& memo_, size_t temp_var_counter_, bool GNF_) : memo_(memo_), temp_var_counter_(temp_var_counter_), GNF_(GNF_) {}

    explicit PrettyPrinter() : temp_var_counter_(0), GNF_(true) {}

    explicit PrettyPrinter(bool GNF_) : temp_var_counter_(0), GNF_(GNF_) {}

    // indent a new body
    Doc PrintBody(const NodeRef& node, int indent = 2) {
      Doc doc = Nil();
      Doc body = Nil();
      doc << "{";
      doc << Indent(indent, body << "\n" << PrintNestedScope(node)) << "\n";
      doc << "}";
      return doc;
    }

    // create a new scope by creating a new printer object.
    Doc PrintNestedScope(const NodeRef& node) {
      if (GNF_) {
        return PrettyPrinter(memo_, temp_var_counter_, GNF_).PrintFinal(node);
      } else {
        return Print(node);
      }
    }

    Doc PrintFinal(const NodeRef& node) {
     Print(node, true, false);
      return doc;
    }

    // note: gnf flag is only one level deep
    Doc Print(const NodeRef& node, bool gnf = true, bool hoist = true) {
      if (node.as_derived<ExprNode>()) {
        return PrintExpr(Downcast<Expr>(node), gnf, hoist);
      } else { assert(false); }
    }

    Doc TempVar(int n) {
      Doc doc = Nil();
      return doc << "\%" << n;
    }

    Doc AllocTemp() {
      return TempVar(temp_var_counter_++);
    }

    Doc PrintExpr(const Expr& expr, bool gnf = true, bool hoist = true) {
      // Exploit memoization to print GNF.
      // The first time we visit an expression, we need to allocate a temp var
      // for it. Every subsequent time we can just use its assigned variable.
      // This works since hashing uses pointer equality.
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      Doc printed_expr = VisitExpr(expr);
      if (gnf && GNF_) {
        if (hoist) {
          Doc temp_var = AllocTemp();
          memo_[expr] = temp_var;
          doc << temp_var << " = " << printed_expr << "\n";
          return temp_var;
        } else {
          memo_[expr] = printed_expr;
          doc << printed_expr;
          return printed_expr;
        }
      } else {
        memo_[expr] = printed_expr;
        return printed_expr;
      }
    }

    Doc VisitExpr_(const ConstantNode* op) final {
      // Print out simple scalars directly.
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
      // TODO: handle tensors
      assert(false);
    }

    Doc VisitExpr_(const TupleNode* op) final {
      std::vector<Doc> fields;
      for (Expr field : op->fields) {
        fields.push_back(Print(field));
      }
      Doc doc = Nil();
      return doc << "(" << PrintVec(fields, Text(", ")) << ")";
    }

    Doc VisitExpr_(const TupleGetItemNode* op) final {
      Doc doc = Nil();
      return doc << Print(op->tuple) << "." << op->index;
    }

    Doc VisitExpr_(const IfNode* op) final {
      Doc doc = Nil();
      doc << "if (" << Print(op->cond) << ") ";
      doc << PrintBody(op->true_branch);
      doc << " else ";
      doc << PrintBody(op->false_branch);
      return doc;
    }

    Doc VisitExpr_(const LetNode* op) final {
      Doc doc = Nil();
      // TODO: this should call a var printer, which needs to differentiate
      //    between free and bound vars
      // TODO: this should have a type annotation
      // TODO: lets in value position need to be scoped
      // 
      // we use ANF mode for the first level of the value position so the final
      // expression isn't hoisted or added to the doc stream
      doc << "let \%" << op->var->name_hint() << " = " << Print(op->value, false) << ";" << "\n";
      doc << PrintNestedScope(op->body);
      return doc;
    }

    // Doc PrintFunc(const Doc& prefix, const FunctionNode* fn) {
    //     // TODO(tqchen, M.K.) support generic function
    //     // Possibly through meta-data
    //     CHECK_EQ(fn->type_params.size(), 0U)
    //     << "generic fn not yet supported";
    //     Doc doc = Nil();
    //     doc << prefix << "(";
    //     AllocVarName(fn->params[i]);
    //     this->PrintVarDecl(fn->params[i], stream_);
    //     doc << ')';
    //     /* if (fn->ret_type.defined()) {
    //       doc << " -> ";
    //       this->PrintType(fn->ret_type, stream_);
    //     } */
    //     doc << PrintBody(fn->body);
    //     return doc;
    // }

  private:
    /*! \brief Map from Expr to Doc */
    Doc doc = Nil();
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
    size_t temp_var_counter_;
    bool GNF_;
};

std::string RelayGNFPrint(const NodeRef& node) {
  return "v0.0.1\n" + Layout(PrettyPrinter().PrintFinal(node)) + "\n";
}

std::string RelayANFPrint(const NodeRef& node) {
  return "v0.0.1\n" + Layout(PrettyPrinter(false).Print(node)) + "\n";
}

TVM_REGISTER_API("relay._expr.gnf_print")
.set_body_typed<std::string(const NodeRef&)>(RelayGNFPrint);

TVM_REGISTER_API("relay._expr.anf_print")
.set_body_typed<std::string(const NodeRef&)>(RelayANFPrint);

} // relay
} // tvm
