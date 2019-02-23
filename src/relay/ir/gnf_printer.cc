/*!
 *  Copyright (c) 2019 by Contributors
 * \file gnf_printer.cc
 * \brief GNF printer for Relay programs
 * Supports GNF and metadata.
 */
#include "doc.h"
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class GNFPrinter :
    public ExprFunctor<Doc(const Expr&)> {
  public:
    explicit GNFPrinter(const std::unordered_map<Expr, Doc, NodeHash, NodeEqual>& memo_, size_t temp_var_counter_) : memo_(memo_), temp_var_counter_(temp_var_counter_) {}

    explicit GNFPrinter() : temp_var_counter_(0) {}

    // create a new scope by creating a new printer object.
    Doc PrintNestedScope(const NodeRef& node) {
      return GNFPrinter(memo_, temp_var_counter_).PrintFinal(node);
    }

    Doc PrintFinal(const NodeRef& node) {
      Print(node, false);
      return doc;
    }

    // note: gnf flag is only one level deep
    Doc Print(const NodeRef& node, bool gnf) {
      if (node.as_derived<ExprNode>()) {
        return this->PrintExpr(Downcast<Expr>(node), gnf);
      } else { assert(false); }
    }

    Doc Print(const NodeRef& node) {
      return this->Print(node, true);
    }

    Doc TempVar(int n) {
      Doc doc = Nil();
      return doc << "\%" << n;
    }

    Doc AllocTemp() {
      return TempVar(temp_var_counter_++);
    }

    Doc PrintExpr(const Expr& expr, bool gnf) {
      // Exploit memoization to print GNF.
      // The first time we visit an expression, we need to allocate a temp var
      // for it. Every subsequent time we can just use its assigned variable.
      // This works since hashing uses pointer equality.
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      Doc printed_expr = this->VisitExpr(expr);
      if (gnf &&
          !expr.as<LetNode>()) {
        Doc temp_var = AllocTemp();
        memo_[expr] = temp_var;
        doc << temp_var << " = " << printed_expr << "\n";
        return temp_var;
      } else {
        memo_[expr] = printed_expr;
        doc << printed_expr;
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
        fields.push_back(this->Print(field));
      }
      Doc doc = Nil();
      return doc << "(" << PrintVec(fields, Text(", ")) << ")";
    }

    Doc VisitExpr_(const TupleGetItemNode* op) final {
      Doc doc = Nil();
      return doc << this->Print(op->tuple) << "." << op->index;
    }

    Doc VisitExpr_(const IfNode* op) final {
      Doc doc = Nil();
      Doc true_b = Nil();
      Doc false_b = Nil();
      doc << "if (" << this->Print(op->cond) << ") {";
      doc << Indent(2, true_b << "\n" << PrintNestedScope(op->true_branch)) << "\n";
      doc << "} else {";
      doc << Indent(2, false_b << "\n" << PrintNestedScope(op->false_branch)) << "\n";
      doc << "}";
      return doc;
    }

    Doc VisitExpr_(const LetNode* op) final {
      Doc ret = Nil();
      // TODO: this should call a var printer, which needs to differentiate
      //    between free and bound vars
      // TODO: this should have a type annotation
      ret << "let \%" << op->var->name_hint() << " = " << PrintNestedScope(op->value) << ";" << "\n";
      ret << PrintNestedScope(op->body);
      return ret;
    }

  private:
    /*! \brief Map from Expr to Doc */
    Doc doc = Nil();
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
    size_t temp_var_counter_;
};

std::string RelayGNFPrint(const NodeRef& node) {
  return "v0.0.1\n" + Layout(GNFPrinter().PrintFinal(node));
}

TVM_REGISTER_API("relay._expr.gnf_print")
.set_body_typed<std::string(const NodeRef&)>(RelayGNFPrint);

} // relay
} // tvm
