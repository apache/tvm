/*!
 *  Copyright (c) 2019 by Contributors
 * \file gnf_printer.cc
 * \brief GNF printer for Relay programs
 * Supports GNF and metadata.
 */
#include <tvm/relay/doc.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class GNFPrinter :
    public ExprFunctor<Doc(const Expr&)> {
  public:
    explicit GNFPrinter() {}

    Doc PrintFinal(const NodeRef& node) {
      Print(node);
      return doc + TempVar(temp_var_counter_ - 1);
    }

    Doc Print(const NodeRef& node) {
      if (node.as_derived<ExprNode>()) {
        return this->PrintExpr(Downcast<Expr>(node));
      } else { assert(false); }
    }

    Doc TempVar(int n) {
      std::ostringstream os;
      os << n;
      return Text("\%") + Text(os.str());
    }

    Doc AllocTemp() {
      return TempVar(temp_var_counter_++);
    }

    Doc PrintExpr(const Expr& expr) {
      // Exploit memoization to print GNF.
      // The first time we visit an expression, we need to allocate a temp var
      // for it. Every subsequent time we can just use its assigned variable.
      // This works since hashing uses pointer equality.
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      Doc printed_expr = this->VisitExpr(expr);
      Doc temp_var = AllocTemp();
      memo_[expr] = temp_var;
      doc << temp_var << " = " << printed_expr << "\n";
      return temp_var;
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



  private:
    /*! \brief Map from Expr to Doc */
    Doc doc = Nil();
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
    size_t temp_var_counter_{0};
};

std::string RelayGNFPrint(const NodeRef& node) {
  return "v0.0.1\n" + Layout(GNFPrinter().PrintFinal(node));
}

TVM_REGISTER_API("relay._expr.gnf_print")
.set_body_typed<std::string(const NodeRef&)>(RelayGNFPrint);

} // relay
} // tvm
