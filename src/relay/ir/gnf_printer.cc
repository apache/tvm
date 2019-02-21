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

    std::string Print(const NodeRef& node) {
      if (node.as_derived<ExprNode>()) {
        return Layout(this->PrintExpr(Downcast<Expr>(node)));
      } else { assert(false); }
    }

    const Doc PrintExpr(const Expr& expr) {
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      Doc val = this->VisitExpr(expr);
      memo_[expr] = val;
      return val;
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

  private:
    /*! \brief Map from Expr to Doc */
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
    size_t temp_var_counter_{0};
};

std::string RelayPrettyPrint(const NodeRef& node) {
  return "v0.0.1\n" + GNFPrinter().Print(node);
}

TVM_REGISTER_API("relay._expr.pretty_print")
.set_body_typed<std::string(const NodeRef&)>(RelayPrettyPrint);

} // relay
} // tvm
