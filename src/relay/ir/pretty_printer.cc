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
    explicit PrettyPrinter(const std::unordered_map<Expr, Doc, NodeHash, NodeEqual>& memo_, const std::unordered_map<std::string, int>& name_alloc_map_, size_t temp_var_counter_, bool GNF_) : memo_(memo_), name_alloc_map_(name_alloc_map_), temp_var_counter_(temp_var_counter_), GNF_(GNF_) {}

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

    // create a new scope by creating a new printer object. This allows temp var
    // numbers to be reused and prevents hoisted vars from escaping too far
    Doc PrintNestedScope(const NodeRef& node) {
      if (GNF_) {
        // print in a new scope
        doc_stack_.push_back(Nil());
        Doc doc = PrintFinal(node);
        doc_stack_.pop_back();
        return doc;
      } else {
        return Print(node);
      }
    }

    Doc PrintFinal(const NodeRef& node) {
      // must print first so doc_stack_.back() reference doesn't become stale
      Doc doc = Print(node, false);
      return doc_stack_.back() << doc;
    }

    // note: gnf flag is only one level deep
    Doc Print(const NodeRef& node, bool gnf = true) {
      if (node.as_derived<ExprNode>()) {
        return PrintExpr(Downcast<Expr>(node), gnf);
      } else { assert(false); }
    }

    Doc TempVar(int n) {
      Doc doc = Nil();
      return doc << "%" << n;
    }

    Doc AllocTemp() {
      return TempVar(temp_var_counter_++);
    }

    /*!
     * \brief get a unique name with the corresponding prefix
     * \param prefix The prefix of the name
     * \return The returned name.
     */
    Doc GetUniqueName(std::string prefix) {
      auto it = name_alloc_map_.find(prefix);
      if (it != name_alloc_map_.end()) {
        while (true) {
          std::ostringstream os;
          os << prefix << (++it->second);
          std::string name = os.str();
          if (name_alloc_map_.count(name) == 0) {
            prefix = name;
            break;
          }
        }
      }
      name_alloc_map_[prefix] = 0;
      return Text(prefix);
    }

    /*!
     * \brief Allocate name to a variable.
     * \param var The input variable.
     * \return The corresponding name.
     */
    Doc AllocVar(const Var& var) {
      std::string name = var->name_hint();
      // always make sure first name is alpha
      if (name.length() != 0 && !std::isalpha(name[0])) {
        name = "v" + name;
      }
      Doc val = GetUniqueName("%" + name);
      // still print if ir is malformed, but show the error.
      if (memo_.count(var)) {
        memo_[var] = val + Text("-malformed-ir");
      }
      memo_[var] = val;
      // TODO: should also return type annotation
      return val;
    }

    Doc PrintExpr(const Expr& expr, bool gnf = true) {
      // Exploit memoization to print GNF.
      // The first time we visit an expression, we need to allocate a temp var
      // for it. Every subsequent time we can just use its assigned variable.
      // This works since hashing uses pointer equality.
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      Doc printed_expr = VisitExpr(expr);
      // we choose to inline some nodes
      if (GNF_ && gnf && !expr.as<GlobalVarNode>() && !expr.as<ConstantNode>() && !expr.as<OpNode>()) {
        Doc temp_var = AllocTemp();
        memo_[expr] = temp_var;
        doc_stack_.back() << temp_var << " = " << printed_expr << "\n";
        return temp_var;
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

      // we use ANF mode for the first level of the value position so the final
      // expression isn't hoisted or added to the doc stream
      doc << "let %" << op->var->name_hint() << " = " << Print(op->value, false) << ";" << "\n";
      // we use a nested scope here so GNF hoisting doesn't escape too far
      // and so consecutive lets don't get hoisted
      doc << PrintNestedScope(op->body);
      return doc;
    }

    Doc PrintFunc(const Doc& prefix, const FunctionNode* fn) {
        // TODO(tqchen, M.K.) support generic function
        // Possibly through meta-data
        CHECK_EQ(fn->type_params.size(), 0U)
        << "generic fn not yet supported";
        Doc doc = Nil();
        doc << prefix << "(";
        std::vector<Doc> params;
        for (Var param : fn->params) {
          params.push_back(AllocVar(param));
        }
        doc << PrintVec(params, Text(", "));
        doc << ") ";
        /* if (fn->ret_type.defined()) {
          doc << " -> ";
          this->PrintType(fn->ret_type, stream_);
        } */
        doc << PrintBody(fn->body);
        return doc;
    }

    Doc VisitExpr_(const FunctionNode* op) final {
      return PrintFunc(Text("fn "), op);
    }

    Doc VisitExpr_(const GlobalVarNode* op) final {
      return Text('@' + op->name_hint);
    }

    Doc VisitExpr_(const OpNode* op) final {
      return Text(op->name);
    }

    Doc VisitExpr_(const CallNode* op) final {
      Doc doc = Nil();
      doc << Print(op->op);
      std::vector<Doc> args;
      for (Expr arg : op->args) {
        args.push_back(Print(arg));
      }
      return doc << "(" << PrintVec(args, Text(", ")) << ")";
    }

  private:
    /*! \brief Stack of docs to implement scoped GNFing. */
    std::vector<Doc> doc_stack_{Nil()};
    /*! \brief Map from Expr to Doc */
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
    std::unordered_map<std::string, int> name_alloc_map_;
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
