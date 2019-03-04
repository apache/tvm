/*!
 *  Copyright (c) 2019 by Contributors
 * \file pretty_printer.cc
 * \brief Pretty printer for Relay programs
 * Supports ANF, GNF, and metadata.
 */
#include "doc.h"
#include <tvm/relay/expr_functor.h>
#include "type_functor.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {

class PrettyPrinter :
    public ExprFunctor<Doc(const Expr&)>,
    public TypeFunctor<Doc(const Type&)>,
    public AttrFunctor<Doc(const NodeRef&)> {
  public:
    explicit PrettyPrinter(const std::unordered_map<Expr, Doc, NodeHash, NodeEqual>& memo_, const std::unordered_map<Type, Doc, NodeHash, NodeEqual>& memo_type_, const std::unordered_map<std::string, int>& name_alloc_map_, size_t temp_var_counter_, bool GNF_) : memo_(memo_), memo_type_(memo_type_), name_alloc_map_(name_alloc_map_), temp_var_counter_(temp_var_counter_), GNF_(GNF_) {}

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

    Doc PrintAttrs(const Attrs& attrs);

    // note: gnf flag is only one level deep
    Doc Print(const NodeRef& node, bool gnf = true) {
      if (node.as_derived<ExprNode>()) {
        return PrintExpr(Downcast<Expr>(node), gnf);
      } else if (node.as_derived<TypeNode>()) {
        return PrintType(Downcast<Type>(node));
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

    //------------------------------------
    // Overload of Expr printing functions
    //------------------------------------
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
      return doc << "(" << PrintVec(fields) << ")";
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
        doc << PrintVec(params) << PrintAttrs(fn->attrs);
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
      return doc << "(" << PrintVec(args) << PrintAttrs(op->attrs) << ")";
    }

    //------------------------------------
    // Overload of Type printing functions
    //------------------------------------
    Doc PrintType(const Type& type) {
      auto it = memo_type_.find(type);
      if (it != memo_type_.end()) return it->second;
      Doc printed_type = VisitType(type);
      memo_type_[type] = printed_type;
      return printed_type;
    }

    Doc VisitType_(const TensorTypeNode* node) final {  // NOLINT(*)
    // scalar type
    if (node->shape.size() == 0) {
      return PrintDType(node->dtype);
    }
    Doc doc = Nil();
    doc << "Tensor[(";
    std::vector<Doc> shapes;
    for (NodeRef shape : node->shape) {
      shapes.push_back(PrintAttr(shape));
    }
    doc << PrintVec(shapes);
    // conform to python tuple format (1,)
    if (node->shape.size() == 1) {
      doc << ",";
    }
    return doc << "), " << PrintDType(node->dtype) << "]";
  }

  //------------------------------------
  // Overload of Attr printing functions
  //------------------------------------

  Doc PrintAttr(const NodeRef& value) {  // NOLINT(*)
    if (value.defined()) {
      return VisitAttr(value);
    } else {
      return Text("None");
    }
  }

  Doc VisitAttr_(const ArrayNode* op) final {  // NOLINT(*)
    Doc doc = Nil();
    doc << "[";
    std::vector<Doc> arr_vals;
    for (NodePtr<Node> val : op->data) {
      arr_vals.push_back(PrintAttr(NodeRef(val)));
    }
    doc << PrintVec(arr_vals);
    doc << "]";
    return doc;
  }

  Doc VisitAttrDefault_(const Node* op) final { // NOLINT(*)
    // os << meta_.GetMetaNode(GetRef<NodeRef>(op));
    assert(false);
  }

  Doc VisitAttr_(const ir::IntImm* op) final {  // NOLINT(*)
    return PrintConstScalar(op->type, &(op->value));
  }

  Doc VisitAttr_(const ir::UIntImm* op) final {  // NOLINT(*)
    return PrintConstScalar(op->type, &(op->value));
  }

  Doc VisitAttr_(const ir::FloatImm* op) final {  // NOLINT(*)
    return PrintConstScalar(op->type, &(op->value));
  }

  Doc VisitAttr_(const ir::StringImm* op) final {  // NOLINT(*)
    return PrintString(op->value);
  }

  private:
    /*! \brief Stack of docs to implement scoped GNFing. */
    std::vector<Doc> doc_stack_{Nil()};
    /*! \brief Map from Expr to Doc */
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
    /*! \brief Map from Type to Doc */
    std::unordered_map<Type, Doc, NodeHash, NodeEqual> memo_type_;
    std::unordered_map<std::string, int> name_alloc_map_;
    size_t temp_var_counter_;
    bool GNF_;
    class AttrPrinter;
    friend class AttrPrinter;
};

/*!
 * \brief Attribute printer which prints the attributes in the call.
 */
class PrettyPrinter::AttrPrinter : public AttrVisitor {
 public:
  AttrPrinter(Doc& doc_, PrettyPrinter* parent_) : doc_(doc_), parent_(parent_) {}

  template<typename T>
  Doc PrintKV(const char* key, const T& value) {
    Doc doc = Nil();
    return doc << ", " << key << "=" << value;
  }

  void Visit(const char* key, double* value) final {
    doc_ << PrintKV(key, value[0]);
  }
  void Visit(const char* key, int64_t* value) final {
    doc_ << PrintKV(key, value[0]);
  }
  void Visit(const char* key, uint64_t* value) final {
    doc_ << PrintKV(key, value[0]);
  }
  void Visit(const char* key, int* value) final {
    doc_ << PrintKV(key, value[0]);
  }
  void Visit(const char* key, bool* value) final {
    doc_ << PrintKV(key, PrintBool(value[0]));
  }
  void Visit(const char* key, std::string* value) final {
    doc_ << PrintKV(key, PrintString(value[0]));
  }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "do not allow void as argument";
  }
  void Visit(const char* key, DataType* value) final {
    doc_ << PrintKV(key, PrintString(runtime::TVMType2String(Type2TVMType(value[0]))));
  }
  void Visit(const char* key, NodeRef* value) final {
    doc_ << PrintKV(key, parent_->PrintAttr(value[0]));
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "do not allow NDarray as argument";
  }

  private:
    Doc& doc_;
    PrettyPrinter* parent_;
};

Doc PrettyPrinter::PrintAttrs(const Attrs& attrs) {  // NOLINT(*)
  // TODO: meta
  if (!attrs.defined()) return Nil();
  Doc doc = Nil();
  AttrPrinter printer(doc, this);
  const_cast<BaseAttrsNode*>(attrs.operator->())->VisitNonDefaultAttrs(&printer);
  return doc;
}

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
