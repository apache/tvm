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

/*!
 * \brief Meta data context for TextPrinter.
 *
 * This is an important part to enable bi-directional serializability.
 * We use tvm's Node system to build the current IR.
 * It can be hard to design a text format for all the possible nodes
 * as the set of nodes can grow when we do more extensions.
 *
 * Instead of trying to design readable text format for every node,
 * we support a meta data section in the text format.
 * We allow the text format to refer to a node in the meta data section.
 *
 * The meta data section is a json serialized string of an Map<string, Array<NodeRef>>.
 * Each element in the meta data section can be referenced by the text format.
 * Each meta data node is printed in the following format.
 *
 * meta[type-key-of-node>][<index-in-meta-section>]
 *
 * Specifically, consider the following IR(constructed by python).
 *
 * \code
 *
 * n = tvm.var("n")
 * x = tvm.relay.var("x", shape=(n, 1))
 * f = tvm.relay.Function([x], x)
 * print(f.astext())
 *
 * \endcode
 *
 * The corresponding text format is shown in the following code block.
 *
 * \code
 *
 * fn (%x: Tensor[(meta[Variable][0],), float32]) {
 *   %x
 * }
 * # Meta data section is a json-serialized string
 * # of the following array.
 * # [tvm.var("n")]
 *
 * \endcode
 *
 * Note that we store tvm.var("n") in the meta data section.
 * Since it is stored in the index-0 in the meta data section,
 * we print it as meta[Variable][0].
 *
 * The text parser can recover this object by loading from the corresponding
 * location in the meta data section.
 *
 * This is is a design trade-off.
 * It allows us to embedded any meta data in the text format,
 * while still being able to tweak the text part of the printed IR easily.
 */
class TextMetaDataContext {
 public:
  /*!
   * \brief Get text representation of meta node.
   * \param node The node to be converted to meta node.
   * \return A string representation of the meta node.
   */
  Doc GetMetaNode(const NodeRef& node) {
    auto it = meta_repr_.find(node);
    if (it != meta_repr_.end()) {
      return it->second;
    }
    Array<NodeRef>& mvector =
        meta_data_[node->type_key()];
    int64_t index = static_cast<int64_t>(mvector.size());
    mvector.push_back(node);
    Doc doc = Nil();
    doc << "meta[" << node->type_key() << "][" << index << "]";
    meta_repr_[node] = doc;
    return meta_repr_[node];
  }
  /*!
   * \brief Get the metadata section in json format.
   * \return the meta datastring.
   */
  std::string GetMetaSection() const {
    if (meta_data_.size() == 0) return std::string();
    return SaveJSON(Map<std::string, NodeRef>(
        meta_data_.begin(), meta_data_.end()));
  }

  /*! \return whether the meta data context is empty. */
  bool empty() const {
    return meta_data_.empty();
  }

 private:
  /*! \brief additional metadata stored in TVM json format */
  std::unordered_map<std::string, Array<NodeRef> > meta_data_;
  /*! \brief map from meta data into its string representation */
  std::unordered_map<NodeRef, Doc, NodeHash, NodeEqual> meta_repr_;
};

class PrettyPrinter :
    public ExprFunctor<Doc(const Expr&)>,
    public TypeFunctor<Doc(const Type&)>,
    public AttrFunctor<Doc(const NodeRef&)> {
  public:
    explicit PrettyPrinter() : temp_var_counter_(0), GNF_(true) {}
    explicit PrettyPrinter(bool GNF, bool show_meta_data) : show_meta_data_(show_meta_data), temp_var_counter_(0), GNF_(GNF) {}

    // indent a new body
    Doc PrintBody(const NodeRef& node, int indent = 2) {
      Doc doc = Nil();
      Doc body = Nil();
      doc << "{";
      doc << Indent(indent, body << "\n" << PrintScope(node)) << "\n";
      doc << "}";
      return doc;
    }

    // create a new scope by creating a new printer object. This allows temp var
    // numbers to be reused and prevents hoisted vars from escaping too far
    Doc PrintScope(const NodeRef& node) {
      if (GNF_) {
        // print in a new scope
        doc_stack_.push_back(Nil());
        // must print first so doc_stack_.back() reference doesn't become stale
        Doc doc = Print(node, false);
        doc = doc_stack_.back() << doc;
        doc_stack_.pop_back();
        return doc;
      } else {
        return Print(node);
      }
    }

    Doc PrintFinal(const NodeRef& node) {
      Doc doc = Nil();
      doc << PrintScope(node);
      if (!meta_.empty()) {
        if (show_meta_data_) {
          std::string meta_json = meta_.GetMetaSection();
          // append meta data in the end.
          doc << "/* meta data */" << "\n" << meta_json << "\n";
        } else {
          doc << "// meta data omitted. you can use show_meta_data=True to include meta data\n";
        }
      };
      return doc;
    }

    Doc PrintAttrs(const Attrs& attrs);

    // note: gnf flag is only one level deep
    Doc Print(const NodeRef& node, bool gnf = true, bool meta = false) {
      if (node.as_derived<ExprNode>()) {
        return PrintExpr(Downcast<Expr>(node), gnf, meta);
      } else if (node.as_derived<TypeNode>()) {
        return PrintType(Downcast<Type>(node), meta);
      } else if (node.as_derived<ModuleNode>()) {
        return PrintMod(Downcast<Module>(node));
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
      if (var->type_annotation.defined()) {
        val << ": " << Print(var->type_annotation);
      }
      return val;
    }

    //------------------------------------
    // Overload of Expr printing functions
    //------------------------------------
    Doc PrintExpr(const Expr& expr, bool gnf, bool meta) {
      // Exploit memoization to print GNF.
      // The first time we visit an expression, we need to allocate a temp var
      // for it. Every subsequent time we can just use its assigned variable.
      // This works since hashing uses pointer equality.
      auto it = memo_.find(expr);
      if (it != memo_.end()) return it->second;
      Doc printed_expr;
      if (meta) {
        printed_expr = meta_.GetMetaNode(GetRef<NodeRef>(expr.get()));
        std::cerr << Layout(printed_expr) << "\n";
        assert(false);
      } else {
        printed_expr = VisitExpr(expr);
      }
      // we choose to inline some nodes
      if (GNF_ && gnf && !expr.as<GlobalVarNode>() && !expr.as<ConstantNode>() && !expr.as<OpNode>() && !expr.as<VarNode>()) {
        Doc temp_var = AllocTemp();
        memo_[expr] = temp_var;
        doc_stack_.back() << temp_var << " = " << printed_expr << "\n";
        return temp_var;
      } else {
        memo_[expr] = printed_expr;
        if (expr.as<VarNode>()) {
          doc_stack_.back() << "free_var " << printed_expr << "\n";
        }
        return printed_expr;
      }
    }

    Doc VisitExpr_(const VarNode* op) final {
      return AllocVar(GetRef<Var>(op));
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
      // default fall-back, record it as meta node.
      return Print(GetRef<NodeRef>(op), true, true);
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
      doc << "let " << AllocVar(op->var) << " = ";
      if (op->value.as<LetNode>()) {
        doc << PrintBody(op->value);
      } else {
        // we use ANF mode for the first level of the value position so the
        // final expression isn't hoisted or added to the doc stream
        doc << Print(op->value, false);
      }
      doc << ";" << "\n";
      // we use a nested scope here so GNF hoisting doesn't escape too far
      // and so consecutive lets don't get hoisted
      doc << PrintScope(op->body);
      return doc;
    }

    Doc PrintFunc(const Doc& prefix, const Function& fn) {
        // TODO(tqchen, M.K.) support generic function
        // Possibly through meta data
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
        if (fn->ret_type.defined()) {
          doc << "-> " << Print(fn->ret_type) << " ";
        }
        doc << PrintBody(fn->body);
        return doc;
    }

    Doc PrintMod(const Module& mod) {
      Doc doc = Nil();
      int counter = 0;
      for (const auto& kv : mod->functions) {
        std::ostringstream os;
        if (counter++ != 0) {
          doc << "\n";
        }
        os << "def @" << kv.first->name_hint;
        doc << PrintFunc(Text(os.str()), kv.second);
        doc << "\n";
      }
      return doc;
    }

    Doc VisitExpr_(const FunctionNode* op) final {
      return PrintFunc(Text("fn "), GetRef<Function>(op));
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
    Doc PrintType(const Type& type, bool meta) {
      auto it = memo_type_.find(type);
      if (it != memo_type_.end()) return it->second;
      Doc printed_type;
      if (meta) {
        printed_type = meta_.GetMetaNode(GetRef<NodeRef>(type.get()));
      } else {
        printed_type = VisitType(type);
      }
      memo_type_[type] = printed_type;
      return printed_type;
    }

    Doc VisitTypeDefault_(const Node* node) final {  // NOLINT(*)
      // by default always print as meta data
      return Print(GetRef<NodeRef>(node), true, true);
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

    Doc VisitType_(const TupleTypeNode* node) final {
      std::vector<Doc> fields;
      for (Type field : node->fields) {
        fields.push_back(Print(field));
      }
      Doc doc = Nil();
      doc << "(" << PrintVec(fields);
      // conform to python tuple format (1,)
      if (node->fields.size() == 1) {
        doc << ",";
      }
      return doc << ")";
    }

    Doc VisitType_(const FuncTypeNode* node) final {
      Doc doc = Nil();
      std::vector<Doc> arg_types;
      for (Type arg_type : node->arg_types) {
        arg_types.push_back(Print(arg_type));
      }
      return doc << "fn (" << PrintVec(arg_types) << ") -> " << Print(node->ret_type);
    }

    //------------------------------------
    // Overload of Attr printing functions
    //------------------------------------

    Doc PrintAttr(const NodeRef& value, bool meta = false) {  // NOLINT(*)
      if (value.defined()) {
        Doc printed_attr;
        if (meta) {
          printed_attr = meta_.GetMetaNode(value);
        } else {
          printed_attr = VisitAttr(value);
        }
        return printed_attr;
      } else {
        return Text("None");
      }
    }

    Doc VisitAttrDefault_(const Node* op) final { // NOLINT(*)
      return PrintAttr(GetRef<NodeRef>(op), true);
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
    /*! \brief Whether to print meta data. */
    bool show_meta_data_;
    /*! \brief Stack of docs to implement scoped GNFing. */
    std::vector<Doc> doc_stack_{};
    /*! \brief Map from Expr to Doc */
    std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
    /*! \brief Map from Type to Doc */
    std::unordered_map<Type, Doc, NodeHash, NodeEqual> memo_type_;
    std::unordered_map<std::string, int> name_alloc_map_;
    /*! \brief meta data context */
    TextMetaDataContext meta_;
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
  Doc doc = Nil();
  doc << "v0.0.1" << "\n" << PrettyPrinter(true, false).PrintFinal(node) << "\n";
  return Layout(doc);
}

std::string RelayANFPrint(const NodeRef& node) {
  return "v0.0.1\n" + Layout(PrettyPrinter(false, false).PrintFinal(node)) + "\n";
}

std::string RelayPrettyPrint(const NodeRef& node, bool gnf, bool show_meta_data) {
  Doc doc = Nil();
  doc << "v0.0.1" << "\n" << PrettyPrinter(gnf, show_meta_data).PrintFinal(node) << "\n";
  return Layout(doc);
}

TVM_REGISTER_API("relay._expr.gnf_print")
.set_body_typed<std::string(const NodeRef&)>(RelayGNFPrint);

TVM_REGISTER_API("relay._expr.anf_print")
.set_body_typed<std::string(const NodeRef&)>(RelayANFPrint);

TVM_REGISTER_API("relay._expr.pretty_print")
.set_body_typed<std::string(const NodeRef&, bool, bool)>(RelayPrettyPrint);

} // relay
} // tvm
