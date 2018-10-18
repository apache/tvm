/*!
 *  Copyright (c) 2018 by Contributors
 * \file text_printer.cc
 * \brief Text printer to print relay in text form.
 */
#include <tvm/relay/environment.h>
#include <tvm/relay/expr_functor.h>
#include <sstream>
#include "../pass/type_functor.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {

/*!
 * \brief the text value used in text printer.
 * Defined as a struct for future compatibility reason
 */
struct TextValue {
  /*! \brief The str representation */
  std::string name;
  // constructor
  TextValue() {}
  // constructor
  explicit TextValue(std::string name) : name(name) {}
};

// operator overloading
inline std::ostream& operator<<(std::ostream& os, const TextValue& val) {  // NOLINT(*)
  return os << val.name;
}

/*!
 * \brief Meta data context for TextPrinter.
 *
 * This is an important part to enable bi-directional serializability.
 * We use tvm's Node system to build the current IR.
 * It can be hard to design a text format for all the possible nodes
 * as the set of nodes can grow when we do more extensions.
 *
 * Instead of trying to design readable text format for every nodes,
 * we support a meta-data section in the text format.
 * We allow the text format to refer to a node in the meta-data section.
 *
 * The meta-data section is a json serialized string of an Array<NodeRef>.
 * Each element in the meta-data section can be referenced by the text format.
 * Each meta data node is printed in the following format.
 *
 * meta.<type-key-of-node>(<index-in-meta-section>)
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
 * function(%x: Tensor[(meta.Variable(id=0),), float32]) {
 *   %x
 * }
 * # Meta data section is a json-serialized string
 * # of the following array.
 * # [tvm.var("n")]
 *
 * \endcode
 *
 * Note that we store tvm.var("n") in the meta data section.
 * Since it is stored in the index-0 in the meta-data seciton,
 * we print it as meta.Variable(0).
 *
 * The text parser can recover this object by loading from the corresponding
 * location in the meta data section.
 *
 * This is is a design trade-off.
 * It allows us to embedded any meta-data in the text format,
 * while still being able to tweak the text part of the printed IR easily.
 */
class TextMetaDataContext {
 public:
  /*!
   * \brief Get text representation of meta node.
   * \param node The node to be converted to meta node.
   * \return A string representation of the meta node.
   */
  std::string GetMetaNode(const NodeRef& node) {
    std::ostringstream os;
    auto it = meta_index_.find(node);
    int64_t index;
    if (it != meta_index_.end()) {
      index = it->second;
    } else {
      index = static_cast<int64_t>(meta_data_.size());
      meta_data_.push_back(node);
      meta_index_[node] = index;
    }
    os << "meta." << node->type_key() << "(id=" << index << ")";
    return os.str();
  }
  /*!
   * \brief Get the metadata section in json format.
   * \return the meta datastring.
   */
  std::string GetMetaSection() const {
    if (meta_data_.size() == 0) return std::string();
    return SaveJSON(Array<NodeRef>(meta_data_));
  }

 private:
  /*! \brief additional metadata stored in TVM json format */
  std::vector<NodeRef> meta_data_;
  /*! \brief map from meta data into its index */
  std::unordered_map<NodeRef, int64_t, NodeHash, NodeEqual> meta_index_;
};

class TextPrinter :
    public ExprFunctor<TextValue(const Expr&)> ,
    public TypeFunctor<void (const Type&, std::ostream& os)>,  // NOLINT(*)
    public AttrFunctor<void (const NodeRef&, std::ostream& os)> { // NOLINT(*)
 public:
  /*!
   * \brief Print a node to string.
   * \param node.
   * \return The string representation.
   */
  std::string Print(const NodeRef& node) {
    if (node.as<FunctionNode>()) {
      this->PrintFunc(Downcast<Function>(node));
    } else if (node.as<EnvironmentNode>()) {
      this->PrintEnv(Downcast<Environment>(node));
    } else if (node.as_derived<TypeNode>()) {
      this->PrintType(Downcast<Type>(node), stream_);
    } else if (node.as_derived<ExprNode>()) {
      this->PrintExpr(Downcast<Expr>(node));
    } else {
      stream_ << node;
    }
    std::string meta_json = meta_.GetMetaSection();
    if (meta_json.length() != 0) {
      // append meta data in the end.
      stream_ << "# meta data\n"
              << "r\"\"\"\n"
              << meta_json << "\n"
              << "\"\"\"";
    }
    return stream_.str();
  }

  void PrintFunc(const Function& func) {
    this->PrintFuncInternal("function", func);
    stream_ << "\n";
  }

  void PrintEnv(const Environment& env) {
    int counter = 0;
    for (const auto& kv : env->functions) {
      std::ostringstream os;
      if (counter++ != 0) {
        stream_ << "\n";
      }
      os << "def @" << kv.first->name_hint;
      this->PrintFuncInternal(os.str(), kv.second);
      stream_ << "\n";
    }
  }

  void PrintExpr(const Expr& expr) {
    TextValue val = GetValue(expr);
    stream_ << val << "\n";
  }

  /*!
   * \brief Get text representation of expr.
   *
   * This function may generate additional instructions
   * in order to compute the final result id of expr.
   *
   * When trying to recursively print out an Expr.
   * The caller should always call GetValue of its children first.
   * Then the caller can print out to stream_ using the obtained value.
   *
   * This is to avoid the call of subsequent GetValue print out
   * additional instructions which get mixed with the partial instruction
   * printed by the caller.
   *
   * \param expr The input expression.
   * \return The text value of Expr.
   */
  TextValue GetValue(const Expr& expr) {
    auto it = memo_.find(expr);
    if (it != memo_.end()) return it->second;
    TextValue val = this->VisitExpr(expr);
    memo_[expr] = val;
    return val;
  }
  //------------------------------------
  // Overload of Expr printing functions
  //------------------------------------
  TextValue VisitExpr_(const ConstantNode* op) final {
    // Print out simple scalar directly.
    if (op->is_scalar()) {
      std::ostringstream os;
      DataType dtype = TVMType2Type(op->data->dtype);
      CHECK_EQ(op->data->ctx.device_type, kDLCPU);
      if (dtype == Int(32)) {
        return ConstScalar(dtype, static_cast<const int32_t*>(op->data->data));
      } else if (dtype == Int(64)) {
        return ConstScalar(dtype, static_cast<const int64_t*>(op->data->data));
      } else if (dtype == Float(32)) {
        return ConstScalar(dtype, static_cast<const float*>(op->data->data));
      } else if (dtype == Float(64)) {
        return ConstScalar(dtype, static_cast<const double*>(op->data->data));
      }
    }
    // default fall-back, record it as meta node.
    TextValue id = this->AllocTempVar();
    this->PrintIndent();
    stream_ << id << " = " << meta_.GetMetaNode(GetRef<NodeRef>(op));
    this->PrintEndInst("\n");
    return id;
  }

  TextValue VisitExpr_(const TupleNode* op) final {
    std::vector<TextValue> fields;
    for (Expr field : op->fields) {
      fields.push_back(GetValue(field));
    }
    // NOTE: always recursively visit to get ids,
    // before print out the current line
    TextValue id = this->AllocTempVar();
    this->PrintIndent();
    stream_ << id << " = (";
    for (size_t i = 0; i < fields.size(); ++i) {
      stream_ << fields[i];
      if (i + 1 != fields.size()) {
        stream_ << ", ";
      }
    }
    stream_ << ')';
    this->PrintEndInst("\n");
    return id;
  }

  TextValue VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    // This is an unbounded var.
    TextValue val = AllocVarName(var);
    this->PrintIndent();
    stream_ << "free_var ";
    this->PrintVarDecl(var, stream_);
    this->PrintEndInst("\n");
    return val;
  }

  TextValue VisitExpr_(const GlobalVarNode* op) final {
    return TextValue('@' + op->name_hint);
  }

  TextValue VisitExpr_(const FunctionNode* op) final {
    TextValue id = AllocTempVar();
    std::ostringstream os;
    os << id << " = function";
    this->PrintFuncInternal(os.str(), GetRef<Function>(op));
    this->PrintEndInst("\n");
    return id;
  }

  TextValue VisitExpr_(const CallNode* op) final {
    // TODO(tqchen, M.K.): support generic call
    // possibly through meta-data
    CHECK_EQ(op->type_args.size(), 0U)
        << "generic call not yet supported";
    TextValue call_op = GetValue(op->op);
    std::vector<TextValue> args;
    for (Expr arg : op->args) {
      args.emplace_back(GetValue(arg));
    }
    TextValue id = this->AllocTempVar();
    this->PrintIndent();
    stream_ << id << " = " << call_op << "(";
    for (size_t i = 0; i < args.size(); ++i) {
      stream_ << args[i];
      if (i + 1 != args.size()) {
        stream_ << ", ";
      }
    }
    this->PrintCallAttrs(op->op, op->attrs, stream_);
    stream_ << ")";
    this->PrintEndInst("");
    this->PrintOptionalInfo(GetRef<Expr>(op));
    stream_ << '\n';
    return id;
  }

  TextValue VisitExpr_(const LetNode* op) final {
    TextValue id = this->AllocTempVar();
    this->PrintIndent();
    stream_ << id << " = ";
    this->PrintScope(GetRef<Expr>(op));
    this->PrintEndInst("\n");
    return id;
  }

  TextValue VisitExpr_(const IfNode* op) final {
    TextValue id = this->AllocTempVar();
    this->PrintIndent();
    stream_ << id << " = ";
    this->PrintScope(GetRef<Expr>(op));
    this->PrintEndInst("\n");
    return id;
  }

  TextValue VisitExpr_(const OpNode* op) final {
    return TextValue(op->name);
  }

  TextValue VisitExpr_(const TupleGetItemNode* op) final {
    TextValue tuple = GetValue(op->tuple);
    TextValue id = this->AllocTempVar();
    this->PrintIndent();
    stream_ << id << " = " << tuple << "[" << op->index << "]";
    this->PrintEndInst("\n");
    return id;
  }

  /*!
   * \brief Print the type to os
   * \param type The type to be printed.
   * \param os The output type.
   */
  void PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
    this->VisitType(type, os);
  }
  //------------------------------------
  // Overload of Expr printing functions
  //------------------------------------
  void VisitType_(const TensorTypeNode* node, std::ostream& os) final {  // NOLINT(*)
    // scalar type
    if (node->shape.size() == 0) {
      os << runtime::TVMType2String(Type2TVMType(node->dtype));
      return;
    }
    os << "Tensor[(";
    for (size_t i = 0; i < node->shape.size(); ++i) {
      this->PrintAttr(node->shape[i], os);
      if (i + 1 != node->shape.size()) {
        os << ", ";
      }
    }
    // conform to python tuple format (1,)
    if (node->shape.size() == 1) {
      os << ",";
    }
    os << "), " << runtime::TVMType2String(Type2TVMType(node->dtype)) << "]";
  }

  void VisitTypeDefault_(const Node* node, std::ostream& os) final {  // NOLINT(*)
    // by default always print as meta-data
    os << meta_.GetMetaNode(GetRef<NodeRef>(node));
  }

  /*!
   * \brief Print an attribute value to os.
   * \param value The value to be printed.
   * \param os The output type.
   */
  void PrintAttr(const NodeRef& value, std::ostream& os) {  // NOLINT(*)
    this->VisitAttr(value, os);
  }
  //------------------------------------
  // Overload of Attr printing functions
  //------------------------------------
  void VisitAttr_(const ArrayNode* op, std::ostream& os) final {  // NOLINT(*)
    os << "[";
    for (size_t i = 0; i < op->data.size(); ++i) {
      this->PrintAttr(NodeRef(op->data[i]), os);
      if (i + 1 != op->data.size()) {
        os << ", ";
      }
    }
    os << "]";
  }
  void VisitAttrDefault_(const Node* op, std::ostream& os) final { // NOLINT(*)
    os << meta_.GetMetaNode(GetRef<NodeRef>(op));
  }

  void VisitAttr_(const ir::IntImm* op, std::ostream& os) final {  // NOLINT(*)
    this->PrintConstScalar(op->type, &(op->value), os);
  }

  void VisitAttr_(const ir::UIntImm* op, std::ostream& os) final {  // NOLINT(*)
    this->PrintConstScalar(op->type, &(op->value), os);
  }

  void VisitAttr_(const ir::FloatImm* op, std::ostream& os) final {  // NOLINT(*)
    this->PrintConstScalar(op->type, &(op->value), os);
  }

  void VisitAttr_(const ir::StringImm* op, std::ostream& os) final {  // NOLINT(*)
    this->PrintString(op->value, os);
  }

 protected:
  /*!
   * \brief Print attributes after call.
   * \param op The operator to be called.
   * \param attrs The attributes.
   * \param os The output stream.
   */
  void PrintCallAttrs(const Expr& op, const Attrs& attrs, std::ostream& os);  // NOLINT(*)

  /*!
   * \brief Print the a new scopr.
   * \param body The body.
   */
  void PrintScope(Expr body) {
    stream_ << "{\n";
    int sid = this->BeginScope();
    this->PrintScopeBody(body);
    this->EndScope(sid);
    this->PrintIndent();
    stream_ << "}";
  }
  /*!
   * \brief Print the body of a new scope without {}
   *
   * This function will keep printing continuous sequence
   * of let/if scope without introducing a new scope in the text.
   *
   * \param body The body.
   */
  void PrintScopeBody(Expr body) {
    if (const LetNode* let = body.as<LetNode>()) {
      TextValue value = GetValue(let->value);
      AllocVarName(let->var);
      // let var = value;
      this->PrintIndent();
      stream_ << "let ";
      this->PrintVarDecl(let->var, stream_);
      stream_ << " = " << value;
      this->PrintEndInst("\n");
      this->PrintScopeBody(let->body);
    } else if (const IfNode* ifnode = body.as<IfNode>()) {
      TextValue cond = GetValue(ifnode->cond);
      this->PrintIndent();
      stream_ << "if (" << cond << ") ";
      this->PrintScope(ifnode->true_branch);
      this->PrintIndent();
      stream_ << "else ";
      this->PrintScope(ifnode->false_branch);
      this->PrintEndInst("\n");
    } else {
      TextValue value = GetValue(body);
      this->PrintIndent();
      stream_ << value;
      this->PrintEndInst("\n");
    }
  }

  /*!
   * \brief Internal function to print a function argument list and its body.
   * \param prefix The prefix before argument list.
   * \param fn The function to be printed.
   */
  void PrintFuncInternal(std::string prefix, const Function& fn) {
    // TODO(tqchen, M.K.) support generic function
    // Possibly through meta-data
    CHECK_EQ(fn->type_params.size(), 0U)
        << "generic fn not yet supported";
    this->PrintIndent();
    stream_ << prefix << "(";
    size_t decl_indent = prefix.length() + 1;
    for (size_t i = 0; i < fn->params.size(); ++i) {
      if (i != 0) {
        this->PrintIndent(decl_indent);
      }
      AllocVarName(fn->params[i]);
      this->PrintVarDecl(fn->params[i], stream_);
      if (i + 1 != fn->params.size()) {
        stream_ << ",\n";
      }
    }
    stream_ << ") ";
    if (fn->ret_type.defined()) {
      stream_ << " -> ";
      this->PrintType(fn->ret_type, stream_);
    }
    this->PrintScope(fn->body);
  }
  /*!
   * \brief Print additional info about expr in comment.
   * \param expr The expression.
   */
  void PrintOptionalInfo(const Expr& expr) {
    // additional information in comment.
    if (expr->checked_type_.defined()) {
      stream_ << " # ty=";
      this->PrintType(expr->checked_type(), stream_);
    }
  }
  /*!
   * \brief print var_name[:type]
   * \param var The variable to be printed
   * \param os The output stream
   */
  void PrintVarDecl(const Var& var, std::ostream& os) {  // NOLINT(*)
    TextValue v = GetValue(var);
    os << v;
    if (var->type_annotation.defined()) {
      os << ": ";
      this->PrintType(var->type_annotation, os);
    }
  }
  /*!
   * \brief Get a constant scalar value.
   * \param dtype The data type.
   * \param data The pointer to the data.
   * \tparam T the content data type holding the data.
   */
  template<typename T>
  TextValue ConstScalar(DataType dtype, const T* data) {
    std::ostringstream os;
    PrintConstScalar(dtype, data, os);
    return TextValue(os.str());
  }
  /*!
   * \brief special method to print out const scalar
   * \param dtype The data type
   * \param data The pointer to hold the data.
   * \param os The output stream.
   */
  template<typename T>
  void PrintConstScalar(DataType dtype, const T* data, std::ostream& os) {  // NOLINT(*)
    if (dtype == Int(32)) {
      os << data[0];
    } else if (dtype == Float(32)) {
      os << data[0] << 'f';
    } else if (dtype == Bool()) {
      PrintBool(data[0] != 0, os);
    } else {
      os << dtype << "(" << data[0] << ")";
    }
  }
  /*!
   * \brief Print constant bool value.
   * \param value The value to be printed.
   * \param os The output stream
   */
  void PrintBool(bool value, std::ostream& os) { // NOLINT(*)
    if (value) {
      os << "True";
    } else {
      os << "False";
    }
  }
  /*!
   * \brief Print constant string.
   * \param value The value to be printed.
   * \param os The output stream
   */
  void PrintString(const std::string& value, std::ostream& os) { // NOLINT(*)
    // TODO(M.K.): add escape.
    os << "\"" << value << "\"";
  }
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  std::string GetUniqueName(std::string prefix) {
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
    return prefix;
  }
  /*!
   * \brief mark the beginning of a new scope
   * \return The scope id.
   */
  int BeginScope() {
    int sid = static_cast<int>(scope_valid_.size());
    scope_valid_.push_back(true);
    indent_ += 2;
    return sid;
  }
  /*!
   * \brief mark the end of an old scope.
   * \param scope_id The scope id to be ended.
   */
  void EndScope(int scope_id) {
    scope_valid_[scope_id] = false;
    indent_ -= 2;
  }
  /*!
   * \brief Print the indent to the stream.
   * \param more_indent More indentation besides the current one.
   */
  void PrintIndent(int64_t more_indent = 0) {
    for (int i = 0; i < indent_ + more_indent; ++i) {
      stream_ << ' ';
    }
  }
  /*!
   * \brief print end of the line.
   */
  void PrintEndInst(const char* suffix) {
    stream_ << suffix;
  }
  /*!
   * \brief Allocate temporary value
   * \return A new text value.
   */
  TextValue AllocTempVar() {
    std::ostringstream os;
    os << '%' << temp_var_counter_++;
    return TextValue(os.str());
  }
  /*!
   * \brief Allocate name to a variable.
   * \param var The input variable.
   * \return The corresponding name.
   */
  TextValue AllocVarName(const Var& var) {
    std::string name = GetUniqueName('%' + var->name_hint);
    TextValue val(name);
    CHECK(!memo_.count(var));
    memo_[var] = val;
    return val;
  }

 private:
  class AttrPrinter;
  friend class AttrPrinter;
  /*! \brief meta data context */
  TextMetaDataContext meta_;
  /*! \brief Check whether scope is still valid */
  std::vector<bool> scope_valid_;
  /*! \brief The current indentation value */
  int indent_{0};
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief Map from expression to its text value */
  std::unordered_map<Expr, TextValue, NodeHash, NodeEqual> memo_;
  /*! \brief counter of temporary variable */
  int64_t temp_var_counter_{0};
  /*! \brief Output stream */
  std::ostringstream stream_;
};

/*!
 * \brief Attribute printer which prints the attributes in the call.
 */
class TextPrinter::AttrPrinter: public AttrVisitor {
 public:
  AttrPrinter(std::ostream& stream, TextPrinter* parent)  // NOLINT(*)
      : stream_(stream), parent_(parent) {}

  void Visit(const char* key, double* value) final {
    PrintSep();
    stream_ << key << "=" << value[0];
  }
  void Visit(const char* key, int64_t* value) final {
    PrintSep();
    stream_ << key << "=" << value[0];
  }
  void Visit(const char* key, uint64_t* value) final {
    PrintSep();
    stream_ << key << "=" << value[0];
  }
  void Visit(const char* key, int* value) final {
    PrintSep();
    stream_ << key << "=" << value[0];
  }
  void Visit(const char* key, bool* value) final {
    PrintSep();
    stream_ << key << "=";
    parent_->PrintBool(value[0], stream_);
  }
  void Visit(const char* key, std::string* value) final {
    PrintSep();
    stream_ << key << "=";
    parent_->PrintString(value[0], stream_);
  }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "do not allow void as argument";
  }
  void Visit(const char* key, DataType* value) final {
    PrintSep();
    stream_ << key << "=";
    parent_->PrintString(runtime::TVMType2String(Type2TVMType(value[0])), stream_);
  }
  void Visit(const char* key, NodeRef* value) final {
    PrintSep();
    stream_ << key << "=";
    parent_->PrintAttr(value[0], stream_);
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "do not allow NDarray as argument";
  }

 private:
  void PrintSep() {
    stream_ << ", ";
  }
  std::ostream& stream_;  // NOLINT(*)
  TextPrinter* parent_;
};

void TextPrinter::PrintCallAttrs(const Expr& op,
                                 const Attrs& attrs,
                                 std::ostream& os) {  // NOLINT(*)
  if (!attrs.defined()) return;
  if (const auto* op_node = op.as<OpNode>()) {
    if (attrs->type_index() == op_node->attrs_type_index) {
      AttrPrinter printer(os, this);
      const_cast<BaseAttrsNode*>(attrs.operator->())
          ->VisitNonDefaultAttrs(&printer);
      return;
    }
  }
  os << ", " << meta_.GetMetaNode(attrs);
}

std::string RelayPrint(const NodeRef& node) {
  return TextPrinter().Print(node);
}

TVM_REGISTER_API("relay._expr._text_print")
.set_body_typed<std::string(const NodeRef&)>(RelayPrint);

}  // namespace relay
}  // namespace tvm
