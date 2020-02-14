/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file text_format_printer.cc
 * \brief Printer to print out the IR text format
 *        that can be parsed by a parser.
 *
 * Supports ANF, GNF in relay and metadata.
 *
 * Inlining heuristics:
 *  - Always inline:
 *    - GlobalVar
 *    - Constant
 *    - Op
 *    - Var
 *  - Otherwise, inline if the node is at the end of a scope and is used at most once.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/ir/module.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include "doc.h"
#include "meta_data.h"
#include "../relay/pass/dependency_graph.h"
#include "../ir/attr_functor.h"

namespace tvm {
namespace relay {

class RelayTextPrinter :
    public ExprFunctor<Doc(const Expr&)>,
    public PatternFunctor<Doc(const Pattern&)>,
    public TypeFunctor<Doc(const Type&)>,
    public AttrFunctor<Doc(const ObjectRef&)> {
 public:
  explicit RelayTextPrinter(bool show_meta_data,
                            runtime::TypedPackedFunc<std::string(ObjectRef)> annotate)
      : show_meta_data_(show_meta_data),
        annotate_(annotate) {}

  /*!
    * \brief Print additional info about expr in comment.
    * \param expr The expression.
    */
  Doc PrintOptionalInfo(const Expr& expr) {
    Doc doc;
    // default annotations
    if (annotate_ == nullptr) {
      if ((expr.as<ConstantNode>() || expr.as<CallNode>()) && expr->checked_type_.defined()) {
        doc << " /* ty=" << Print(expr->checked_type()) << " */";
      }
    } else {
      std::string annotated_expr = annotate_(expr);
      if (annotated_expr != "") {
        doc << annotated_expr;
      }
    }

    return doc;
  }

  // indent a new body
  Doc PrintBody(const ObjectRef& node, int indent = 2) {
    Doc doc;
    Doc body;
    doc << "{";
    doc << Doc::Indent(indent, body << Doc::NewLine() << PrintScope(node)) << Doc::NewLine();
    doc << "}";
    return doc;
  }

  // create a new scope by creating a new printer object. This allows temp var
  // numbers to be reused and prevents hoisted vars from escaping too far
  Doc PrintScope(const ObjectRef& node) {
    // print in a new scope
    doc_stack_.push_back(Doc());
    // must print first so doc_stack_.back() reference doesn't become stale
    Doc doc = Print(node, false, true);
    doc = doc_stack_.back() << doc;
    doc_stack_.pop_back();
    return doc;
  }

  Doc PrintFinal(const ObjectRef& node) {
    if (node.as<ExprNode>()) {
      Expr expr = Downcast<Expr>(node);
      dg_ = DependencyGraph::Create(&arena_, expr);
    }

    Doc doc;
    doc << PrintScope(node);
    if (!meta_.empty()) {
      doc << Doc::NewLine();
      if (show_meta_data_) {
        // append meta data in the end.
        doc << "METADATA:" << Doc::NewLine() << meta_.GetMetaSection();
      } else {
        doc << "// meta data omitted. you can use show_meta_data=True to include meta data";
      }
    }
    return doc;
  }

  std::vector<Doc> PrintCallAttrs(const Attrs& attrs, const Expr& op);
  std::vector<Doc> PrintFuncAttrs(const Attrs& attrs);

  Doc Print(const ObjectRef& node, bool meta = false, bool try_inline = false) {
    if (node.as<ExprNode>()) {
      return PrintExpr(Downcast<Expr>(node), meta, try_inline);
    } else if (node.as<TypeNode>()) {
      return PrintType(Downcast<Type>(node), meta);
    } else if (node.as<PatternNode>()) {
      return PrintPattern(Downcast<Pattern>(node), meta);
    } else if (node.as<IRModuleNode>()) {
      return PrintMod(Downcast<IRModule>(node));
    } else {
      // default module.
      std::ostringstream os;
      os << node;
      return Doc() << os.str();
    }
  }

  Doc TempVar(int n) {
    Doc doc;
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
  Doc GetUniqueName(const std::string& prefix) {
    std::string unique_prefix = prefix;
    auto it = name_alloc_map_.find(prefix);
    if (it != name_alloc_map_.end()) {
      while (true) {
        std::ostringstream os;
        os << prefix << (++it->second);
        std::string name = os.str();
        if (name_alloc_map_.count(name) == 0) {
          unique_prefix = name;
          break;
        }
      }
    }
    name_alloc_map_[unique_prefix] = 0;
    return Doc::Text(unique_prefix);
  }

  Doc Print(Kind k) {
    switch (k) {
    case kType:
      return Doc::Text("Type");
    case kShapeVar:
      return Doc::Text("Shape");
    case kBaseType:
      return Doc::Text("BaseType");
    case kConstraint:
      return Doc::Text("Constraint");
    case kAdtHandle:
      return Doc::Text("AdtHandle");
    case kTypeData:
      return Doc::Text("TypeData");
    default:
      LOG(ERROR) << "Unknown Kind";
      throw;
    }
  }
  /*!
   * \brief Allocate name to a type variable.
   * \param var The input type variable.
   * \return The corresponding name.
   */
  Doc AllocTypeVar(const TypeVar& var) {
    if (memo_type_.count(var)) {
      Doc val = memo_type_[var];
      val << "-malformed-ir";
      return val;
    }
    std::string name = var->name_hint;
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "t" + name;
    }
    Doc val = GetUniqueName(name);
    memo_type_[var] = val;
    if (var->kind != kType) {
      val << ": " << Print(var->kind);
    }
    return val;
  }

  /*!
   * \brief Allocate name to a variable.
   * \param var The input variable.
   * \return The corresponding name.
   */
  Doc AllocVar(const Var& var) {
    // still print if ir is malformed, but show the error.
    if (memo_.count(var)) {
      Doc val = memo_[var];
      val << "-malformed-ir";
      return val;
    }
    std::string name = var->name_hint();
    // always make sure first name is alpha
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "v" + name;
    }
    Doc val = GetUniqueName("%" + name);
    memo_[var] = val;
    if (var->type_annotation.defined()) {
      val << ": " << Print(var->type_annotation);
    }
    return val;
  }

  bool IsUnique(const Expr& expr) {
    auto it = dg_.expr_node.find(expr);
    if (it == dg_.expr_node.end()) {
      return true;
    } else {
      return !(it->second->parents.head && it->second->parents.head->next);
    }
  }

  bool AlwaysInline(const Expr& expr) {
    return expr.as<GlobalVarNode>() || expr.as<ConstantNode>() || expr.as<OpNode>() ||
           expr.as<VarNode>() || expr.as<ConstructorNode>();
  }

  //------------------------------------
  // Overload of Expr printing functions
  //------------------------------------
  Doc PrintExpr(const Expr& expr, bool meta, bool try_inline) {
    // Exploit memoization to print GNF.
    // The first time we visit an expression, we need to allocate a temp var
    // for it. Every subsequent time we can just use its assigned variable.
    // This works since hashing uses pointer equality.

    // determine whether to inline
    bool inline_expr = AlwaysInline(expr);
    if (try_inline) {
      inline_expr |= IsUnique(expr);
    }

    auto it = memo_.find(expr);
    if (it != memo_.end()) return it->second;

    Doc printed_expr;
    if (meta) {
      printed_expr = meta_.GetMetaNode(GetRef<ObjectRef>(expr.get()));
    } else if (!inline_expr && expr.as<LetNode>()) {
      // wrap GNFed let in brackets
      Doc body;
      printed_expr << "(";
      printed_expr << Doc::Indent(2, body << Doc::NewLine() << VisitExpr(expr)) << Doc::NewLine();
      printed_expr << ")";
    } else {
      printed_expr = VisitExpr(expr);
    }

    printed_expr << PrintOptionalInfo(expr);

    // add expr to doc
    if (expr.as<VarNode>()) {
      // This is our first time visiting the var and we hit the VarNode case
      // in the visitor. Thus the variable is free.
      doc_stack_.back() << "free_var " << printed_expr << Doc::NewLine();
      // Memoization is done in AllocVar.
      return memo_[expr];
    } else if (inline_expr) {
      memo_[expr] = printed_expr;
      return printed_expr;
    } else {
      Doc temp_var = AllocTemp();
      memo_[expr] = temp_var;
      doc_stack_.back() << temp_var << " = " << printed_expr << ";" << Doc::NewLine();
      return temp_var;
    }
  }

  // Should only be triggered when op is a free variable being visited for the
  // first time.
  Doc VisitExpr_(const VarNode* op) final {
    return AllocVar(GetRef<Var>(op));
  }

  /*!
   * \brief special method to print out const scalar
   * \param dtype The data type
   * \param value The value to be printed.
   */
  template<typename T>
  static Doc ScalarLiteral(DataType dtype, const T& value) {
    std::ostringstream os;
    if (dtype == DataType::Int(32)) {
      os << value;
    } else if (dtype == DataType::Float(32)) {
      os << value << 'f';
    } else if (dtype == DataType::Float(64)) {
      os << value;
    } else if (dtype == DataType::Bool()) {
      return Doc::PyBoolLiteral(value != 0);
    } else {
      os << value;
    }
    return Doc::Text(os.str());
  }

  Doc VisitExpr_(const ConstantNode* op) final {
    // Print out simple scalars directly.
    if (op->is_scalar()) {
      std::ostringstream os;
      DataType dtype = DataType(op->data->dtype);
      CHECK_EQ(op->data->ctx.device_type, kDLCPU);
      if (dtype == DataType::Int(32)) {
        return ScalarLiteral(dtype, static_cast<const int32_t*>(op->data->data)[0]);
      } else if (dtype == DataType::Int(64)) {
        return ScalarLiteral(dtype, static_cast<const int64_t*>(op->data->data)[0]);
      } else if (dtype == DataType::Float(32)) {
        return ScalarLiteral(dtype, static_cast<const float*>(op->data->data)[0]);
      } else if (dtype == DataType::Float(64)) {
        return ScalarLiteral(dtype, static_cast<const double*>(op->data->data)[0]);
      } else if (dtype == DataType::Bool()) {
        return ScalarLiteral(dtype, static_cast<const uint8_t*>(op->data->data)[0]);
      }
    }
    // default fall-back, record it as meta node.
    Doc doc;
    return doc << Print(GetRef<ObjectRef>(op), true);
  }

  Doc VisitExpr_(const TupleNode* op) final {
    std::vector<Doc> fields;
    for (Expr field : op->fields) {
      fields.push_back(Print(field));
    }
    Doc doc;
    doc << "(" << Doc::Concat(fields);
    // conform to python tuple format (1,)
    if (op->fields.size() == 1) {
      doc << ",";
    }
    return doc << ")";
  }

  Doc VisitExpr_(const TupleGetItemNode* op) final {
    Doc doc;
    return doc << Print(op->tuple) << "." << op->index;
  }

  Doc VisitExpr_(const IfNode* op) final {
    Doc doc;
    doc << "if (" << Print(op->cond) << ") ";
    doc << PrintBody(op->true_branch);
    doc << " else ";
    doc << PrintBody(op->false_branch);
    return doc;
  }

  Doc VisitExpr_(const LetNode* op) final {
    Doc doc;
    doc
      << "let "
      << AllocVar(op->var)
      << " = "
      << Print(op->value, false, true)
      << ";"
      << Doc::NewLine();
    // we use a scope here so GNF hoisting doesn't escape too far
    // and nested, unique lets are not hoisted
    doc << PrintScope(op->body);
    return doc;
  }

  Doc PrintFunc(const Doc& prefix, const relay::Function& fn) {
    Doc doc;
    doc << prefix;
    if (fn->type_params.size() > 0) {
      doc << "[";
      std::vector<Doc> type_params;
      for (const TypeVar& tv : fn->type_params) {
        type_params.push_back(Doc::Text(tv->name_hint));
      }
      doc << Doc::Concat(type_params);
      doc << "]";
    }
    doc << "(";
    std::vector<Doc> params;
    for (Var param : fn->params) {
      params.push_back(AllocVar(param));
    }
    for (const Doc& d : PrintFuncAttrs(fn->attrs)) {
      params.push_back(d);
    }
    doc << Doc::Concat(params) << ") ";
    if (fn->ret_type.defined()) {
      doc << "-> " << Print(fn->ret_type) << " ";
    }
    doc << PrintBody(fn->body);
    return doc;
  }

  Doc PrintFunc(const Doc& prefix, const BaseFunc& base_func) {
    if (auto* n = base_func.as<relay::FunctionNode>()) {
      return PrintFunc(prefix, GetRef<relay::Function>(n));
    } else {
      // def @xyz = meta['ExternalFunc'][id]
      Doc doc;
      doc << prefix << " = " <<  meta_.GetMetaNode(base_func);
      return doc;
    }
  }

  Doc PrintMod(const IRModule& mod) {
    Doc doc;
    int counter = 0;
    // type definitions
    for (const auto& kv : mod->type_definitions) {
      if (counter++ != 0) {
        doc << Doc::NewLine();
      }
      doc << Print(kv.second);
      doc << Doc::NewLine();
    }
    // functions
    for (const auto& kv : mod->functions) {
      dg_ = DependencyGraph::Create(&arena_, kv.second);

      if (counter++ != 0) {
        doc << Doc::NewLine();
      }
      std::ostringstream os;
      os << "def @" << kv.first->name_hint;
      doc << PrintFunc(Doc::Text(os.str()), kv.second);
      doc << Doc::NewLine();
    }
    return doc;
  }

  Doc VisitExpr_(const FunctionNode* op) final {
    return PrintFunc(Doc::Text("fn "), GetRef<Function>(op));
  }

  Doc VisitExpr_(const GlobalVarNode* op) final {
    return Doc::Text('@' + op->name_hint);
  }

  Doc VisitExpr_(const OpNode* op) final {
    return Doc::Text(op->name);
  }

  Doc VisitExpr_(const CallNode* op) final {
    Doc doc;
    // visit args first so they are lifted before the op
    // this places op closer to its call site
    std::vector<Doc> args;
    for (const Expr& arg : op->args) {
      args.push_back(Print(arg));
    }
    for (const Doc& d : PrintCallAttrs(op->attrs, op->op)) {
      args.push_back(d);
    }
    const auto* cons_node = op->op.as<ConstructorNode>();
    if (cons_node) {
      doc << cons_node->name_hint;
    } else {
      doc << Print(op->op);
    }

    if (cons_node && cons_node->inputs.size() == 0) {
      // don't print as a call if it's a 0-arity cons
      return doc;
    } else {
      return doc << "(" << Doc::Concat(args) << ")";
    }
  }

  Doc VisitExpr_(const RefCreateNode* op) final {
    Doc doc;
    return doc << "ref(" << Print(op->value) << ")";
  }

  Doc VisitExpr_(const RefReadNode* op) final {
    Doc doc;
    return doc << Print(op->ref) << "^";
  }

  Doc VisitExpr_(const RefWriteNode* op) final {
    Doc doc;
    return doc << "(" << Print(op->ref) << " := " << Print(op->value) << ")";
  }

  Doc VisitExpr_(const MatchNode* op) final {
    // TODO(jmp): Lots of code duplication here because PrintBody and PrintScope don't accept Docs.
    Doc doc;
    Doc body;
    doc << "match";
    if (!op->complete) {
      doc << "?";
    }
    doc << " (" << Print(op->data) << ") {";
    std::vector<Doc> clause_docs;
    for (const auto& clause : op->clauses) {
      Doc clause_doc;
      clause_doc << PrintPattern(clause->lhs, false) << " => ";
      Doc rhs_doc = PrintScope(clause->rhs);
      if (clause->rhs.as<LetNode>()) {
        // only add braces if there are multiple lines on the rhs
        rhs_doc = Doc::Brace("{", rhs_doc, "}");
      }
      clause_doc << rhs_doc << ",";
      clause_docs.push_back(clause_doc);
    }
    doc << Doc::Indent(2, body << Doc::NewLine() << Doc::Concat(clause_docs, Doc::NewLine()))
        << Doc::NewLine() << "}";
    return doc;
  }

  Doc PrintPattern(const Pattern& pattern, bool meta) {
    auto it = memo_pattern_.find(pattern);
    if (it != memo_pattern_.end()) return it->second;
    Doc printed_pattern;
    if (meta) {
      printed_pattern = meta_.GetMetaNode(GetRef<ObjectRef>(pattern.get()));
    } else {
      printed_pattern = VisitPattern(pattern);
    }
    memo_pattern_[pattern] = printed_pattern;
    return printed_pattern;
  }

  Doc VisitPattern_(const PatternConstructorNode* p) final {
    Doc doc;
    doc << p->constructor->name_hint;
    if (!p->patterns.empty()) {
      doc << "(";
      std::vector<Doc> pats;
      for (const auto& pat : p->patterns) {
        pats.push_back(Print(pat));
      }
      doc << Doc::Concat(pats) << ")";
    }
    return doc;
  }

  Doc VisitPattern_(const PatternTupleNode* pt) final {
    Doc doc;
    doc << "(";
    std::vector<Doc> pats;
    for (const auto& pat : pt->patterns) {
      pats.push_back(Print(pat));
    }
    doc << Doc::Concat(pats) << ")";
    return doc;
  }

  Doc VisitPattern_(const PatternWildcardNode* pw) final {
    return Doc::Text("_");
  }

  Doc VisitPattern_(const PatternVarNode* pv) final {
    return AllocVar(pv->var);
  }

  Doc VisitExpr_(const ConstructorNode* n) final {
    Doc doc;
    doc << n->name_hint;
    if (in_adt_def_ && n->inputs.size() != 0) {
      doc << "(";
      std::vector<Doc> inputs;
      for (Type input : n->inputs) {
        inputs.push_back(Print(input));
      }
      doc << Doc::Concat(inputs) << ")";
    }
    return doc;
  }

  //------------------------------------
  // Overload of Type printing functions
  //------------------------------------
  Doc PrintType(const Type& type, bool meta) {
    auto it = memo_type_.find(type);
    if (it != memo_type_.end()) return it->second;
    Doc printed_type;
    if (meta) {
      printed_type = meta_.GetMetaNode(GetRef<ObjectRef>(type.get()));
    } else {
      printed_type = VisitType(type);
    }
    memo_type_[type] = printed_type;
    return printed_type;
  }

  Doc VisitTypeDefault_(const Object* node) final {
    // by default always print as meta data
    return Print(GetRef<ObjectRef>(node), true);
  }

  Doc VisitType_(const TypeVarNode* node) final {
    return Doc::Text(node->name_hint);
  }

  Doc VisitType_(const GlobalTypeVarNode* node) final {
    return Doc::Text(node->name_hint);
  }

  Doc VisitType_(const TypeCallNode* node) final {
    Doc doc = PrintType(node->func, false);
    std::vector<Doc> args;
    for (const Type& t : node->args) {
      args.push_back(PrintType(t, false));
    }
    doc << "[";
    doc << Doc::Concat(args);
    doc << "]";
    return doc;
  }

  Doc PrintDType(DataType dtype) {
    return Doc::Text(runtime::DLDataType2String(dtype));
  }

  Doc VisitType_(const TensorTypeNode* node) final {
    // scalar type
    if (node->shape.size() == 0) {
      return PrintDType(node->dtype);
    }
    Doc doc;
    doc << "Tensor[(";
    std::vector<Doc> shapes;
    for (ObjectRef shape : node->shape) {
      shapes.push_back(PrintAttr(shape));
    }
    doc << Doc::Concat(shapes);
    return doc << "), " << PrintDType(node->dtype) << "]";
  }

  Doc VisitType_(const TupleTypeNode* node) final {
    std::vector<Doc> fields;
    for (Type field : node->fields) {
      fields.push_back(Print(field));
    }
    Doc doc;
    doc << "(" << Doc::Concat(fields);
    // conform to python tuple format (1,)
    if (node->fields.size() == 1) {
      doc << ",";
    }
    return doc << ")";
  }

  Doc VisitType_(const FuncTypeNode* node) final {
    Doc doc;
    doc << "fn ";
    if (node->type_params.size() != 0) {
      doc << "[";
      std::vector<Doc> type_params;
      for (Type type_param : node->type_params) {
        type_params.push_back(Print(type_param));
      }
      doc << Doc::Concat(type_params);
      doc << "]";
    }
    std::vector<Doc> arg_types;
    for (Type arg_type : node->arg_types) {
      arg_types.push_back(Print(arg_type));
    }
    return doc << "(" << Doc::Concat(arg_types) << ") -> " << Print(node->ret_type);
  }

  Doc VisitType_(const RelayRefTypeNode* node) final {
    Doc doc;
    return doc << "ref(" << Print(node->value) << ")";
  }

  Doc VisitType_(const TypeDataNode* node) final {
    in_adt_def_ = true;
    Doc doc;
    doc << "type " << Print(node->header);

    // type vars
    if (node->type_vars.size() != 0) {
      doc << "[";
      std::vector<Doc> type_vars;
      for (Type type_var : node->type_vars) {
        type_vars.push_back(Print(type_var));
      }
      doc << Doc::Concat(type_vars) << "]";
    }
    doc << " ";

    std::vector<Doc> constructor_docs;
    for (Constructor constructor : node->constructors) {
      constructor_docs.push_back(Print(constructor, /* meta */ false, /* try_inline */ true));
    }
    Doc separator;
    separator << "," << Doc::NewLine();
    Doc adt_body;
    adt_body << Doc::Concat(constructor_docs, separator);
    // add trailing comma if there are any constructors
    if (!constructor_docs.empty()) {
      adt_body << ",";
    }
    doc << Doc::Brace("{", adt_body, "}");
    in_adt_def_ = false;
    return doc;
  }

  //------------------------------------
  // Overload of Attr printing functions
  //------------------------------------

  Doc PrintAttr(const ObjectRef& value, bool meta = false) {
    if (value.defined()) {
      Doc printed_attr;
      if (value.as<tvm::tir::AnyNode>()) {
        printed_attr << "?";
      } else if (meta) {
        printed_attr = meta_.GetMetaNode(Downcast<ObjectRef>(value));
      } else {
        printed_attr = VisitAttr(value);
      }
      return printed_attr;
    } else {
      return Doc::Text("None");
    }
  }

  Doc VisitAttrDefault_(const Object* op) final {
    return PrintAttr(GetRef<ObjectRef>(op), true);
  }

  Doc VisitAttr_(const ArrayNode* op) final {
    Doc doc;
    doc << "[";
    std::vector<Doc> arr_vals;
    for (auto val : op->data) {
      arr_vals.push_back(PrintAttr(val));
    }
    doc << Doc::Concat(arr_vals);
    doc << "]";
    return doc;
  }

  Doc VisitAttr_(const tir::IntImmNode* op) final {
    return ScalarLiteral(op->dtype, op->value);
  }

  Doc VisitAttr_(const tir::FloatImmNode* op) final {
    return ScalarLiteral(op->dtype, op->value);
  }

  Doc VisitAttr_(const tir::StringImmNode* op) final {
    return Doc::StrLiteral(op->value);
  }

 private:
  /*! \brief Whether to print meta data. */
  bool show_meta_data_;
  /*! \brief additional comment function */
  runtime::TypedPackedFunc<std::string(ObjectRef)> annotate_;
  /*! \brief Stack of docs to implement scoped GNFing. */
  std::vector<Doc> doc_stack_{};
  /*! \brief Map from Expr to Doc */
  std::unordered_map<Expr, Doc, ObjectHash, ObjectEqual> memo_;
  /*! \brief Map from Type to Doc */
  std::unordered_map<Type, Doc, ObjectHash, ObjectEqual> memo_type_;
  /*! \brief Map from Type to Doc */
  std::unordered_map<Pattern, Doc, ObjectHash, ObjectEqual> memo_pattern_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief meta data context */
  TextMetaDataContext meta_;
  /*! \brief counter of temporary variable */
  size_t temp_var_counter_{0};
  /*! \brief whether the printer is currently in an ADT definition */
  bool in_adt_def_;
  /*! \brief arena for dependency graph */
  support::Arena arena_;
  /*! \brief dependency graph of the expr */
  DependencyGraph dg_;
  class AttrPrinter;
  friend class AttrPrinter;
};

/*!
 * \brief Attribute printer which prints the attributes in the call.
 */
class RelayTextPrinter::AttrPrinter :
      public AttrVisitor {
 public:
  AttrPrinter(std::vector<Doc>* doc, RelayTextPrinter* parent)
      : docs(doc), parent_(parent) {}

  template<typename T>
  void PrintKV(const char* key, const T& value) {
    Doc doc;
    doc << key << "=" << value;
    docs->push_back(doc);
  }

  void Visit(const char* key, double* value) final {
    Doc doc;
    doc << key << "=" << *value << "f";
    docs->push_back(doc);
  }
  void Visit(const char* key, int64_t* value) final {
    PrintKV(key, *value);
  }
  void Visit(const char* key, uint64_t* value) final {
    PrintKV(key, *value);
  }
  void Visit(const char* key, int* value) final {
    PrintKV(key, *value);
  }
  void Visit(const char* key, bool* value) final {
    PrintKV(key, Doc::PyBoolLiteral(*value));
  }
  void Visit(const char* key, std::string* value) final {
    PrintKV(key, Doc::StrLiteral(*value));
  }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "do not allow void as argument";
  }
  void Visit(const char* key, DataType* value) final {
    PrintKV(key, Doc::StrLiteral(runtime::DLDataType2String(*value)));
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "do not allow NDarray as argument";
  }
  void Visit(const char* key, runtime::ObjectRef* obj) final {
    PrintKV(key, parent_->PrintAttr(*obj));
  }

 private:
  std::vector<Doc>* docs;
  RelayTextPrinter* parent_;
};

std::vector<Doc> RelayTextPrinter::PrintCallAttrs(
    const Attrs& attrs, const Expr& op) {
  std::vector<Doc> docs;
  if (!attrs.defined()) return docs;
  const auto* op_node = op.as<OpNode>();
  if (op_node && (attrs->type_index() != op_node->attrs_type_index)) {
    // fallback
    Doc doc;
    doc << meta_.GetMetaNode(attrs);
    docs.push_back(doc);
    return docs;
  } else {
    AttrPrinter printer(&docs, this);
    const_cast<BaseAttrsNode*>(attrs.operator->())->VisitNonDefaultAttrs(&printer);
    return docs;
  }
}

std::vector<Doc> RelayTextPrinter::PrintFuncAttrs(const Attrs& attrs) {
  std::vector<Doc> docs;
  if (!attrs.defined()) return docs;
  const auto* dict_attrs = attrs.as<DictAttrsNode>();
  CHECK(dict_attrs);
  for (const auto& k : dict_attrs->dict) {
    Doc doc;
    doc << k.first << "=" << Print(k.second);
    docs.push_back(doc);
  }
  return docs;
}
}  // namespace relay

static const char* kSemVer = "v0.0.4";

// TODO(tvm-team): split into files, related: arith/analyzer.h
//
// - text_printer.h (common header)
// - text_printer.cc (prints modules dispatch into relay and tir files)
//    - type_text_printer.cc(specific printing logics for types,
//      can also consider put under type_text_printer)
//    - Implements AsText
// - relay_text_printer.cc (specific printing logics for relay)
// - tir_text_printer.cc (specific printing logics for TIR)
std::string PrettyPrint(const ObjectRef& node) {
  Doc doc;
  doc << relay::RelayTextPrinter(false, nullptr).PrintFinal(node);
  return doc.str();
}

std::string AsText(const ObjectRef& node,
                   bool show_meta_data,
                   runtime::TypedPackedFunc<std::string(ObjectRef)> annotate) {
  Doc doc;
  doc << kSemVer << Doc::NewLine();
  doc << relay::RelayTextPrinter(show_meta_data, annotate).PrintFinal(node);
  return doc.str();
}


TVM_REGISTER_GLOBAL("ir.PrettyPrint")
.set_body_typed(PrettyPrint);

TVM_REGISTER_GLOBAL("ir.AsText")
.set_body_typed(AsText);
}  // namespace tvm
