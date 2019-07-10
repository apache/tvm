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
 *  Copyright (c) 2019 by Contributors
 * \file pretty_printer.cc
 * \brief Pretty printer for Relay programs
 * Supports ANF, GNF, and metadata.
 *
 * Inlining heuristics:
 *  - Always inline:
 *    - GlobalVar
 *    - Constant
 *    - Op
 *    - Var
 *  - Otherwise, inline if the node is at the end of a scope and is used at most once.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/module.h>
#include <tvm/relay/pattern_functor.h>
#include "doc.h"
#include "type_functor.h"
#include "../pass/dependency_graph.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {

/*!
 * \brief Meta data context for PrettyPrinter.
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
    Doc doc;
    doc << "meta[" << node->type_key() << "][" << index << "]";
    meta_repr_[node] = doc;
    return meta_repr_[node];
  }
  /*!
   * \brief Get the metadata section in json format.
   * \return the meta data string.
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
    public PatternFunctor<Doc(const Pattern&)>,
    public TypeFunctor<Doc(const Type&)>,
    public AttrFunctor<Doc(const NodeRef&)> {
 public:
  explicit PrettyPrinter(bool show_meta_data,
                         runtime::TypedPackedFunc<std::string(Expr)> annotate) :
                         show_meta_data_(show_meta_data),
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
  // TODO(jmp): indent should be an instance variable of the printer
  Doc PrintBody(const NodeRef& node, int indent = 2) {
    Doc doc;
    Doc body;
    doc << "{";
    doc << Indent(indent, body << "\n" << PrintScope(node)) << "\n";
    doc << "}";
    return doc;
  }

  // create a new scope by creating a new printer object. This allows temp var
  // numbers to be reused and prevents hoisted vars from escaping too far
  Doc PrintScope(const NodeRef& node) {
    // print in a new scope
    doc_stack_.push_back(Doc());
    // must print first so doc_stack_.back() reference doesn't become stale
    Doc doc = Print(node, false, true);
    doc = doc_stack_.back() << doc;
    doc_stack_.pop_back();
    return doc;
  }

  Doc PrintFinal(const NodeRef& node) {
    if (node.as_derived<ExprNode>()) {
      Expr expr = Downcast<Expr>(node);
      dg_ = DependencyGraph::Create(&arena_, expr);
    }

    Doc doc;
    doc << PrintScope(node);
    if (!meta_.empty()) {
      if (show_meta_data_) {
        std::string meta_json = meta_.GetMetaSection();
        // append meta data in the end.
        doc << "\n" << "/* meta data */" << "\n" << meta_json;
      } else {
        doc << "\n"
            << "// meta data omitted. you can use show_meta_data=True to include meta data";
      }
    }
    return doc;
  }

  std::vector<Doc> PrintCallAttrs(const Attrs& attrs, const Expr& op);
  std::vector<Doc> PrintFuncAttrs(const Attrs& attrs);

  Doc Print(const NodeRef& node, bool meta = false, bool try_inline = false) {
    if (node.as_derived<ExprNode>()) {
      return PrintExpr(Downcast<Expr>(node), meta, try_inline);
    } else if (node.as_derived<TypeNode>()) {
      return PrintType(Downcast<Type>(node), meta);
    } else if (node.as_derived<ModuleNode>()) {
      return PrintMod(Downcast<Module>(node));
    } else {
      Doc doc;
      return doc << node;
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
    return Doc(unique_prefix);
  }

  Doc Print(Kind k) {
    switch (k) {
    case kType:
      return Doc("Type");
    case kShapeVar:
      return Doc("Shape");
    case kBaseType:
      return Doc("BaseType");
    case kConstraint:
      return Doc("Constraint");
    case kAdtHandle:
      return Doc("AdtHandle");
    case kTypeData:
      return Doc("TypeData");
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
    std::string name = var->var->name_hint;
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "t" + name;
    }
    Doc val = GetUniqueName("%" + name);
    if (memo_type_.count(var)) {
      val << "-malformed-ir";
    }
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
    std::string name = var->name_hint();
    // always make sure first name is alpha
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "v" + name;
    }
    Doc val = GetUniqueName("%" + name);
    // still print if ir is malformed, but show the error.
    if (memo_.count(var)) {
      val << "-malformed-ir";
    }
    memo_[var] = val;
    if (var->type_annotation.defined()) {
      val << ": " << Print(var->type_annotation);
    }
    return val;
  }

  bool IsUnique(const Expr& expr) {
    return !(dg_.expr_node.at(expr)->parents.head &&
             dg_.expr_node.at(expr)->parents.head->next);
  }

  bool AlwaysInline(const Expr& expr) {
    return expr.as<GlobalVarNode>() || expr.as<ConstantNode>() ||
           expr.as<OpNode>() || expr.as<VarNode>();
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
      printed_expr = meta_.GetMetaNode(GetRef<NodeRef>(expr.get()));
    } else if (!inline_expr && expr.as<LetNode>()) {
      // wrap GNFed let in brackets
      Doc body;
      printed_expr << "{";
      printed_expr << Indent(2, body << "\n" << VisitExpr(expr)) << "\n";
      printed_expr << "}";
    } else {
      printed_expr = VisitExpr(expr);
    }

    printed_expr << PrintOptionalInfo(expr);

    // add expr to doc
    if (expr.as<VarNode>()) {
      // This is our first time visiting the var and we hit the VarNode case
      // in the visitor. Thus the variable is free.
      doc_stack_.back() << "free_var " << printed_expr << "\n";
      // Memoization is done in AllocVar.
      return memo_[expr];
    } else if (inline_expr) {
      memo_[expr] = printed_expr;
      return printed_expr;
    } else {
      Doc temp_var = AllocTemp();
      memo_[expr] = temp_var;
      doc_stack_.back() << temp_var << " = " << printed_expr << ";" << PrintNewLine();
      return temp_var;
    }
  }

  // Should only be triggered when op is a free variable being visited for the
  // first time.
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
    Doc doc;
    return doc << Print(GetRef<NodeRef>(op), true);
  }

  Doc VisitExpr_(const TupleNode* op) final {
    std::vector<Doc> fields;
    for (Expr field : op->fields) {
      fields.push_back(Print(field));
    }
    Doc doc;
    doc << "(" << PrintVec(fields);
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
      << PrintNewLine();
    // we use a scope here so GNF hoisting doesn't escape too far
    // and nested, unique lets are not hoisted
    doc << PrintScope(op->body);
    return doc;
  }

  Doc PrintFunc(const Doc& prefix, const Function& fn) {
      Doc doc;
      doc << prefix;
      if (fn->type_params.size() > 0) {
        doc << "<";
        std::vector<Doc> type_params;
        for (const TypeVar& tv : fn->type_params) {
          type_params.push_back(AllocTypeVar(tv));
        }
        doc << PrintVec(type_params);
        doc << ">";
      }
      doc << "(";
      std::vector<Doc> params;
      for (Var param : fn->params) {
        params.push_back(AllocVar(param));
      }
      for (const Doc& d : PrintFuncAttrs(fn->attrs)) {
        params.push_back(d);
      }
      doc << PrintVec(params) << ") ";
      if (fn->ret_type.defined()) {
        doc << "-> " << Print(fn->ret_type) << " ";
      }
      doc << PrintBody(fn->body);
      return doc;
  }

  Doc PrintMod(const Module& mod) {
    Doc doc;
    int counter = 0;
    for (const auto& kv : mod->functions) {
      dg_ = DependencyGraph::Create(&arena_, kv.second);

      std::ostringstream os;
      if (counter++ != 0) {
        doc << "\n";
      }
      os << "def @" << kv.first->name_hint;
      doc << PrintFunc(Doc(os.str()), kv.second);
      doc << "\n";
    }
    return doc;
  }

  Doc VisitExpr_(const FunctionNode* op) final {
    return PrintFunc(Doc("fn "), GetRef<Function>(op));
  }

  Doc VisitExpr_(const GlobalVarNode* op) final {
    return Doc('@' + op->name_hint);
  }

  Doc VisitExpr_(const OpNode* op) final {
    return Doc(op->name);
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
    doc << Print(op->op);
    return doc << "(" << PrintVec(args) << ")";
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
    doc << "match " << Print(op->data) << " ";
    doc << "{";
    std::vector<Doc> clauses;
    for (const auto& clause : op->clauses) {
      Doc clause_doc;
      clauses.push_back(clause_doc << Print(clause->lhs) << " -> "
                                   << Print(clause->rhs));
    }
    doc << Indent(2, body << "\n" << PrintVec(clauses, Doc("\n"))) << "\n";
    doc << "}";
    return doc;
  }

  Doc VisitPattern_(const PatternConstructorNode* p) final {
    Doc doc;
    doc << p->constructor->name_hint << "(";
    std::vector<Doc> pats;
    for (const auto& pat : p->patterns) {
      pats.push_back(Print(pat));
    }
    return doc << PrintVec(pats) << ")";
  }

  Doc VisitPattern_(const PatternVarNode* pv) final {
    return AllocVar(pv->var);
  }

  Doc VisitExpr_(const ConstructorNode* n) final {
    return Doc(n->name_hint);
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

  Doc VisitTypeDefault_(const Node* node) final {
    // by default always print as meta data
    return Print(GetRef<NodeRef>(node), true);
  }

  Doc VisitType_(const TypeVarNode* node) final {
    return AllocTypeVar(GetRef<TypeVar>(node));
  }

  Doc VisitType_(const GlobalTypeVarNode* node) final {
    return Doc(node->var->name_hint);
  }

  Doc VisitType_(const TypeCallNode* node) final {
    Doc doc = PrintType(node->func, false);
    std::vector<Doc> args;
    for (const Type& t : node->args) {
      args.push_back(PrintType(t, false));
    }
    doc << "[";
    doc << PrintVec(args);
    doc << "]";
    return doc;
  }

  Doc VisitType_(const TensorTypeNode* node) final {
    // scalar type
    if (node->shape.size() == 0) {
      return PrintDType(node->dtype);
    }
    Doc doc;
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
    Doc doc;
    doc << "(" << PrintVec(fields);
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
      doc << "<";
      std::vector<Doc> type_params;
      for (Type type_param : node->type_params) {
        type_params.push_back(Print(type_param));
      }
      doc << PrintVec(type_params);
      doc << ">";
    }
    std::vector<Doc> arg_types;
    for (Type arg_type : node->arg_types) {
      arg_types.push_back(Print(arg_type));
    }
    return doc << "(" << PrintVec(arg_types) << ") -> " << Print(node->ret_type);
  }

  Doc VisitType_(const RefTypeNode* node) final {
    Doc doc;
    return doc << "ref(" << Print(node->value) << ")";
  }

  //------------------------------------
  // Overload of Attr printing functions
  //------------------------------------

  Doc PrintAttr(const NodeRef& value, bool meta = false) {
    if (value.defined()) {
      Doc printed_attr;
      if (value.as<tvm::ir::Any>()) {
        printed_attr << "?";
      } else if (meta) {
        printed_attr = meta_.GetMetaNode(value);
      } else {
        printed_attr = VisitAttr(value);
      }
      return printed_attr;
    } else {
      return Doc("None");
    }
  }

  Doc VisitAttrDefault_(const Node* op) final {
    return PrintAttr(GetRef<NodeRef>(op), true);
  }

  Doc VisitAttr_(const ArrayNode* op) final {
    Doc doc;
    doc << "[";
    std::vector<Doc> arr_vals;
    for (NodePtr<Node> val : op->data) {
      arr_vals.push_back(PrintAttr(NodeRef(val)));
    }
    doc << PrintVec(arr_vals);
    doc << "]";
    return doc;
  }

  Doc VisitAttr_(const ir::IntImm* op) final {
    return PrintConstScalar(op->type, &(op->value));
  }

  Doc VisitAttr_(const ir::UIntImm* op) final {
    return PrintConstScalar(op->type, &(op->value));
  }

  Doc VisitAttr_(const ir::FloatImm* op) final {
    return PrintConstScalar(op->type, &(op->value));
  }

  Doc VisitAttr_(const ir::StringImm* op) final {
    return PrintString(op->value);
  }

 private:
  /*! \brief Whether to print meta data. */
  bool show_meta_data_;
  /*! \brief additional comment function */
  runtime::TypedPackedFunc<std::string(Expr)> annotate_;
  /*! \brief Stack of docs to implement scoped GNFing. */
  std::vector<Doc> doc_stack_{};
  /*! \brief Map from Expr to Doc */
  std::unordered_map<Expr, Doc, NodeHash, NodeEqual> memo_;
  /*! \brief Map from Type to Doc */
  std::unordered_map<Type, Doc, NodeHash, NodeEqual> memo_type_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief meta data context */
  TextMetaDataContext meta_;
  /*! \brief counter of temporary variable */
  size_t temp_var_counter_{0};
  /*! \brief arena for dependency graph */
  common::Arena arena_;
  /*! \brief dependency graph of the expr */
  DependencyGraph dg_;
  class AttrPrinter;
  friend class AttrPrinter;
};

/*!
 * \brief Attribute printer which prints the attributes in the call.
 */
class PrettyPrinter::AttrPrinter : public AttrVisitor {
 public:
  AttrPrinter(std::vector<Doc>* doc, PrettyPrinter* parent) : docs(doc), parent_(parent) {}

  template<typename T>
  void PrintKV(const char* key, const T& value) {
    Doc doc;
    doc << key << "=" << value;
    docs->push_back(doc);
  }

  void Visit(const char* key, double* value) final {
    PrintKV(key, *value);
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
    PrintKV(key, PrintBool(*value));
  }
  void Visit(const char* key, std::string* value) final {
    PrintKV(key, PrintString(*value));
  }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "do not allow void as argument";
  }
  void Visit(const char* key, DataType* value) final {
    PrintKV(key, PrintString(runtime::TVMType2String(Type2TVMType(*value))));
  }
  void Visit(const char* key, NodeRef* value) final {
    PrintKV(key, parent_->PrintAttr(*value));
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "do not allow NDarray as argument";
  }
  void Visit(const char* key, runtime::Object* obj) final {
    LOG(FATAL) << "do not allow Object as argument";
  }

 private:
  std::vector<Doc>* docs;
  PrettyPrinter* parent_;
};

std::vector<Doc> PrettyPrinter::PrintCallAttrs(const Attrs& attrs, const Expr& op) {
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

std::vector<Doc> PrettyPrinter::PrintFuncAttrs(const Attrs& attrs) {
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

std::string PrettyPrint_(const NodeRef& node,
                         bool show_meta_data,
                         runtime::TypedPackedFunc<std::string(Expr)> annotate) {
  Doc doc;
  doc << "v0.0.3" << "\n"
      << PrettyPrinter(show_meta_data, annotate).PrintFinal(node);
  return doc.str();
}

std::string PrettyPrint(const NodeRef& node) {
  Doc doc;
  doc << PrettyPrinter(false, runtime::TypedPackedFunc<std::string(Expr)>()).PrintFinal(node);
  return doc.str();
}

std::string AsText(const NodeRef& node,
                       bool show_meta_data,
                       runtime::TypedPackedFunc<std::string(Expr)> annotate) {
  return PrettyPrint_(node, show_meta_data, annotate);
}

TVM_REGISTER_API("relay._expr.AsText")
.set_body_typed<std::string(const NodeRef&,
                            bool,
                            runtime::TypedPackedFunc<std::string(Expr)>)>(AsText);

}  // namespace relay
}  // namespace tvm
