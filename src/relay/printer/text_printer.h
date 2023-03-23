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
 * \file text_printer.h
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */

#ifndef TVM_RELAY_PRINTER_TEXT_PRINTER_H_
#define TVM_RELAY_PRINTER_TEXT_PRINTER_H_

#include <tvm/ir/module.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/var.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../ir/attr_functor.h"
#include "../analysis/dependency_graph.h"
#include "doc.h"
#include "meta_data.h"

namespace tvm {
namespace relay {

class TextPrinter;

class RelayTextPrinter : public ExprFunctor<Doc(const Expr&)>,
                         public PatternFunctor<Doc(const Pattern&)>,
                         public TypeFunctor<Doc(const Type&)>,
                         public AttrFunctor<Doc(const ObjectRef&)> {
 public:
  explicit RelayTextPrinter(bool show_meta_data, TextMetaDataContext* meta,
                            runtime::TypedPackedFunc<std::string(ObjectRef)> annotate)
      : show_meta_data_(show_meta_data), annotate_(annotate), meta_(meta) {}
  Doc VisitExpr(const Expr& expr) override;
  virtual Doc VisitLeaf(const Expr& expr);
  virtual bool CheckVisited(const Expr& expr);

  /*!
   * \brief Print additional info about expr in comment.
   * \param expr The expression.
   */
  Doc PrintOptionalInfo(const Expr& expr);
  // indent a new body
  Doc PrintBody(const ObjectRef& node, int indent = 2);
  // create a new scope by creating a new printer object. This allows temp var
  // numbers to be reused and prevents hoisted vars from escaping too far
  Doc PrintScope(const ObjectRef& node);
  Doc PrintFinal(const ObjectRef& node);

  /*!
   * \brief Returns \p attrs printed using the generic attribute visitor, as a sequence
   * of key=value entries, if any.
   */
  void AppendGenericAttrs(std::vector<Doc>* docs, const Attrs& attrs, bool include_type_key);

  /*!
   * \brief Returns \p attrs printed as a sequence of key=value entries, if any.
   * This is used for call attributes.
   */
  std::vector<Doc> PrintCallAttrs(const Attrs& attrs, const Expr& op);

  /*!
   * \brief Returns \p dict_attrs printed as a sequence of key=value entries, if any.
   * This is used for function definition attributes.
   */
  std::vector<Doc> PrintDictAttrs(const DictAttrs& dict_attrs);
  std::vector<Doc> PrintDictAttrs(const Map<String, ObjectRef>& dict_attrs);

  /*!
   * \brief Returns \p value printed as the rhs of an attribute key=value entry. If \p force_meta
   * is true then value is printed in meta[...] for irrespective of the show_meta_data_ flag.
   */
  Doc PrintAttributeValue(const ObjectRef& value, bool force_meta = false);

  /*!
   * \brief Returns \p attrs printed as a self-contained value, ie wrapped in braces.
   */
  Doc PrintAttrsAsAttributeValue(const Attrs& attrs);

  /*!
   * \brief Returns \p map printed as a self-contained value, ie wrapped in braces.
   */
  Doc PrintMapAsAttributeValue(const Map<ObjectRef, ObjectRef>& map);

  Doc PrintSpan(const Span& span);

  Doc Print(const ObjectRef& node, bool meta = false, bool try_inline = false);

  Doc TempVar(int n);
  Doc AllocTemp();
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  Doc GetUniqueName(const std::string& prefix);
  Doc Print(Kind k);
  /*!
   * \brief Allocate name to a type variable.
   * \param var The input type variable.
   * \return The corresponding name.
   */
  Doc AllocTypeVar(const TypeVar& var);
  /*!
   * \brief Allocate name to a variable.
   * \param var The input variable.
   * \return The corresponding name.
   */
  Doc AllocVar(const Var& var);
  bool IsUnique(const Expr& expr);
  bool AlwaysInline(const Expr& expr);

  Doc PrintFunc(const Doc& prefix, const relay::Function& fn);
  Doc PrintFunc(const Doc& prefix, const BaseFunc& base_func);
  Doc PrintMod(const IRModule& mod);

  //------------------------------------
  // Overload of Expr printing functions
  //------------------------------------
  Doc PrintExpr(const Expr& expr, bool meta, bool try_inline, bool optional_info = true);
  // Should only be triggered when op is a free variable being visited for the
  // first time.
  Doc VisitExpr_(const VarNode* op) final;
  Doc VisitExpr_(const ConstantNode* op) final;
  Doc VisitExpr_(const TupleNode* op) final;
  Doc VisitExpr_(const TupleGetItemNode* op) final;
  Doc VisitExpr_(const IfNode* op) final;
  Doc VisitExpr_(const LetNode* op) final;
  Doc VisitExpr_(const FunctionNode* op) final;
  Doc VisitExpr_(const GlobalVarNode* op) final;
  Doc VisitExpr_(const OpNode* op) final;
  Doc VisitExpr_(const CallNode* op) final;
  Doc VisitExpr_(const RefCreateNode* op) final;
  Doc VisitExpr_(const RefReadNode* op) final;
  Doc VisitExpr_(const RefWriteNode* op) final;
  Doc VisitExpr_(const MatchNode* op) final;
  Doc PrintPattern(const Pattern& pattern, bool meta);
  Doc VisitPattern_(const PatternConstructorNode* p) final;
  Doc VisitPattern_(const PatternTupleNode* pt) final;
  Doc VisitPattern_(const PatternWildcardNode* pw) final;
  Doc VisitPattern_(const PatternVarNode* pv) final;
  Doc VisitExpr_(const ConstructorNode* n) final;
  //------------------------------------
  // Overload of Type printing functions
  //------------------------------------
  Doc PrintType(const Type& type, bool meta);
  Doc VisitTypeDefault_(const Object* node) final;
  Doc VisitType_(const TypeVarNode* node) final;
  Doc VisitType_(const GlobalTypeVarNode* node) final;
  Doc VisitType_(const TypeCallNode* node) final;
  Doc PrintDType(DataType dtype);
  Doc VisitType_(const TensorTypeNode* node) final;
  Doc VisitType_(const TupleTypeNode* node) final;
  Doc VisitType_(const FuncTypeNode* node) final;
  Doc VisitType_(const RelayRefTypeNode* node) final;
  Doc VisitType_(const TypeDataNode* node) final;
  //------------------------------------
  // Overload of Attr printing functions
  //------------------------------------
  Doc VisitAttrDefault_(const Object* op) final;
  Doc VisitAttr_(const ArrayNode* op) final;
  Doc VisitAttr_(const tir::IntImmNode* op) final;
  Doc VisitAttr_(const tir::FloatImmNode* op) final;
  Doc VisitAttr_(const tir::StringImmNode* op) final;

 private:
  /*! \brief Whether to print meta data. */
  bool show_meta_data_;
  /*! \brief additional comment function */
  runtime::TypedPackedFunc<std::string(ObjectRef)> annotate_;
  /*! \brief Stack of docs to implement scoped GNFing. */
  std::vector<Doc> doc_stack_{};
  /*! \brief Set for introduced vars */
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> var_memo_;
  /*! \brief Set for exprs have been printed optional information */
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> opt_info_memo_;
  /*! \brief Map for result and memo_ diffs for visited expression */
  std::unordered_map<Expr, Doc, ObjectPtrHash, ObjectPtrEqual> result_memo_;
  /*! \brief Map from Expr to Doc */
  std::unordered_map<Expr, Doc, ObjectPtrHash, ObjectPtrEqual> memo_;
  /*! \brief Map from Type to Doc */
  std::unordered_map<Type, Doc, ObjectPtrHash, ObjectPtrEqual> memo_type_;
  /*! \brief Map from Type to Doc */
  std::unordered_map<Pattern, Doc, ObjectPtrHash, ObjectPtrEqual> memo_pattern_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief meta data context */
  TextMetaDataContext* meta_;
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
  friend class tvm::relay::TextPrinter;
};

using namespace ::tvm::tir;

/*!
 *  \brief Meta node collector
 *  If we decide to put some node into meta, then all the sub-nodes inside
 *  it need to be put in meta as well, since when parsing we need to know
 *  whether two refs are the same
 */
class MetaCollector : public StmtExprVisitor {
 public:
  explicit MetaCollector(TextMetaDataContext* meta) : meta_(meta) {}

  void Collect(const ObjectRef& n) {
    // these nodes can be print directly(StringLiteral or use identifier to identify)
    if (!n.defined() || n.as<StringImmNode>() || n.as<StringObj>() || n.as<SizeVarNode>() ||
        n.as<VarNode>() || n.as<BufferNode>() || n.as<IterVarNode>()) {
      return;
    }
    if (n->IsInstance<StmtNode>()) {
      VisitStmt(Downcast<Stmt>(n));
    } else if (n->IsInstance<PrimExprNode>()) {
      VisitExpr(Downcast<PrimExpr>(n));
    }
  }

  void VisitStmt(const Stmt& n) override {
    meta_->GetMetaNode(n);
    StmtVisitor::VisitStmt(n);
  }

  void VisitExpr(const PrimExpr& n) override {
    meta_->GetMetaNode(n);
    ExprVisitor::VisitExpr(n);
  }

 private:
  TextMetaDataContext* meta_;
};

class TIRTextPrinter : public StmtFunctor<Doc(const Stmt&)>,
                       public tir::ExprFunctor<Doc(const PrimExpr&)>,
                       public TypeFunctor<Doc(const Type&)> {
 public:
  explicit TIRTextPrinter(bool show_meta, TextMetaDataContext* meta)
      : show_meta_(show_meta), meta_(meta), meta_collector_(meta) {}

  /*! \brief Output a newline */
  virtual Doc NewLine();

  /*! \brief Print the node */
  Doc Print(const ObjectRef& node);

  /*! \brief Place into `s` the name used in the preceding Print call for `v`.
   * \param v Var instance to check. Must point to a VarNode visited by Print.
   * \param s String to receive the name.
   * \return true when a name re-mapping was found.
   */
  bool GetVarName(::tvm::tir::Var v, std::string* s);

 protected:
  Doc VisitExpr_(const IntImmNode* op) override;
  Doc VisitExpr_(const FloatImmNode* op) override;
  Doc VisitExpr_(const StringImmNode* op) override;
  Doc VisitExpr_(const CastNode* op) override;
  Doc VisitExpr_(const tir::VarNode* op) override;
  Doc VisitExpr_(const AddNode* op) override;
  Doc VisitExpr_(const SubNode* op) override;
  Doc VisitExpr_(const MulNode* op) override;
  Doc VisitExpr_(const DivNode* op) override;
  Doc VisitExpr_(const ModNode* op) override;
  Doc VisitExpr_(const FloorDivNode* op) override;
  Doc VisitExpr_(const FloorModNode* op) override;
  Doc VisitExpr_(const MinNode* op) override;
  Doc VisitExpr_(const MaxNode* op) override;
  Doc VisitExpr_(const EQNode* op) override;
  Doc VisitExpr_(const NENode* op) override;
  Doc VisitExpr_(const LTNode* op) override;
  Doc VisitExpr_(const LENode* op) override;
  Doc VisitExpr_(const GTNode* op) override;
  Doc VisitExpr_(const GENode* op) override;
  Doc VisitExpr_(const AndNode* op) override;
  Doc VisitExpr_(const OrNode* op) override;
  Doc VisitExpr_(const NotNode* op) override;
  Doc VisitExpr_(const SelectNode* op) override;
  Doc VisitExpr_(const BufferLoadNode* op) override;
  Doc VisitExpr_(const ProducerLoadNode* op) override;
  Doc VisitExpr_(const RampNode* op) override;
  Doc VisitExpr_(const BroadcastNode* op) override;
  Doc VisitExpr_(const tir::LetNode* op) override;
  Doc VisitExpr_(const tir::CallNode* op) override;
  Doc VisitExpr_(const ShuffleNode* op) override;
  Doc VisitExpr_(const ReduceNode* op) override;
  Doc VisitExprDefault_(const Object* op) override;

  Doc VisitStmt_(const LetStmtNode* op) override;
  Doc VisitStmt_(const AttrStmtNode* op) override;
  Doc VisitStmt_(const AssertStmtNode* op) override;
  Doc VisitStmt_(const BufferStoreNode* op) override;
  Doc VisitStmt_(const ProducerStoreNode* op) override;
  Doc VisitStmt_(const BufferRealizeNode* op) override;
  Doc VisitStmt_(const ProducerRealizeNode* op) override;
  Doc VisitStmt_(const AllocateNode* op) override;
  Doc VisitStmt_(const AllocateConstNode* op) override;
  Doc VisitStmt_(const DeclBufferNode* op) override;
  Doc VisitStmt_(const IfThenElseNode* op) override;
  Doc VisitStmt_(const SeqStmtNode* op) override;
  Doc VisitStmt_(const EvaluateNode* op) override;
  Doc VisitStmt_(const ForNode* op) override;
  Doc VisitStmt_(const WhileNode* op) override;
  Doc VisitStmt_(const PrefetchNode* op) override;
  Doc VisitStmt_(const BlockRealizeNode* op) override;
  Doc VisitStmtDefault_(const Object* op) override;

 private:
  /*! \brief whether show meta data */
  bool show_meta_;
  /*! \brief meta data context */
  TextMetaDataContext* meta_;
  /*! \brief meta collector */
  MetaCollector meta_collector_;
  /*! \brief Map from Var to Doc */
  std::unordered_map<tir::Var, Doc, ObjectPtrHash, ObjectPtrEqual> memo_var_;
  /*! \brief Map from Buffer to Doc */
  std::unordered_map<Buffer, Doc, ObjectPtrHash, ObjectPtrEqual> memo_buf_;
  /*! \brief Map from Buffer to Doc */
  std::unordered_map<DataProducer, Doc, ObjectPtrHash, ObjectPtrEqual> memo_producer_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;

  friend class TextPrinter;

  Doc VisitType_(const PrimTypeNode* node) override;
  Doc VisitType_(const PointerTypeNode* node) override;
  Doc VisitType_(const TupleTypeNode* node) override;

  Doc PrintIRModule(const IRModule& module);
  Doc PrintPrimFunc(const PrimFunc& primFunc);
  Doc PrintArray(const ArrayNode* op);
  Doc PrintIterVar(const IterVarNode* op);
  Doc PrintRange(const RangeNode* op);
  Doc PrintBuffer(const BufferNode* op);
  Doc PrintProducer(const DataProducerNode* op);
  Doc BufferNode2Doc(const BufferNode* op, Doc doc);
  Doc DataProducerNode2Doc(const DataProducerNode* op, Doc doc);
  Doc PrintString(const StringObj* op) { return Doc::StrLiteral(op->data); }
  Doc PrintBufferRegion(const BufferRegionNode* op);

  /*!
   * \brief special method to print out data type
   * \param dtype The data type
   */
  static Doc PrintDType(DataType dtype);
  /*!
   * \brief special method to print out const scalar
   * \param dtype The data type
   * \param data The pointer to hold the data.
   */
  template <typename T>
  static Doc PrintConstScalar(DataType dtype, const T& data);
  Doc GetUniqueName(std::string prefix);
  Doc AllocVar(const tir::Var& var);
  Doc AllocConst(const AllocateConst& var);
  Doc AllocBuf(const Buffer& buffer);
  Doc AllocProducer(const DataProducer& buffer);
  /*!
   * \brief special method to render vectors of docs with a separator
   * \param vec vector of docs
   * \param sep separator
   */
  static Doc PrintSep(const std::vector<Doc>& vec, const Doc& sep);
  Doc PrintBody(const Stmt& body, bool indent = true);
};

String AsTVMScriptWithDiagnostic(const ObjectRef& mod, const String& tir_prefix, bool show_meta,
                                 runtime::TypedPackedFunc<std::string(Stmt)> annotate);

class TextPrinter {
 public:
  explicit TextPrinter(bool show_meta_data,
                       const runtime::TypedPackedFunc<std::string(ObjectRef)>& annotate,
                       bool show_warning = true)
      : show_meta_data_(show_meta_data),
        show_warning_(show_warning),
        annotate_(annotate),
        relay_text_printer_(show_meta_data, &meta_, annotate),
        tir_text_printer_(show_meta_data, &meta_) {}

  /*! \brief whether show meta data */
  bool show_meta_data_;

  /*! \brief whether show the meta data warning message */
  bool show_warning_;

  /*! \brief meta data context */
  TextMetaDataContext meta_;
  /*! \brief additional comment function */
  runtime::TypedPackedFunc<std::string(ObjectRef)> annotate_;
  /*! \brief Relay Text Printer */
  relay::RelayTextPrinter relay_text_printer_;
  /*! \brief TIR Text Printer */
  TIRTextPrinter tir_text_printer_;

  bool GetVarName(::tvm::tir::Var v, std::string* s) { return tir_text_printer_.GetVarName(v, s); }

  Doc PrintFinal(const ObjectRef& node) {
    Doc doc;
    if (node.defined() && node->IsInstance<IRModuleNode>()) {
      doc << PrintMod(Downcast<IRModule>(node));
    } else if (node.defined() &&
               (node->IsInstance<tir::PrimFuncNode>() || node->IsInstance<PrimExprNode>() ||
                node->IsInstance<tir::StmtNode>())) {
      doc << tir_text_printer_.Print(node);
    } else {
      doc << relay_text_printer_.PrintFinal(node);
    }
    if (!meta_.empty()) {
      doc << Doc::NewLine();
      if (show_meta_data_) {
        doc << "#[metadata]" << Doc::NewLine() << meta_.GetMetaSection();
      } else if (show_warning_) {
        doc << "/* For debugging purposes the metadata section has been omitted." << Doc::NewLine()
            << " * If you would like to see the full metadata section you can set the "
            << Doc::NewLine() << " * option to `True` when invoking `astext`. " << Doc::NewLine()
            << " */";
      }
    }
    return doc;
  }

  Doc PrintMod(const IRModule& mod);
};
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PRINTER_TEXT_PRINTER_H_
