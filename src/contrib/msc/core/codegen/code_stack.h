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
 * \file src/contrib/msc/core/codegen/code_stack.h
 * \brief CodeStack for doc printer.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_CODE_STACK_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_CODE_STACK_H_

#include <tvm/script/printer/doc.h>

#include <stack>
#include <string>
#include <vector>

#include "../printer/msc_doc.h"
#include "../printer/print_utils.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief Inner class for doc stack
 */
class BaseStack {
 public:
  /*!
   * \brief The constructor of CodeStack
   */
  BaseStack() { Reset(); }

  /*! \brief Cleanup blocks*/
  void Reset() {
    while (!blocks_.empty()) {
      blocks_.pop();
    }
    BlockStart();
  }

  /*! \brief Get the docs*/
  const Array<Doc> GetDocs() const;

 protected:
  /*! \brief Push Id Doc*/
  void Line(const Doc& doc);
  void Line(const String& line = "");

  /*! \brief Push Comment Doc*/
  void Comment(const String& comment, bool attach = false);

  /*! \brief Push Assign Doc*/
  template <typename LT, typename RT>
  inline void Assign(const LT& lhs, const RT& rhs, const String& annotation = "") {
    PushDoc(DocUtils::ToAssign(lhs, rhs, annotation));
  }

  /*! \brief Push declare Doc*/
  void Declare(const String& type, const String& variable, size_t len = 0,
               bool use_constructor = true);

  /*! \brief Cache declare argument*/
  void DeclareArgBase(const ExprDoc& value);

  /*! \brief Cache declare typed argument*/
  template <typename T>
  inline void DeclareArg(const T& value) {
    DeclareArgBase(DocUtils::ToDoc(value));
  }

  /*! \brief Cache class Doc*/
  void ClassDef(const String& class_name);

  /*! \brief Cache class decorator*/
  void ClassDecorator(const String& decorator);

  /*! \brief Start class body block*/
  void ClassStart();

  /*! \brief End class body block*/
  void ClassEnd();

  /*! \brief Start struct body block*/
  void StructStart(const String& struct_name);

  /*! \brief End struct body block*/
  void StructEnd();

  /*! \brief Cache function Doc*/
  void FuncDef(const String& func_name, const String& ret_type = "");

  /*! \brief Cache function argument*/
  void FuncArg(const String& arg, const String& annotation = "", const String& value = "");

  /*! \brief Cache function decorator*/
  void FuncDecorator(const String& decorator);

  /*! \brief Start function body block*/
  void FuncStart();

  /*! \brief End function body block*/
  void FuncEnd();

  template <typename T>
  void FuncEnd(const T& ret_val) {
    PushDoc(ReturnDoc(DocUtils::ToDoc(ret_val)));
    FuncEnd();
  }

  /*! \brief Cache constructor Doc*/
  void ConstructorDef(const String& constructor_name);

  /*! \brief Cache constructor argument*/
  void ConstructorArg(const String& arg, const String& annotation = "", const String& value = "");

  /*! \brief Start constructor body block*/
  void ConstructorStart();

  /*! \brief End constructor body block*/
  void ConstructorEnd();

  /*! \brief Cache lambda Doc*/
  void LambdaDef(const String& lambda_name);

  /*! \brief Cache lambda argument*/
  void LambdaArg(const String& arg, const String& annotation = "", const String& value = "");

  /*! \brief Cache lambda reference*/
  void LambdaRef(const String& ref);

  /*! \brief Start lambda body block*/
  void LambdaStart();

  /*! \brief End lambda body block*/
  void LambdaEnd(const String& ret_val = "");
  void LambdaEnd(const ExprDoc& ret_val);

  /*! \brief Push call and maybe assign Doc*/
  void FuncCall(const String& callee, Optional<ExprDoc> assign_to,
                Optional<ExprDoc> caller = NullOpt);
  void FuncCall(const String& callee, const String& assign_to = "", const String& caller = "");

  /*! \brief Push method call Doc*/
  void MethodCall(const String& callee, bool new_line = false);

  /*! \brief Push inplace call and maybe assign Doc*/
  void InplaceStart(const String& callee, Optional<ExprDoc> assign_to,
                    Optional<ExprDoc> caller = NullOpt);
  void InplaceStart(const String& callee, const String& assign_to = "", const String& caller = "");

  /*! \brief End inplace call*/
  void InplaceEnd();

  /*! \brief Push nested expr to last Doc*/
  void PopNest(const String& key = "");

  /*! \brief Cache call typed argument*/
  void CallArgBase(const ExprDoc& value, const String& key = "");

  /*! \brief Cache call normal argument*/
  template <typename T>
  inline void CallArg(T value, const String& key = "") {
    const auto& doc_value = DocUtils::ToDoc(value);
    if (doc_value.defined()) {
      CallArgBase(doc_value, key);
    }
  }
  inline void CallArg(const Array<ExprDoc>& values) {
    for (const auto& v : values) {
      if (v.defined()) {
        CallArgBase(v);
      }
    }
  }

  /*! \brief Push if to cache and start if block*/
  void ConditionIf(const String& predicate);

  /*! \brief Push then branch to cached and start block*/
  void ConditionElse();

  /*! \brief Push else branch to cached*/
  void ConditionEnd();

  /*! \brief Push for to cache and start for block*/
  template <typename LT, typename RT>
  void ForStart(const LT& lhs, const RT& rhs) {
    PushDoc(ForDoc(DocUtils::ToDoc(lhs), DocUtils::ToDoc(rhs), Array<StmtDoc>()));
    BlockStart();
  }

  /*! \brief Push for range to cache and start for block*/
  template <typename ST, typename ET>
  void ForStart(const String& lhs, const ST& start, const ET& end) {
    Array<ExprDoc> range{DocUtils::ToDoc(start), DocUtils::ToDoc(end)};
    PushDoc(ForDoc(IdDoc(lhs), TupleDoc(range), Array<StmtDoc>()));
    BlockStart();
  }

  /*! \brief End a for block*/
  void ForEnd();

  /*! \brief Push while to cache and start while block*/
  void WhileStart(const String& predicate);

  /*! \brief End a while block*/
  void WhileEnd();

  /*! \brief Push switch to cache and start switch block*/
  void SwitchStart(const String& predicate);

  /*! \brief Add new case to switch*/
  void SwitchCase(const String& predicate = "");

  /*! \brief Push switch to cached*/
  void SwitchEnd();

  /*! \brief Start a new block*/
  void BlockStart();

  /*! \brief End a block*/
  void BlockEnd(bool block_docs = true);

  /*! \brief Start a new scope*/
  void ScopeStart(const String& scope_def = "", const String& scope_ref = "");

  /*! \brief End a scope*/
  void ScopeEnd();

 private:
  /*! \brief Check if has block left*/
  bool HasBlock() const;

  /*! \brief Get the last the block*/
  const Array<Doc> TopBlock() const;

  /*! \brief Pop last the block*/
  const Array<Doc> PopBlock();

  /*! \brief Check if doc left*/
  bool HasDoc();

  /*! \brief Get the last doc*/
  const Doc TopDoc();

  /*! \brief Pop last doc*/
  const Doc PopDoc();

  /*! \brief Pop last doc with type checked*/
  template <typename TDoc, typename TDocNode>
  const TDoc PopCheckedDoc();

  /*! \brief Push doc*/
  void PushDoc(const Doc& doc);

  /*! \brief The blocks, each has docs array*/
  std::stack<Array<Doc>> blocks_;
};

#define COMMON_WRAPPERS(Stack)                                                                    \
  Stack& line(const Doc& doc) {                                                                   \
    Line(doc);                                                                                    \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& line(const String& line = "") {                                                          \
    Line(line);                                                                                   \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& comment(const String& comment, bool attach = false) {                                    \
    Comment(comment, attach);                                                                     \
    return *this;                                                                                 \
  }                                                                                               \
  template <typename LT, typename RT>                                                             \
  Stack& assign(const LT& lhs, const RT& rhs, const String& annotation = "") {                    \
    Assign(lhs, rhs, annotation);                                                                 \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& declare(const String& type, const String& variable, size_t len = 0,                      \
                 bool use_constructor = true) {                                                   \
    Declare(type, variable, len, use_constructor);                                                \
    return *this;                                                                                 \
  }                                                                                               \
  template <typename T>                                                                           \
  Stack& declare_arg(const T& value) {                                                            \
    DeclareArg(value);                                                                            \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& class_def(const String& class_name) {                                                    \
    ClassDef(class_name);                                                                         \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& class_decorator(const String& decorator) {                                               \
    ClassDecorator(decorator);                                                                    \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& class_start() {                                                                          \
    ClassStart();                                                                                 \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& class_end() {                                                                            \
    ClassEnd();                                                                                   \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& struct_start(const String& struct_name) {                                                \
    StructStart(struct_name);                                                                     \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& struct_end() {                                                                           \
    StructEnd();                                                                                  \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& func_def(const String& func_name, const String& ret_type = "") {                         \
    FuncDef(func_name, ret_type);                                                                 \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& func_arg(const String& arg, const String& annotation = "", const String& value = "") {   \
    FuncArg(arg, annotation, value);                                                              \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& func_decorator(const String& decorator) {                                                \
    FuncDecorator(decorator);                                                                     \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& func_start() {                                                                           \
    FuncStart();                                                                                  \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& func_end() {                                                                             \
    FuncEnd();                                                                                    \
    return *this;                                                                                 \
  }                                                                                               \
  template <typename T>                                                                           \
  Stack& func_end(const T& ret_val) {                                                             \
    FuncEnd(ret_val);                                                                             \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& func_call(const String& callee, Optional<ExprDoc> assign_to,                             \
                   Optional<ExprDoc> caller = NullOpt) {                                          \
    FuncCall(callee, assign_to, caller);                                                          \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& func_call(const String& callee, const String& assign_to = "",                            \
                   const String& caller = "") {                                                   \
    FuncCall(callee, assign_to, caller);                                                          \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& method_call(const String& callee, bool new_line = false) {                               \
    MethodCall(callee, new_line);                                                                 \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& inplace_start(const String& callee, Optional<ExprDoc> assign_to,                         \
                       Optional<ExprDoc> caller = NullOpt) {                                      \
    InplaceStart(callee, assign_to, caller);                                                      \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& inplace_start(const String& callee, const String& assign_to = "",                        \
                       const String& caller = "") {                                               \
    InplaceStart(callee, assign_to, caller);                                                      \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& inplace_end() {                                                                          \
    InplaceEnd();                                                                                 \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& constructor_def(const String& func_name) {                                               \
    ConstructorDef(func_name);                                                                    \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& constructor_arg(const String& arg, const String& annotation = "",                        \
                         const String& value = "") {                                              \
    ConstructorArg(arg, annotation, value);                                                       \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& constructor_start() {                                                                    \
    ConstructorStart();                                                                           \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& constructor_end() {                                                                      \
    ConstructorEnd();                                                                             \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& lambda_def(const String& lambda_name) {                                                  \
    LambdaDef(lambda_name);                                                                       \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& lambda_arg(const String& arg, const String& annotation = "", const String& value = "") { \
    LambdaArg(arg, annotation, value);                                                            \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& lambda_ref(const String& ref) {                                                          \
    LambdaRef(ref);                                                                               \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& lambda_start() {                                                                         \
    LambdaStart();                                                                                \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& lambda_end(const String& ret_val = "") {                                                 \
    LambdaEnd(ret_val);                                                                           \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& lambda_end(const ExprDoc& ret_val) {                                                     \
    LambdaEnd(ret_val);                                                                           \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& pop_nest(const String& key = "") {                                                       \
    PopNest(key);                                                                                 \
    return *this;                                                                                 \
  }                                                                                               \
  template <typename T>                                                                           \
  Stack& call_arg(T value, const String& key = "") {                                              \
    CallArg(value, key);                                                                          \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& call_arg(const ExprDoc& value, const String& key = "") {                                 \
    CallArg(value, key);                                                                          \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& call_arg(const Array<ExprDoc>& values) {                                                 \
    CallArg(values);                                                                              \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& cond_if(const String& predicate) {                                                       \
    ConditionIf(predicate);                                                                       \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& cond_else() {                                                                            \
    ConditionElse();                                                                              \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& cond_end() {                                                                             \
    ConditionEnd();                                                                               \
    return *this;                                                                                 \
  }                                                                                               \
  template <typename LT, typename RT>                                                             \
  Stack& for_start(const LT& lhs, const RT& rhs) {                                                \
    ForStart(lhs, rhs);                                                                           \
    return *this;                                                                                 \
  }                                                                                               \
  template <typename ST, typename ET>                                                             \
  Stack& for_start(const String& lhs, const ST& start, const ET& end) {                           \
    ForStart(lhs, start, end);                                                                    \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& for_start(const String& lhs, const String& start, const String& end) {                   \
    ForStart(lhs, start, end);                                                                    \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& for_end() {                                                                              \
    ForEnd();                                                                                     \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& while_start(const String& predicate) {                                                   \
    WhileStart(predicate);                                                                        \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& while_end() {                                                                            \
    WhileEnd();                                                                                   \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& switch_start(const String& predicate) {                                                  \
    SwitchStart(predicate);                                                                       \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& switch_case(const String& predicate = "") {                                              \
    SwitchCase(predicate);                                                                        \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& switch_end() {                                                                           \
    SwitchEnd();                                                                                  \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& block_start() {                                                                          \
    BlockStart();                                                                                 \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& block_end(bool block_docs = true) {                                                      \
    BlockEnd(block_docs);                                                                         \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& scope_start(const String& scope_def = "", const String& scope_ref = "") {                \
    ScopeStart(scope_def, scope_ref);                                                             \
    return *this;                                                                                 \
  }                                                                                               \
  Stack& scope_end() {                                                                            \
    ScopeEnd();                                                                                   \
    return *this;                                                                                 \
  }

/*!
 * \brief Stack Doc for common codegen
 */
class CodeStack : public BaseStack {
 public:
  /*!
   * \brief The constructor of CodeStack
   */
  CodeStack() : BaseStack() {}

  COMMON_WRAPPERS(CodeStack)
};

/*!
 * \brief Stack Doc for codes
 */
template <typename OpCodeGenType>
class OpCodeStack : public BaseStack {
 public:
  /*!
   * \brief The constructor of OpCodeStack
   */
  OpCodeStack() : BaseStack() {}

  /*! \brief Set codegen*/
  void Config(OpCodeGenType* codegen, bool reset = true) {
    codegen_ = codegen;
    if (reset) {
      Reset();
    }
  }

  COMMON_WRAPPERS(OpCodeStack<OpCodeGenType>)

  /*! \brief Push op_call Doc*/
  OpCodeStack<OpCodeGenType>& op_call(const String& callee = "msc::auto",
                                      const String& assign_to = "msc::auto") {
    const String& v_callee = callee == "msc::auto" ? codegen_->callee_name() : callee;
    const String& v_assign = assign_to == "msc::auto" ? codegen_->ret_name() : assign_to;
    return func_call(v_callee, v_assign);
  }

  /*! \brief Push op comment Doc*/
  OpCodeStack<OpCodeGenType>& op_comment(const String& comment_str = "msc::auto") {
    const String& v_comment = (comment_str == "msc::auto" ? codegen_->Comment() : comment_str);
    return comment(v_comment);
  }

  /*! \brief Cache typed attribute as argument*/
  template <typename T>
  OpCodeStack<OpCodeGenType>& op_arg(const String& attr_key, const String& key = "msc::auto") {
    T attr_val;
    if (codegen_->node()->GetAttr(attr_key, &attr_val)) {
      const String& valid_key = key == "msc::auto" ? attr_key : key;
      return call_arg(attr_val, valid_key);
    }
    return *this;
  }

  /*! \brief Cache str attribute as argument*/
  OpCodeStack<OpCodeGenType>& op_str_arg(const String& attr_key, const String& key = "msc::auto") {
    std::string attr_val;
    if (codegen_->node()->GetAttr(attr_key, &attr_val)) {
      const String& valid_key = key == "msc::auto" ? attr_key : key;
      return call_arg(DocUtils::ToStr(attr_val), valid_key);
    }
    return *this;
  }

  /*! \brief Cache list attribute as argument*/
  template <typename T>
  OpCodeStack<OpCodeGenType>& op_list_arg(const String& attr_key, const String& key = "msc::auto",
                                          bool allow_empty = false) {
    std::vector<T> attr_val;
    if (codegen_->node()->GetAttr(attr_key, &attr_val)) {
      const String& valid_key = key == "msc::auto" ? attr_key : key;
      return call_arg(DocUtils::ToList(attr_val, allow_empty), valid_key);
    }
    return *this;
  }

  /*! \brief Cache input as argument*/
  OpCodeStack<OpCodeGenType>& op_input_arg(int idx = 0, const String& key = "") {
    return call_arg(codegen_->IdxInput(idx, true), key);
  }

  /*! \brief Cache inputs as argument*/
  OpCodeStack<OpCodeGenType>& op_inputs_arg(bool as_list = true, const String& key = "") {
    Array<String> inputs;
    for (size_t i = 0; i < codegen_->node()->inputs.size(); i++) {
      inputs.push_back(codegen_->IdxInput(i, true));
    }
    if (as_list) {
      return call_arg(DocUtils::ToList(inputs), key);
    } else {
      return call_arg(DocUtils::ToDocList(inputs));
    }
  }

  /*! \brief Cache output as argument*/
  OpCodeStack<OpCodeGenType>& op_output_arg(int idx = 0, const String& key = "") {
    return call_arg(codegen_->IdxOutput(idx), key);
  }

  /*! \brief Cache weight as argument*/
  OpCodeStack<OpCodeGenType>& op_weight_arg(const String& wtype, const String& key = "") {
    if (codegen_->node()->weights.count(wtype)) {
      return call_arg(codegen_->IdxWeight(wtype, true), key);
    }
    return *this;
  }

  /*! \brief Cache name as argument*/
  OpCodeStack<OpCodeGenType>& op_name_arg(const String& key = "msc::auto",
                                          const String& name = "msc::auto") {
    const String& valid_key = key == "msc::auto" ? "name" : key;
    const String& valid_name = name == "msc::auto" ? codegen_->node()->name : name;
    return call_arg(DocUtils::ToStr(valid_name), valid_key);
    return *this;
  }

  OpCodeStack<OpCodeGenType>& op_dtype_arg(const DataType& dtype, const String& key = "") {
    return call_arg(codegen_->DType(dtype), key);
  }

 private:
  OpCodeGenType* codegen_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CODE_STACK_H_
