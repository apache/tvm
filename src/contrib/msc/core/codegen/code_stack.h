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
  void Comment(const String& comment = "");

  /*! \brief Push assign Doc*/
  void AssignBase(const String& lhs, const ExprDoc& rhs, const String& annotation = "");

  /*! \brief Push typed assign Doc*/
  template <typename T>
  void Assign(const String& lhs, const T& rhs, const String& annotation = "") {
    AssignBase(lhs, DocUtils::ToDoc(rhs), annotation);
  }

  /*! \brief Push assign for list Doc*/
  template <typename T>
  inline void AssignList(const String& lhs, const std::vector<T>& rhs,
                         const String& annotation = "") {
    AssignBase(lhs, DocUtils::ToListDoc(rhs), annotation);
  }

  template <typename T>
  inline void AssignList(const String& lhs, const Array<T>& rhs, const String& annotation = "") {
    AssignBase(lhs, DocUtils::ToListDoc(rhs), annotation);
  }

  /*! \brief Push assign for index Doc*/
  void AssignIndexBase(const String& lhs, const String& rhs, const Array<ExprDoc>& indices,
                       const String& annotation = "");

  template <typename T>
  inline void AssignIndex(const String& lhs, const String& rhs, const std::vector<T>& indices,
                          const String& annotation = "") {
    AssignIndexBase(lhs, rhs, DocUtils::ToDocList(indices), annotation);
  }

  template <typename T>
  inline void AssignIndex(const String& lhs, const String& rhs, const Array<T>& indices,
                          const String& annotation = "") {
    AssignIndexBase(lhs, rhs, DocUtils::ToDocList(indices), annotation);
  }

  inline void AssignIndex(const String& lhs, const String& rhs, const Array<ExprDoc>& indices,
                          const String& annotation = "") {
    AssignIndexBase(lhs, rhs, indices, annotation);
  }

  /*! \brief Push attr access Doc*/
  void AttrAccess(const String& attr);

  /*! \brief Cache function Doc*/
  void FuncDef(const String& func_name, const String& ret_type = "");

  /*! \brief Cache func argument*/
  void FuncArg(const String& arg, const String& annotation = "", const String& value = "");

  /*! \brief Cache func decorator*/
  void FuncDecorator(const String& decorator);

  /*! \brief Start function body block*/
  void FuncStart();

  /*! \brief End function body block*/
  void FuncEnd(const String& ret_val = "");

  /*! \brief Cache class Doc*/
  void ClassDef(const String& class_name);

  /*! \brief Cache class decorator*/
  void ClassDecorator(const String& decorator);

  /*! \brief Start class body block*/
  void ClassStart();

  /*! \brief End class body block*/
  void ClassEnd();

  /*! \brief Cache call Doc*/
  void CallStart(const String& callee);

  /*! \brief Push call or/and assign Doc*/
  void CallEnd(const String& assign = "");

  /*! \brief Cache inplace call Doc*/
  void InplaceStart(const String& callee);

  /*! \brief Push inplace call or/and assign Doc*/
  void InplaceEnd();

  /*! \brief Cache call argument*/
  void CallArgBase(const ExprDoc& value, const String& key = "");

  template <typename T>
  inline void CallArg(T value, const String& key = "") {
    CallArgBase(DocUtils::ToDoc(value), key);
  }

  void CallStrArg(const String& value, const String& key = "");

  /*! \brief Cache call list argument*/
  void CallListArgBase(const Array<ExprDoc>& values, const String& key = "",
                       bool allow_empty = false, bool as_list = true);

  template <typename T>
  inline void CallListArg(const std::vector<T>& values, const String& key = "",
                          bool allow_empty = false, bool as_list = true) {
    return CallListArgBase(DocUtils::ToDocList(values), key, allow_empty, as_list);
  }

  template <typename T>
  inline void CallListArg(const Array<T>& values, const String& key = "", bool allow_empty = false,
                          bool as_list = true) {
    return CallListArgBase(DocUtils::ToDocList(values), key, allow_empty, as_list);
  }

  inline void CallListArg(const Array<ExprDoc>& values, const String& key = "",
                          bool allow_empty = false, bool as_list = true) {
    return CallListArgBase(values, key, allow_empty, as_list);
  }

  /*! \brief Cache call inplace func argument*/
  void CallInplaceStart(const String& callee);

  /*! \brief Push call inplace func argument*/
  void CallInplaceEnd(const String& key = "");

  /*! \brief Push if to cache and start if block*/
  void ConditionIf(const String& predicate);

  /*! \brief Push then branch to cached and start block*/
  void ConditionElse();

  /*! \brief Push else branch to cached*/
  void ConditionEnd();

  /*! \brief Start a new block*/
  void BlockStart();

  /*! \brief End a block*/
  void BlockEnd(bool block_docs = true);

  /*! \brief Start a new scope*/
  void ScopeStart(const String& scope_def, const String& scope_ref = "");

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

#define COMMON_WRAPPERS(Stack)                                                                   \
  Stack& line(const Doc& doc) {                                                                  \
    Line(doc);                                                                                   \
    return *this;                                                                                \
  }                                                                                              \
  Stack& line(const String& line = "") {                                                         \
    Line(line);                                                                                  \
    return *this;                                                                                \
  }                                                                                              \
  Stack& comment(const String& comment) {                                                        \
    Comment(comment);                                                                            \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& assign(const String& lhs, const T& rhs, const String& annotation = "") {                \
    Assign(lhs, rhs, annotation);                                                                \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& assign_list(const String& lhs, const std::vector<T>& rhs,                               \
                     const String& annotation = "") {                                            \
    AssignList(lhs, rhs, annotation);                                                            \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& assign_list(const String& lhs, const Array<T>& rhs, const String& annotation = "") {    \
    AssignList(lhs, rhs, annotation);                                                            \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& assign_index(const String& lhs, const String& rhs, const std::vector<T>& indices,       \
                      const String& annotation = "") {                                           \
    AssignIndex(lhs, rhs, indices, annotation);                                                  \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& assign_index(const String& lhs, const String& rhs, const Array<T>& indices,             \
                      const String& annotation = "") {                                           \
    AssignIndex(lhs, rhs, indices, annotation);                                                  \
    return *this;                                                                                \
  }                                                                                              \
  Stack& attr_access(const String& attr) {                                                       \
    AttrAccess(attr);                                                                            \
    return *this;                                                                                \
  }                                                                                              \
  Stack& block_start() {                                                                         \
    BlockStart();                                                                                \
    return *this;                                                                                \
  }                                                                                              \
  Stack& block_end(bool block_docs = true) {                                                     \
    BlockEnd(block_docs);                                                                        \
    return *this;                                                                                \
  }                                                                                              \
  Stack& scope_start(const String& scope_def, const String& scope_ref = "") {                    \
    ScopeStart(scope_def, scope_ref);                                                            \
    return *this;                                                                                \
  }                                                                                              \
  Stack& scope_end() {                                                                           \
    ScopeEnd();                                                                                  \
    return *this;                                                                                \
  }                                                                                              \
  Stack& func_def(const String& func_name, const String& ret_type = "") {                        \
    FuncDef(func_name, ret_type);                                                                \
    return *this;                                                                                \
  }                                                                                              \
  Stack& func_arg(const String& arg, const String& annotation = "", const String& value = "") {  \
    FuncArg(arg, annotation, value);                                                             \
    return *this;                                                                                \
  }                                                                                              \
  Stack& func_decorator(const String& decorator) {                                               \
    FuncDecorator(decorator);                                                                    \
    return *this;                                                                                \
  }                                                                                              \
  Stack& func_start() {                                                                          \
    FuncStart();                                                                                 \
    return *this;                                                                                \
  }                                                                                              \
  Stack& func_end(const String& ret_val = "") {                                                  \
    FuncEnd(ret_val);                                                                            \
    return *this;                                                                                \
  }                                                                                              \
  Stack& class_def(const String& class_name) {                                                   \
    ClassDef(class_name);                                                                        \
    return *this;                                                                                \
  }                                                                                              \
  Stack& class_decorator(const String& decorator) {                                              \
    ClassDecorator(decorator);                                                                   \
    return *this;                                                                                \
  }                                                                                              \
  Stack& class_start() {                                                                         \
    ClassStart();                                                                                \
    return *this;                                                                                \
  }                                                                                              \
  Stack& class_end() {                                                                           \
    ClassEnd();                                                                                  \
    return *this;                                                                                \
  }                                                                                              \
  Stack& call_start(const String& callee) {                                                      \
    CallStart(callee);                                                                           \
    return *this;                                                                                \
  }                                                                                              \
  Stack& call_end(const String& assign = "") {                                                   \
    CallEnd(assign);                                                                             \
    return *this;                                                                                \
  }                                                                                              \
  Stack& inplace_start(const String& callee) {                                                   \
    InplaceStart(callee);                                                                        \
    return *this;                                                                                \
  }                                                                                              \
  Stack& inplace_end() {                                                                         \
    InplaceEnd();                                                                                \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& call_arg(T value, const String& key = "") {                                             \
    CallArg(value, key);                                                                         \
    return *this;                                                                                \
  }                                                                                              \
  Stack& call_str_arg(const String& value, const String& key = "") {                             \
    CallStrArg(value, key);                                                                      \
    return *this;                                                                                \
  }                                                                                              \
  Stack& call_list_arg(const Array<ExprDoc>& values, const String& key = "",                     \
                       bool allow_empty = false, bool as_list = true) {                          \
    CallListArg(values, key, allow_empty, as_list);                                              \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& call_list_arg(const std::vector<T>& values, const String& key = "",                     \
                       bool allow_empty = false, bool as_list = true) {                          \
    CallListArg(values, key, allow_empty, as_list);                                              \
    return *this;                                                                                \
  }                                                                                              \
  template <typename T>                                                                          \
  Stack& call_list_arg(const Array<T>& values, const String& key = "", bool allow_empty = false, \
                       bool as_list = true) {                                                    \
    CallListArg(values, key, allow_empty, as_list);                                              \
    return *this;                                                                                \
  }                                                                                              \
  Stack& call_inplace_start(const String& callee) {                                              \
    CallInplaceStart(callee);                                                                    \
    return *this;                                                                                \
  }                                                                                              \
  Stack& call_inplace_end(const String& key = "") {                                              \
    CallInplaceEnd(key);                                                                         \
    return *this;                                                                                \
  }                                                                                              \
  Stack& cond_if(const String& predicate) {                                                      \
    ConditionIf(predicate);                                                                      \
    return *this;                                                                                \
  }                                                                                              \
  Stack& cond_else() {                                                                           \
    ConditionElse();                                                                             \
    return *this;                                                                                \
  }                                                                                              \
  Stack& cond_end() {                                                                            \
    ConditionEnd();                                                                              \
    return *this;                                                                                \
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

  /*! \brief Cache op_call Doc*/
  OpCodeStack<OpCodeGenType>& op_start(const String& callee = "msc::auto") {
    const String& v_callee = callee == "msc::auto" ? codegen_->callee_name() : callee;
    return call_start(v_callee);
  }

  /*! \brief Push op_call Doc*/
  OpCodeStack<OpCodeGenType>& op_end(const String& assign_str = "msc::auto") {
    const String& v_assign = assign_str == "msc::auto" ? codegen_->IdxNode(true) : assign_str;
    return call_end(v_assign);
  }

  /*! \brief Push op comment Doc*/
  OpCodeStack<OpCodeGenType>& op_comment(const String& comment_str = "msc::auto") {
    const String& v_comment = (comment_str == "msc::auto" ? codegen_->Comment() : comment_str);
    return comment(v_comment);
  }

  /*! \brief Cache attribute as argument*/
  template <typename T>
  OpCodeStack<OpCodeGenType>& op_arg(const String& attr_key, const String& key = "msc::auto") {
    T attr_val;
    if (codegen_->node()->GetAttr(attr_key, &attr_val)) {
      const String& valid_key = key == "msc::auto" ? attr_key : key;
      return call_arg(attr_val, valid_key);
    }
    return *this;
  }

  OpCodeStack<OpCodeGenType>& op_str_arg(const String& attr_key, const String& key = "msc::auto") {
    std::string attr_val;
    if (codegen_->node()->GetAttr(attr_key, &attr_val)) {
      const String& valid_key = key == "msc::auto" ? attr_key : key;
      return call_str_arg(attr_val, valid_key);
    }
    return *this;
  }

  /*! \brief Cache list attribute as argument*/
  template <typename T>
  OpCodeStack<OpCodeGenType>& op_list_arg(const String& attr_key, const String& key = "msc::auto",
                                          bool allow_empty = false, bool as_list = true) {
    std::vector<T> attr_val;
    if (codegen_->node()->GetAttr(attr_key, &attr_val)) {
      const String& valid_key = key == "msc::auto" ? attr_key : key;
      return call_list_arg(attr_val, valid_key, allow_empty, as_list);
    }
    return *this;
  }

  /*! \brief Cache input as argument*/
  OpCodeStack<OpCodeGenType>& op_input_arg(int idx = 0, const String& key = "") {
    return call_arg(codegen_->IdxInput(idx, false), key);
  }

  /*! \brief Cache inputs as argument*/
  OpCodeStack<OpCodeGenType>& op_inputs_arg(bool as_list = true, const String& key = "") {
    Array<String> inputs;
    for (size_t i = 0; i < codegen_->node()->inputs.size(); i++) {
      inputs.push_back(codegen_->IdxInput(i, false));
    }
    return call_list_arg(inputs, key, false, as_list);
  }

  /*! \brief Cache output as argument*/
  OpCodeStack<OpCodeGenType>& op_output_arg(int idx = 0, const String& key = "") {
    return call_arg(codegen_->IdxOutput(idx, false), key);
  }

  /*! \brief Cache weight as argument*/
  OpCodeStack<OpCodeGenType>& op_weight_arg(const String& wtype, const String& key = "") {
    if (codegen_->node()->weights.count(wtype)) {
      return call_arg(codegen_->IdxWeight(wtype, false), key);
    }
    return *this;
  }

  OpCodeStack<OpCodeGenType>& call_dtype_arg(const DataType& dtype, const String& key = "") {
    return call_arg(codegen_->DType(dtype), key);
  }

 private:
  OpCodeGenType* codegen_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CODE_STACK_H_
