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
#ifndef SRC_SCRIPT_PRINTER_IR_UTILS_H_
#define SRC_SCRIPT_PRINTER_IR_UTILS_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/stmt.h>
#include <tvm/ir/type.h>
#include <tvm/script/printer/doc_translator.h>

#include <optional>
#include <utility>

namespace tvm {
namespace script {
namespace printer {

namespace details {

ffi::Optional<ExprDoc> InvokeDocHook(ffi::AnyView hook, DocTranslatorObj* d, ffi::AnyView input,
                                     const ffi::Object* destination = nullptr);
ffi::Optional<ExprDoc> VarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination);
ffi::Array<StmtDoc> Body(const Stmt& stmt, DocTranslatorObj* d);
ffi::Array<Doc> TensorIndices(DocTranslatorObj* d, const ffi::Array<PrimExpr>& indices,
                              bool store = false);
ExprDoc ForIterator(DocTranslatorObj* d, const ForNode* loop, ffi::Array<ffi::String> keys,
                    ffi::Array<ExprDoc> values, const ffi::Map<ffi::String, ffi::Any>& annotations,
                    bool thread_binding = false);
ExprDoc TensorRegionValue(DocTranslatorObj* d, const TensorRegionNode* region, bool require_region);

// Preserve nested owner contexts while keeping the shared defaults dialect-free.
template <typename T>
class ExtraStateScope {
 public:
  ExtraStateScope(DocTranslatorObj* d, const ffi::String& key, T value)
      : d_(d), key_(key), saved_(d->GetOrCreateExtraState<T>(key)) {
    d_->SetExtraState(key_, value);
  }
  ~ExtraStateScope() { d_->SetExtraState(key_, saved_); }

 private:
  DocTranslatorObj* d_;
  ffi::String key_;
  T saved_;
};

ExprDoc AddMetadata(DocTranslatorObj* d, ffi::Any value);
IdDoc VarDoc(DocTranslatorObj* d, const Var& var, bool explicit_def = true);
ExprDoc GlobalReference(DocTranslatorObj* d, const ffi::String& name);
ExprDoc NamedCallCallee(const ffi::String& canonical_name);

bool IsTypeValue(DocTranslatorObj* d, ffi::AnyView input);
ExprDoc TypeValue(DocTranslatorObj* d, const Type& type, bool dtype_literal = true);
ExprDoc MaterializeCallArgument(DocTranslatorObj* d, const Expr& arg, ExprDoc doc);
ExprDoc RawCall(DocTranslatorObj* d, const CallNode* call,
                ffi::Optional<ffi::Array<ExprDoc>> translated_args = std::nullopt);
ExprDoc AnyValue(DocTranslatorObj* d, ffi::AnyView value);
ffi::Dict<Var, IdDoc> CopyImplicitDefs(DocTranslatorObj* d);
void FinalizeFunctionDefinitions(DocTranslatorObj* d, const ffi::Dict<Var, IdDoc>& signature,
                                 const FunctionDoc& function);

class VarScope {
 public:
  explicit VarScope(DocTranslatorObj* d) : d_(d) { d_->BeginVarScope(); }
  ~VarScope() noexcept {
    if (d_) {
      try {
        d_->EndVarScope();
      } catch (...) { /* Preserve translation failure. */
      }
    }
  }
  void Close() {
    DocTranslatorObj* d = std::exchange(d_, nullptr);
    d->EndVarScope();
  }

 private:
  DocTranslatorObj* d_;
};

}  // namespace details

namespace type_attr {
// Generic constants delegate payload syntax to the component owning that value type.
inline constexpr const char* kConstantDocTranslate = "__tvm_doc_translate_constant__";
// Variables and stores delegate concrete type policies to their owner.
inline constexpr const char* kTensorStoreDocTranslate = "__tvm_doc_translate_tensor_store__";
// Optional For iterator syntax; the shared hook owns the loop variable and body.
inline constexpr const char* kForIteratorDocTranslate = "__tvm_doc_translate_for_iterator__";
inline constexpr const char* kVarDocTranslate = "__tvm_doc_translate_var__";
// A pointee type may name its complete pointer annotation constructor.
inline constexpr const char* kPointerConstructor = "__tvm_doc_pointer_constructor__";
// Ordering is independent of the ordinary child Doc translation hook.
inline constexpr const char* kModuleFunctionOrder = "__tvm_doc_module_function_order__";
}  // namespace type_attr

namespace op_attr {
// Receives the original RegionStmt and returns its scope expression.
inline constexpr const char* kRegionDocTranslate = "__tvm_doc_translate_region__";
}  // namespace op_attr

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // SRC_SCRIPT_PRINTER_IR_UTILS_H_
