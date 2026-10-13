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

/*! \brief Check if TIRx is well-formed. */

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {
namespace {

// This state and traversal belong to well-formedness checking. In particular,
// ordinary visitors do not impose a definition/use or lexical-scope policy.
class WellFormednessVerifier : public StmtExprVisitor {
  friend class tvm::ObjectVisitor;

 public:
  enum class Check { kVariables, kTensors, kTensorLoadType, kLoopControl, kThreadBinding };
  using AccessPath = ffi::reflection::AccessPath;
  using Result = ffi::Optional<VisitInterrupt>;

  WellFormednessVerifier(Check check, bool assert_mode)
      : WellFormednessVerifier(GlobalVTable(), check, assert_mode) {}

  bool Run(const ffi::ObjectRef& root) {
    root_ = root;
    try {
      this->Visit(root);
    } catch (const ffi::Error& error) {
      // Resolve at the verifier boundary for both direct and pass-driven calls.
      // Once reported, consume the visit context to avoid a second enrichment.
      auto context = ffi::VisitErrorContext::TryGetFromError(error);
      if (!context) throw;
      auto normalized = ffi::make_object<ffi::VisitErrorContextObj>();
      normalized->prev_error_context = context.value()->prev_error_context;
      for (const auto& object : context.value()->reverse_visit_pattern) {
        if (!IsContainer(object)) normalized->reverse_visit_pattern.push_back(object);
      }
      auto location = ffi::VisitErrorContext(std::move(normalized));
      auto paths = ffi::VisitErrorContext::FindAccessPaths(root, location);
      if (paths.empty()) throw;
      std::ostringstream path;
      path << paths[0];
      std::string message = error.message();
      const std::string marker = "<current location>";
      size_t pos = 0;
      bool replaced = false;
      while ((pos = message.find(marker, pos)) != std::string::npos) {
        message.replace(pos, marker.size(), path.str());
        pos += path.str().size();
        replaced = true;
      }
      if (!replaced) message += " at " + path.str();
      throw ffi::Error(error.kind(), message, error.backtrace(), error.cause_chain(),
                       location->prev_error_context.has_value()
                           ? std::optional<ffi::ObjectRef>(location->prev_error_context.value())
                           : std::nullopt);
    }
    return !has_error_;
  }

 protected:
  using VTable = StmtExprVisitor::VTable;
  using StmtExprVisitor::Visit_;
  explicit WellFormednessVerifier(const VTable* table, Check check, bool assert_mode)
      : StmtExprVisitor(table), check_(check), table_(table), assert_mode_(assert_mode) {}

  static void InitVTable(VTable* table) {
    StmtExprVisitor::InitVTable(table);
    StmtExprVisitor::SetDispatch<WellFormednessVerifier, IRModuleNode>(table);
    StmtExprVisitor::SetDispatch<WellFormednessVerifier, FunctionNode>(table);
    StmtExprVisitor::SetDispatch<WellFormednessVerifier, LambdaExprNode>(table);
  }

  Result Visit(ffi::AnyView value) override {
    auto object = value.as<ffi::ObjectRef>();
    if (!object) return StmtExprVisitor::Visit(value);
    visit_stack_.push_back(object.value());
    struct Pop {
      std::vector<ffi::ObjectRef>* stack;
      ~Pop() { stack->pop_back(); }
    } pop{&visit_stack_};
    // Function entrypoints and module members are roots of verification;
    // a Function embedded as an expression remains unsupported.
    if (value.as<Function>() && visit_stack_.size() != 1 &&
        !(visit_stack_.size() == 2 && root_.as<IRModule>())) {
      TVM_FFI_VISIT_THROW(TypeError, object.value())
          << "Unsupported non-primitive TIR expression " << object.value()->GetTypeKey();
    }
    if (value.as<Stmt>() && !table_->CanDispatch(value)) {
      TVM_FFI_VISIT_BEGIN();
      Verify(false) << "TIR verifier does not support statement " << object.value()->GetTypeKey()
                    << " at <current location>";
      return std::nullopt;
      TVM_FFI_VISIT_END(object.value());
    }
    if (value.as<Expr>() && !table_->CanDispatch(value)) {
      if (value.as<OpaqueExprNode>()) return std::nullopt;
      if (auto var = value.as<VarNode>()) {
        TVM_FFI_VISIT_BEGIN();
        return this->Visit_(var);
        TVM_FFI_VISIT_END(object.value());
      }
      TVM_FFI_VISIT_THROW(TypeError, object.value())
          << "Unsupported non-primitive TIR expression " << object.value()->GetTypeKey();
    }
    return StmtExprVisitor::Visit(value);
  }

  class VerifyStream {
   public:
    explicit VerifyStream(bool fail) {
      if (fail) message_.emplace();
    }
    VerifyStream(const VerifyStream&) = delete;
    VerifyStream(VerifyStream&& other) { std::swap(message_, other.message_); }
    template <typename T>
    VerifyStream& operator<<(T&& value) {
      if (message_) *message_ << std::forward<T>(value);
      return *this;
    }
    ~VerifyStream() noexcept(false) {
      if (message_) LOG(FATAL) << message_->str();
    }

   private:
    std::optional<std::ostringstream> message_;
  };

  VerifyStream Verify(bool condition) {
    has_error_ |= !condition;
    return VerifyStream(!condition && assert_mode_);
  }

  // Only earlier definitions need a saved occurrence. Keep their enclosing
  // objects and the defining field, rather than searching for a Var that may
  // also occur at arbitrarily many use sites.
  struct DefinitionSite {
    ffi::VisitErrorContext context;
    AccessPath field;
  };

  // FindAccessPaths descends through containers without matching their identity.
  // Native visiting can record them; keep only matchable object breadcrumbs.
  static bool IsContainer(const ffi::ObjectRef& object) {
    switch (object->type_index()) {
      case ffi::TypeIndex::kTVMFFIArray:
      case ffi::TypeIndex::kTVMFFIList:
      case ffi::TypeIndex::kTVMFFIMap:
      case ffi::TypeIndex::kTVMFFIDict:
        return true;
      default:
        return false;
    }
  }

  DefinitionSite DefinitionAt(AccessPath field) const {
    auto context = ffi::make_object<ffi::VisitErrorContextObj>();
    for (auto it = visit_stack_.rbegin(); it != visit_stack_.rend(); ++it) {
      if (!IsContainer(*it)) context->reverse_visit_pattern.push_back(*it);
    }
    return {ffi::VisitErrorContext(std::move(context)), std::move(field)};
  }

  AccessPath DefinitionPath(const DefinitionSite& site) const {
    auto paths = ffi::VisitErrorContext::FindAccessPaths(root_, site.context);
    auto path = paths.empty() ? AccessPath::Root() : paths[0];
    for (const auto& step : site.field->ToSteps()) path = path->Extend(step);
    return path;
  }

  template <typename F>
  Result WithScope(F&& body) {
    return scopes_.WithNewScope([&]() -> Result {
      Result result = body();
      // Definition exit never diagnoses an error. On an exception the entire
      // verifier is discarded, so only normal completion needs history updates.
      for (auto it = scopes_.Current().rbegin(); it != scopes_.Current().rend(); ++it) {
        current_.erase(it->first);
        previous_.insert(*it);
      }
      return result;
    });
  }

  void Define(const Var& var, AccessPath field, bool visit_type = true) {
    if (visit_type) this->Visit(var->ty);
    auto site = DefinitionAt(std::move(field));
    if (check_ == Check::kVariables) {
      {
        auto it = current_.find(var);
        auto verify = Verify(it == current_.end());
        verify << "ValueError: TIR is ill-formed, due to multiple nested definitions of variable "
               << var->name << ".";
        if (it != current_.end()) {
          verify << " It was first defined at " << DefinitionPath(it->second)
                 << ", and was re-defined at <current location>";
        }
      }
      {
        auto it = previous_.find(var);
        auto verify = Verify(it == previous_.end());
        verify << "ValueError: TIR is ill-formed, due to multiple definitions of variable "
               << var->name << ".";
        if (it != previous_.end()) {
          verify << " It was first defined at " << DefinitionPath(it->second)
                 << ", and was later re-defined at <current location>";
        }
      }
    }
    current_.insert({var, site});
    scopes_.Current().emplace_back(var, std::move(site));
  }

  void MatchTensorDefinitions(const TensorVar& tensor, AccessPath field) {
    auto define_if_needed = [&](const Var& var, AccessPath site) {
      if (!current_.count(var)) Define(var, std::move(site));
    };
    for (size_t i = 0; i < tensor->shape.size(); ++i) {
      auto site = field->Attr("shape")->ArrayItem(i);
      auto walk = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        define_if_needed(var, site);
        return ffi::WalkResult::Advance();
      };
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(tensor->shape[i], walk);
    }
    for (size_t i = 0; i < tensor->strides.size(); ++i) {
      if (auto var = tensor->strides[i].as<Var>()) {
        define_if_needed(var.value(), field->Attr("strides")->ArrayItem(i));
      }
    }
    if (auto var = tensor->elem_offset.as<Var>()) {
      define_if_needed(var.value(), field->Attr("elem_offset"));
    }
  }

  Result Visit_(const IRModuleNode* op) {
    std::vector<GlobalVar> globals;
    for (const auto& [global, function] : op->functions) globals.push_back(global);
    std::sort(globals.begin(), globals.end(), [&](const GlobalVar& a, const GlobalVar& b) {
      bool a_public = op->functions[a]->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value();
      bool b_public = op->functions[b]->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value();
      return a_public != b_public ? a_public > b_public : a->name_hint < b->name_hint;
    });
    for (const auto& global : globals) {
      if (auto function = op->functions[global].as<Function>()) this->Visit(function.value());
    }
    return std::nullopt;
  }

  Result Visit_(const FunctionNode* op) {
    int enclosing_depth = loop_depth_;
    loop_depth_ = 0;
    WithScope([&]() -> Result {
      auto params = AccessPath::Root()->Attr("params");
      for (size_t i = 0; i < op->params.size(); ++i) {
        if (!op->params[i]->ty.as<TensorTypeNode>()) Define(op->params[i], params->ArrayItem(i));
      }
      for (size_t i = 0; i < op->params.size(); ++i) {
        if (auto tensor = op->params[i].as<TensorVar>()) {
          MatchTensorDefinitions(tensor.value(), params->ArrayItem(i)->Attr("ty"));
        }
      }
      for (size_t i = 0; i < op->params.size(); ++i) {
        if (op->params[i].as<TensorVar>()) Define(op->params[i], params->ArrayItem(i));
      }
      return WithScope([&]() { return this->Visit(op->body); });
    });
    loop_depth_ = enclosing_depth;
    if (check_ == Check::kTensors) {
      current_.clear();
      previous_.clear();
    }
    return std::nullopt;
  }

  Result Visit_(const VarNode* op) override {
    auto var = ffi::GetRef<Var>(op);
    if (check_ == Check::kVariables) {
      auto verify = Verify(current_.count(var));
      verify << "ValueError: Invalid use of undefined variable " << var->name
             << " at <current location>.";
      if (auto it = previous_.find(var); it != previous_.end()) {
        verify << ".  While this variable was previously defined at " << DefinitionPath(it->second)
               << ", this definition is no longer in-scope.";
      }
    }
    return std::nullopt;
  }

  void TensorUse(const TensorVar& tensor) {
    if (check_ != Check::kTensors) return;
    if (current_.count(tensor.var())) return;
    auto it = previous_.find(tensor.var());
    auto verify = Verify(false);
    verify << "TIR is ill-formed: tensor " << tensor.name() << " is used at <current location>";
    if (it != previous_.end()) {
      verify << " but its declaration is no longer in-scope. It was declared at "
             << DefinitionPath(it->second) << ".";
    } else {
      verify << " without a prior DeclTensor or other declaration.";
    }
  }

  Result Visit_(const BindNode* op) override {
    this->Visit(op->value);
    Define(op->var, AccessPath::Root()->Attr("var"));
    return std::nullopt;
  }

  Result Visit_(const RegionStmtNode* op) override {
    this->Visit(op->args);
    this->Visit(op->attrs);
    for (const auto& param : op->body_params) this->Visit(param->ty);
    WithScope([&]() -> Result {
      for (size_t i = 0; i < op->body_params.size(); ++i) {
        Define(op->body_params[i], AccessPath::Root()->Attr("body_params")->ArrayItem(i), false);
      }
      return this->Visit(op->body);
    });
    for (size_t i = 0; i < op->result_vars.size(); ++i) {
      Define(op->result_vars[i], AccessPath::Root()->Attr("result_vars")->ArrayItem(i));
    }
    return std::nullopt;
  }

  Result Visit_(const ForNode* op) override {
    this->Visit(op->min);
    this->Visit(op->extent);
    if (check_ == Check::kLoopControl) {
      this->Visit(op->step);
      ++loop_depth_;
      this->Visit(op->body);
      --loop_depth_;
      return std::nullopt;
    }
    return WithScope([&]() -> Result {
      Define(op->loop_var, AccessPath::Root()->Attr("loop_var"));
      return WithScope([&]() { return this->Visit(op->body); });
    });
  }

  Result Visit_(const WhileNode* op) override {
    this->Visit(op->condition);
    ++loop_depth_;
    WithScope([&]() { return this->Visit(op->body); });
    --loop_depth_;
    return std::nullopt;
  }

  Result Visit_(const IfNode* op) override {
    this->Visit(op->condition);
    WithScope([&]() { return this->Visit(op->then_case); });
    return WithScope([&]() { return this->Visit(op->else_case); });
  }

  Result Visit_(const prim::LetNode* op) override {
    this->Visit(op->value);
    return WithScope([&]() -> Result {
      Define(op->var, AccessPath::Root()->Attr("var"));
      return this->Visit(op->body);
    });
  }

  Result Visit_(const LambdaExprNode* op) {
    this->Visit(op->ty);
    return WithScope([&]() -> Result {
      for (size_t i = 0; i < op->vars.size(); ++i) {
        Define(op->vars[i], AccessPath::Root()->Attr("vars")->ArrayItem(i));
      }
      return this->Visit(op->body);
    });
  }

  Result Visit_(const CallNode* op) override {
    if (op->op.as<GlobalVar>() || op->op.as<OpaqueExprNode>()) this->Visit(op->op);
    return this->Visit(op->args);
  }

  Result Visit_(const TensorStoreNode* op) override {
    this->Visit(op->value);
    TensorUse(op->dest.as_or_throw<TensorVar>());
    return this->Visit(op->indices);
  }

  Result Visit_(const TensorRegionNode* op) override {
    if (auto tensor = op->source.as<TensorVar>()) {
      TensorUse(tensor.value());
    } else {
      this->Visit(op->source);
    }
    return this->Visit(op->region);
  }

  Result Visit_(const GenericConstNode* op) override { return std::nullopt; }

  Result Visit_(const AssertStmtNode* op) override {
    this->Visit(op->condition);
    this->Visit(op->error_kind);
    return this->Visit(op->message_parts);
  }

  Result Visit_(const prim::RampNode* op) override {
    this->Visit(op->base);
    this->Visit(op->stride);
    return this->Visit(op->lanes);
  }

  Result Visit_(const prim::BroadcastNode* op) override {
    this->Visit(op->value);
    return this->Visit(op->lanes);
  }

  Result Visit_(const BreakNode* op) override {
    if (check_ == Check::kLoopControl) {
      Verify(loop_depth_ > 0) << "ValueError: break at <current location> requires an enclosing "
                                 "loop in the same function.";
    }
    return std::nullopt;
  }

  Result Visit_(const ContinueNode* op) override {
    if (check_ == Check::kLoopControl) {
      Verify(loop_depth_ > 0) << "ValueError: continue at <current location> requires an enclosing "
                                 "loop in the same function.";
    }
    return std::nullopt;
  }

  Result Visit_(const TensorLoadNode* op) override;

  Check check_;

 private:
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }

  const VTable* table_;
  bool assert_mode_;
  bool has_error_{false};
  int loop_depth_{0};
  ffi::ObjectRef root_;
  std::vector<ffi::ObjectRef> visit_stack_;
  ScopeStack<std::vector<std::pair<Var, DefinitionSite>>> scopes_;
  std::unordered_map<Var, DefinitionSite> current_;
  std::unordered_map<Var, DefinitionSite> previous_;
};

auto WellFormednessVerifier::Visit_(const TensorLoadNode* op) -> Result {
  if (check_ != Check::kTensorLoadType) {
    TensorUse(op->source.as_or_throw<TensorVar>());
    return this->Visit(op->indices);
  }
  auto tensor = op->source.as<TensorVar>();
  auto valid_source = Verify(tensor.has_value());
  valid_source << "TypeError: TIR TensorLoad source at " << "<current location>.source"
               << " must be a TensorVar.";
  if (!tensor.has_value()) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  bool valid_indices = tensor.value()->shape.size() == op->indices.size();
  auto valid_rank = Verify(valid_indices);
  valid_rank << "ValueError: TIR TensorLoad at " << "<current location>" << " indexes "
             << tensor.value()->shape.size() << "-dimensional tensor " << tensor.value().name()
             << " with " << op->indices.size() << " indices.";
  if (!valid_indices) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  for (size_t i = 0; i + 1 < op->indices.size(); ++i) {
    bool is_scalar = op->indices[i].ty().IsScalar();
    auto valid_index = Verify(is_scalar);
    valid_index << "TypeError: TIR TensorLoad index " << i << " at "
                << "<current location>.indices[" << i << "]"
                << " must be scalar because only the final index may be vector-valued.";
    valid_indices = valid_indices && is_scalar;
  }
  if (!valid_indices) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  ffi::Optional<PrimType> index_ty = op->indices.empty()
                                         ? ffi::Optional<PrimType>(tensor.value()->dtype)
                                         : op->indices.back().ty().as<PrimType>();
  auto valid_index_type = Verify(index_ty.has_value());
  valid_index_type << "TypeError: TIR TensorLoad final index at <current location>.indices";
  if (!op->indices.empty()) valid_index_type << "[" << op->indices.size() - 1 << "]";
  valid_index_type << " must have a primitive type.";
  if (!index_ty.has_value()) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  bool scalable_compatible = op->indices.empty() || !(tensor.value()->dtype.IsScalableVector() &&
                                                      index_ty.value().IsScalableVector());
  auto valid_scalability = Verify(scalable_compatible);
  valid_scalability << "TypeError: TIR TensorLoad at " << "<current location>"
                    << " cannot combine a scalable tensor dtype with a scalable index.";
  if (!scalable_compatible) {
    this->Visit(op->indices);
    return std::nullopt;
  }

  TensorLoad expected = MakeTensorLoad(tensor.value(), op->indices, op->loc);
  ffi::Optional<PrimType> asserted_ty = op->ty.as<PrimType>();
  ffi::Optional<PrimType> expected_ty = expected->ty.as<PrimType>();
  auto valid_type = Verify(asserted_ty.has_value() && expected_ty.has_value() &&
                           asserted_ty.value() == expected_ty.value());
  valid_type << "TypeError: TIR TensorLoad at " << "<current location>" << " asserts result type "
             << op->ty << ", but its source and indices imply " << expected->ty << ".";
  TensorUse(tensor.value());
  return this->Visit(op->indices);
}

bool VerifyWellFormedCommon(const ffi::ObjectRef& node, bool assert_mode) {
  using Check = WellFormednessVerifier::Check;
  for (Check check :
       {Check::kVariables, Check::kTensors, Check::kTensorLoadType, Check::kLoopControl}) {
    if (!ffi::make_object<WellFormednessVerifier>(check, assert_mode)->Run(node)) return false;
  }
  return true;
}

}  // namespace

bool VerifyWellFormed(const Function& func, bool assert_mode) {
  return VerifyWellFormedCommon(func, assert_mode);
}

bool VerifyWellFormed(const IRModule& mod, bool assert_mode) {
  return VerifyWellFormedCommon(mod, assert_mode);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.analysis.VerifyWellFormed", [](const ffi::ObjectRef& obj, bool assert_mode) {
        if (auto opt = obj.as<Function>()) {
          return VerifyWellFormed(opt.value(), assert_mode);
        } else if (auto opt = obj.as<IRModule>()) {
          return VerifyWellFormed(opt.value(), assert_mode);
        } else {
          TVM_FFI_THROW(InternalError)
              << "Expected VerifyWellFormed argument to be a Function or IRModule, but found "
              << obj->GetTypeKey();
        }
      });
}

}  // namespace tirx
}  // namespace tvm
