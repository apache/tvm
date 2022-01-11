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
#include <tvm/ir/attrs.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace transform {

namespace {

/*! \brief Collect all attributes whose name contains "layout".
 */
struct CollectAttrs : public AttrVisitor {
  void Visit(const char* key, std::string* value) final {
    if (std::string(key).find("layout") != std::string::npos) {
      attrs[key] = String(*value);
    }
  }
  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, runtime::NDArray* value) final {}
  void Visit(const char* key, ObjectRef* value) final {
    if (std::string(key).find("layout") != std::string::npos) {
      attrs[key] = *value;
    }
  }
  void Visit(const char* key, DataType* value) final {}
  void Visit(const char* key, void** value) final {}
  std::unordered_map<std::string, ObjectRef> attrs;
};
}  // namespace

/*! \brief Visitor to add structural hash and layout information to `Function`
 * nodes. Sets the "hash" field on the attr to the structural hash of the
 * function. Propogates any attributes with "layout" in their name from call
 * nodes in the Function to the Function's attrs.
 */
class LabelOpsMutator : public MixedModeMutator {
 private:
  using MixedModeMutator::VisitExpr_;
  std::unordered_map<std::string, ObjectRef> body_attrs;
  Expr VisitExpr_(const FunctionNode* op) final {
    if (op->GetAttr<String>("hash").defined()) {
      // Already labelled.
      return ExprMutator::VisitExpr_(op);
    }

    // body_attrs collects attrs from Calls in the body of this Function. Reset
    // it so we only get attrs from this Function.
    body_attrs = {};
    auto updated = ExprMutator::VisitExpr_(op);
    size_t hash = StructuralHash()(updated);

    // format hash as fixed length hex string so it is easier to read
    std::stringstream s;
    s << std::setfill('0') << std::setw(sizeof(size_t) * 2) << std::hex << hash;

    Function f = WithAttr(Downcast<Function>(updated), "hash", String(s.str()));
    for (auto p : body_attrs) {
      f = WithAttr(f, p.first, p.second);
    }
    return std::move(f);
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      this->Mutate(op->var);
      this->Mutate(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Var var = Downcast<Var>(this->Mutate(op->var));
      auto value = this->Mutate(op->value);
      auto body = this->Mutate(op->body);
      auto expr = GetRef<Expr>(op);
      if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
        this->memo_[expr] = expr;
      } else {
        this->memo_[expr] = Let(var, value, body);
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr Rewrite_(const CallNode* op, const Expr& post) final {
    auto updated = MixedModeMutator::Rewrite_(op, post);
    if (op->attrs.defined()) {
      CollectAttrs collect;
      const_cast<BaseAttrsNode*>(op->attrs.get())->VisitAttrs(&collect);
      for (auto p : collect.attrs) {
        if (body_attrs.find(p.first) != body_attrs.end() && p.second == body_attrs[p.first]) {
          LOG(WARNING) << "LabelOps found two call sites with different values for " << p.first
                       << " (" << p.second << " vs " << body_attrs[p.first]
                       << "). Only the first will be recorded.";
        }
        body_attrs[p.first] = p.second;
      }
    }
    return updated;
  }
};

/*! \brief Add structural hash and layout information to Function nodes. This
 * information is used later by the profiler.
 *
 * The hash and layout information is added to the attrs field of the Function.
 * The key "hash" contains the structural hash of the node. Any attributes with
 * "layout" in their name are also added to attrs (for example,
 * `attrs["src_layout"]` contains the `src_layout` attribute of the TVM op
 * corresponding to this function call).
 */
Pass LabelOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(LabelOpsMutator().Mutate(f));
      };
  return CreateFunctionPass(pass_func, 1, "LabelOps", {});
}

}  // namespace transform
}  // namespace relay
}  // namespace tvm
