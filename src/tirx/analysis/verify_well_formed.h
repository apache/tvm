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

#ifndef TVM_TIRX_ANALYSIS_VERIFY_WELL_FORMED_H_
#define TVM_TIRX_ANALYSIS_VERIFY_WELL_FORMED_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/op/region.h>

#include <utility>

#include "../ir/tir_visitor_with_path.h"
namespace tvm {
namespace tirx {
using AccessPath = ffi::reflection::AccessPath;

template <typename PathVisitor>
class UndefinedVarVerifier : public Verifier<UndefinedVarVerifier<PathVisitor>, PathVisitor> {
  using Verifier = tirx::Verifier<UndefinedVarVerifier<PathVisitor>, PathVisitor>;

 public:
  // Until templated-this arrives in C++23, the CRTP can't inject a
  // constructor into the child class.  Therefore, must explicitly add
  // the constructor.
  using Verifier::Verifier;
  using Verifier::Verify;

 private:
  using Verifier::Visit;
  void EnterDef(const Var& var, AccessPath path) override {
    {
      auto it = currently_defined_.find(var);
      auto verify = Verify(it == currently_defined_.end());
      verify << "ValueError: "
             << "TIR is ill-formed, "
             << "due to multiple nested definitions of variable " << var->name << ".";
      if (it != currently_defined_.end()) {
        verify << " It was first defined at " << it->second << ", and was re-defined at " << path;
      }
    }

    {
      auto it = previously_defined_.find(var);
      auto verify = Verify(it == previously_defined_.end());
      verify << "ValueError: "
             << "TIR is ill-formed, "
             << "due to multiple definitions of variable " << var->name << ".";
      if (it != previously_defined_.end()) {
        verify << " It was first defined at " << it->second << ", and was later re-defined at "
               << path;
      }
    }

    currently_defined_.insert({var, path});
  }

  void ExitDef(const Var& var, AccessPath path) override {
    auto active_def = currently_defined_.find(var);

    if (active_def != currently_defined_.end()) currently_defined_.erase(active_def);
    previously_defined_.insert({var, path});
  }

  void Dispatch_(const VarNode* op, AccessPath path) override {
    auto var = ffi::GetRef<Var>(op);

    auto active_def = currently_defined_.find(var);
    auto verify = Verify(active_def != currently_defined_.end());
    verify << "ValueError: "
           << "Invalid use of undefined variable " << var->name << " at " << path << ".";

    // Check if there was a previous definition, and append the
    // location to the error message if there was.  This is to aid in
    // debugging, by distinguishing between a variable that is
    // currently out-of-scope, and a variable that never had a
    // definition in the first place.
    if (auto prev_def = previously_defined_.find(var); prev_def != previously_defined_.end()) {
      verify << ".  While this variable was previously defined at " << prev_def->second
             << ", this definition is no longer in-scope.";
    }
  }

  // Variables that are defined in the currently-visited scope.
  std::unordered_map<Var, AccessPath> currently_defined_;

  // Variables that were previously defined, and are now out of scope.
  std::unordered_map<Var, AccessPath> previously_defined_;
};

/*! \brief Verify that buffers with a declaration are not used outside their declared scope.
 *
 * When a buffer is declared via one of the following sites:
 *   - TensorType-annotated Function parameters
 *   - DeclTensor statement
 *   - Dialect-specific definitions exposed by PathVisitor
 *
 * it must not appear in a TensorLoad, TensorStore, or BufferRegion outside that declaration's
 * scope.
 *
 * All buffers that appear in TensorLoad or TensorStore must have a prior declaration.
 */
template <typename PathVisitor>
class UndefinedBufferVerifier : public Verifier<UndefinedBufferVerifier<PathVisitor>, PathVisitor> {
  using Verifier = tirx::Verifier<UndefinedBufferVerifier<PathVisitor>, PathVisitor>;

 public:
  using Verifier::Verifier;
  using Verifier::Verify;

 private:
  using Verifier::Visit;

  void Visit(const Function& function, AccessPath path) override {
    Verifier::Visit(function, path);
    // Clear per-function state (buffers should not cross function boundaries).
    currently_defined_.clear();
    previously_defined_.clear();
  }

  void EnterDef(const Var& var, AccessPath path) override {
    if (auto buffer = var.as<TensorVar>()) {
      currently_defined_.insert({buffer.value(), path});
    }
  }

  void ExitDef(const Var& var, AccessPath path) override {
    if (!var->ty.as<TensorTypeNode>()) return;
    auto buffer = var.as_or_throw<TensorVar>();
    auto active_def = currently_defined_.find(buffer);
    if (active_def != currently_defined_.end()) {
      currently_defined_.erase(active_def);
    }
    previously_defined_.insert({buffer, path});
  }

  void VisitBufferUse(const TensorVar& buffer, AccessPath path) override {
    bool is_declared = currently_defined_.count(buffer);
    bool was_declared = previously_defined_.count(buffer);

    if (was_declared && !is_declared) {
      // TensorVar was previously declared but is now out of scope — always an error.
      auto prev_def = previously_defined_.find(buffer);
      Verify(false) << "TIR is ill-formed: buffer " << buffer.name() << " is used at " << path
                    << " but its declaration is no longer in-scope. "
                    << "It was declared at " << prev_def->second << ".";
    } else if (!is_declared && !was_declared) {
      // TensorVar was never declared — error.
      Verify(false) << "TIR is ill-formed: buffer " << buffer.name() << " is used at " << path
                    << " without a prior DeclTensor or other declaration.";
    }
    // TensorVar fields are visited at definition site (EnterDef), not here.
    Verifier::VisitBufferUse(buffer, path);
  }

  // Buffers defined in the currently-visited scope.
  std::unordered_map<TensorVar, AccessPath, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      currently_defined_;
  // Buffers that were previously defined and are now out of scope.
  std::unordered_map<TensorVar, AccessPath, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      previously_defined_;
};

/*! \brief Verify the asserted type of each tirx buffer load. */
template <typename PathVisitor>
class TensorLoadTypeVerifier : public Verifier<TensorLoadTypeVerifier<PathVisitor>, PathVisitor> {
  using Verifier = tirx::Verifier<TensorLoadTypeVerifier<PathVisitor>, PathVisitor>;

 public:
  using Verifier::Verifier;
  using Verifier::Verify;

 private:
  using Verifier::Visit;
  void Dispatch_(const TensorLoadNode* op, AccessPath path) override {
    auto buffer = op->source.as<TensorVar>();
    auto valid_source = Verify(buffer.has_value());
    valid_source << "TypeError: TIR TensorLoad source at " << path->Attr("source")
                 << " must be a TensorVar.";
    if (!buffer.has_value()) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    bool valid_indices = buffer.value()->shape.size() == op->indices.size();
    auto valid_rank = Verify(valid_indices);
    valid_rank << "ValueError: TIR TensorLoad at " << path << " indexes "
               << buffer.value()->shape.size() << "-dimensional buffer " << buffer.value().name()
               << " with " << op->indices.size() << " indices.";
    if (!valid_indices) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    for (size_t i = 0; i + 1 < op->indices.size(); ++i) {
      bool is_scalar = op->indices[i].ty().IsScalar();
      auto valid_index = Verify(is_scalar);
      valid_index << "TypeError: TIR TensorLoad index " << i << " at "
                  << path->Attr("indices")->ArrayItem(i)
                  << " must be scalar because only the final index may be vector-valued.";
      valid_indices = valid_indices && is_scalar;
    }
    if (!valid_indices) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    ffi::Optional<PrimType> index_ty = op->indices.empty()
                                           ? ffi::Optional<PrimType>(buffer.value()->dtype)
                                           : op->indices.back().ty().as<PrimType>();
    AccessPath final_index_path = op->indices.empty()
                                      ? path->Attr("indices")
                                      : path->Attr("indices")->ArrayItem(op->indices.size() - 1);
    auto valid_index_type = Verify(index_ty.has_value());
    valid_index_type << "TypeError: TIR TensorLoad final index at " << final_index_path
                     << " must have a primitive type.";
    if (!index_ty.has_value()) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    bool scalable_compatible = op->indices.empty() || !(buffer.value()->dtype.IsScalableVector() &&
                                                        index_ty.value().IsScalableVector());
    auto valid_scalability = Verify(scalable_compatible);
    valid_scalability << "TypeError: TIR TensorLoad at " << path
                      << " cannot combine a scalable buffer dtype with a scalable index.";
    if (!scalable_compatible) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    TensorLoad expected = MakeTensorLoad(buffer.value(), op->indices, op->span);
    ffi::Optional<PrimType> asserted_ty = op->ty.as<PrimType>();
    ffi::Optional<PrimType> expected_ty = expected->ty.as<PrimType>();
    auto valid_type = Verify(asserted_ty.has_value() && expected_ty.has_value() &&
                             asserted_ty.value() == expected_ty.value());
    valid_type << "TypeError: TIR TensorLoad at " << path << " asserts result type " << op->ty
               << ", but its source and indices imply " << expected->ty << ".";
    PathVisitor::Dispatch_(op, path);
  }
};

/*! \brief Verify that loop control belongs to a loop body in the same function. */
template <typename PathVisitor>
class LoopControlVerifier : public Verifier<LoopControlVerifier<PathVisitor>, PathVisitor> {
  using Verifier = tirx::Verifier<LoopControlVerifier<PathVisitor>, PathVisitor>;

 public:
  using Verifier::Verifier;
  using Verifier::Verify;

 private:
  using Verifier::Visit;

  void Visit(const Function& function, AccessPath path) override {
    int enclosing_depth = loop_depth_;
    loop_depth_ = 0;
    Verifier::Visit(function, path);
    loop_depth_ = enclosing_depth;
  }

  void Dispatch_(const ForNode* op, AccessPath path) override {
    Visit(op->min, path->Attr("min"));
    Visit(op->extent, path->Attr("extent"));
    Visit(op->step, path->Attr("step"));
    ++loop_depth_;
    Visit(op->body, path->Attr("body"));
    --loop_depth_;
  }

  void Dispatch_(const WhileNode* op, AccessPath path) override {
    Visit(op->condition, path->Attr("condition"));
    ++loop_depth_;
    Visit(op->body, path->Attr("body"));
    --loop_depth_;
  }

  void Dispatch_(const BreakNode* op, AccessPath path) override {
    Verify(loop_depth_ > 0) << "ValueError: break at " << path
                            << " requires an enclosing loop in the same function.";
  }

  void Dispatch_(const ContinueNode* op, AccessPath path) override {
    Verify(loop_depth_ > 0) << "ValueError: continue at " << path
                            << " requires an enclosing loop in the same function.";
  }

  int loop_depth_{0};
};

template <typename PathVisitor>
class ExecScopeVerifier : public Verifier<ExecScopeVerifier<PathVisitor>, PathVisitor> {
  using Verifier = tirx::Verifier<ExecScopeVerifier<PathVisitor>, PathVisitor>;

 public:
  using Verifier::Verifier;
  using Verifier::Verify;

 private:
  using Verifier::Visit;

  void Dispatch_(const tirx::TileOpCallNode* op, ffi::reflection::AccessPath path) override {
    static const auto& category_map = Op::GetAttrMap<tirx::TIRxOpCategory>("TIRxOpCategory");
    Verify(category_map.get(op->op, ffi::String("")) == "tile_primitive")
        << "TIRxError: TileOpCall at " << path << " has non-tile op " << op->op;
  }
};

template <typename PathVisitor>
class ScopeIdVerifier : public Verifier<ScopeIdVerifier<PathVisitor>, PathVisitor> {
  using Verifier = tirx::Verifier<ScopeIdVerifier<PathVisitor>, PathVisitor>;

 public:
  using Verifier::Verifier;
  using Verifier::Verify;

 private:
  using Verifier::Visit;

  void Visit(const Function& function, AccessPath path) override {
    Array<ScopeIdDef> enclosing_defs = std::exchange(scope_id_def_, {});
    Verifier::Visit(function, path);
    scope_id_def_ = std::move(enclosing_defs);
  }

  void Dispatch_(const RegionStmtNode* op, ffi::reflection::AccessPath path) override {
    if (op->op.same_as(tirx::device_entry_op())) {
      // Device-region marker: defs gathered from the body are verified when
      // the region exits, with launch-param sanity enforced as ``is_root``.
      size_t baseline = scope_id_def_.size();
      Verifier::Dispatch_(op, path);
      size_t total = scope_id_def_.size();
      if (total > baseline) {
        RunScopeIdVerify(path, baseline, /*is_root=*/true);
      }
      while (scope_id_def_.size() > baseline) {
        scope_id_def_.pop_back();
      }
      return;
    }
    Verifier::Dispatch_(op, path);
  }

  void RunScopeIdVerify(ffi::reflection::AccessPath path, size_t baseline, bool is_root) {
    ScopeIdDefVerifier verifier;
    Verify(verifier.Verify(scope_id_def_, ScopeIdDefVerifier::Mode::kRelaxed))
        << "TIRxError: Scope at " << path << " has invalid scope_id_def";
    if (is_root) {
      // Enforce launch-parameter sanity at the device-region root.
      auto cta_thread_it = verifier.id_set.find(ScopeBinding::kCtaThread);
      if (cta_thread_it != verifier.id_set.end() && !(*cta_thread_it).second.is_deferred()) {
        PrimExpr ext = (*cta_thread_it).second.fused_extent();
        if (const auto* imm = ext.as<IntImmNode>()) {
          Verify(imm->value > 0) << "TIRxError: kernel at " << path
                                 << " has non-positive thread count " << imm->value;
          bool needs_warp_align = verifier.id_set.count(ScopeBinding::kCtaWarp) ||
                                  verifier.id_set.count(ScopeBinding::kWarpThread) ||
                                  verifier.id_set.count(ScopeBinding::kCtaWarpgroup) ||
                                  verifier.id_set.count(ScopeBinding::kWarpgroupWarp);
          if (needs_warp_align) {
            Verify(imm->value % 32 == 0)
                << "TIRxError: kernel at " << path << " uses warp-granular bindings"
                << " but has thread count " << imm->value << " not a multiple of 32";
          }
        }
      }
    }
  }

  void Dispatch_(const ScopeIdDefStmtNode* op, ffi::reflection::AccessPath path) override {
    scope_id_def_.push_back(op->def);
    Verifier::Dispatch_(op, path);
  }

  Array<ScopeIdDef> scope_id_def_;
};

template <typename PathVisitor, typename NodeRef>
bool VerifyWellFormedCommon(const NodeRef& node, bool assert_mode) {
  return UndefinedVarVerifier<PathVisitor>::Verify(node, assert_mode) &&
         UndefinedBufferVerifier<PathVisitor>::Verify(node, assert_mode) &&
         TensorLoadTypeVerifier<PathVisitor>::Verify(node, assert_mode) &&
         LoopControlVerifier<PathVisitor>::Verify(node, assert_mode) &&
         ExecScopeVerifier<PathVisitor>::Verify(node, assert_mode) &&
         ScopeIdVerifier<PathVisitor>::Verify(node, assert_mode);
}
}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_ANALYSIS_VERIFY_WELL_FORMED_H_
