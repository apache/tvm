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
 * \file feature.cc
 * \brief Detect features used in Expr/Module
 */
#include <tvm/ir/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>

#include "../transforms/pass_utils.h"

namespace tvm {
namespace relay {

FeatureSet DetectFeature(const Expr& expr) {
  if (!expr.defined()) {
    return FeatureSet::No();
  }
  struct FeatureDetector : ExprVisitor {
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visited_;
    FeatureSet fs = FeatureSet::No();

    void VisitExpr(const Expr& expr) final {
      if (visited_.count(expr) == 0) {
        visited_.insert(expr);
        ExprVisitor::VisitExpr(expr);
      } else {
        if (!IsAtomic(expr)) {
          fs += fGraph;
        }
      }
    }
#define DETECT_CONSTRUCT(CONSTRUCT_NAME, STMT) \
  void VisitExpr_(const CONSTRUCT_NAME##Node* op) final { STMT fs += f##CONSTRUCT_NAME; }
#define DETECT_DEFAULT_CONSTRUCT(CONSTRUCT_NAME) \
  DETECT_CONSTRUCT(CONSTRUCT_NAME, { ExprVisitor::VisitExpr_(op); })
    DETECT_DEFAULT_CONSTRUCT(Var)
    DETECT_DEFAULT_CONSTRUCT(GlobalVar)
    DETECT_DEFAULT_CONSTRUCT(Constant)
    DETECT_DEFAULT_CONSTRUCT(Tuple)
    DETECT_DEFAULT_CONSTRUCT(TupleGetItem)
    DETECT_CONSTRUCT(Function, {
      if (!op->HasNonzeroAttr(attr::kPrimitive)) {
        ExprVisitor::VisitExpr_(op);
      }
    })
    DETECT_DEFAULT_CONSTRUCT(Op)
    DETECT_DEFAULT_CONSTRUCT(Call)
    DETECT_CONSTRUCT(Let, {
      for (const Var& v : FreeVars(op->value)) {
        if (op->var == v) {
          fs += fLetRec;
        }
      }
      ExprVisitor::VisitExpr_(op);
    })
    DETECT_DEFAULT_CONSTRUCT(If)
    DETECT_DEFAULT_CONSTRUCT(RefCreate)
    DETECT_DEFAULT_CONSTRUCT(RefRead)
    DETECT_DEFAULT_CONSTRUCT(RefWrite)
    DETECT_DEFAULT_CONSTRUCT(Constructor)
    DETECT_DEFAULT_CONSTRUCT(Match)
#undef DETECT_DEFAULT_CONSTRUCT
  } fd;
  fd(expr);
  return fd.fs;
}

std::string FeatureSet::ToString() const {
  std::string ret;
  ret += "[";
  size_t detected = 0;
#define DETECT_FEATURE(FEATURE_NAME) \
  ++detected;                        \
  if (bs_[FEATURE_NAME]) {           \
    ret += #FEATURE_NAME;            \
    ret += ", ";                     \
  }
  DETECT_FEATURE(fVar);
  DETECT_FEATURE(fGlobalVar);
  DETECT_FEATURE(fConstant);
  DETECT_FEATURE(fTuple);
  DETECT_FEATURE(fTupleGetItem);
  DETECT_FEATURE(fFunction);
  DETECT_FEATURE(fOp);
  DETECT_FEATURE(fCall);
  DETECT_FEATURE(fLet);
  DETECT_FEATURE(fIf);
  DETECT_FEATURE(fRefCreate);
  DETECT_FEATURE(fRefRead);
  DETECT_FEATURE(fRefWrite);
  DETECT_FEATURE(fConstructor);
  DETECT_FEATURE(fMatch);
  DETECT_FEATURE(fGraph);
  DETECT_FEATURE(fLetRec);
#undef DETECT_FEATURE
  ICHECK(detected == feature_count) << "some feature not printed";
  ret += "]";
  return ret;
}

FeatureSet DetectFeature(const IRModule& mod) {
  FeatureSet fs = FeatureSet::No();
  for (const auto& f : mod->functions) {
    fs += DetectFeature(f.second);
  }
  return fs;
}

Array<Integer> PyDetectFeature(const Expr& expr, const Optional<IRModule>& mod) {
  FeatureSet fs = DetectFeature(expr);
  if (mod.defined()) {
    fs = fs + DetectFeature(mod.value());
  }
  return static_cast<Array<Integer>>(fs);
}

TVM_REGISTER_GLOBAL("relay.analysis.detect_feature").set_body_typed(PyDetectFeature);

void CheckFeature(const Expr& expr, const FeatureSet& fs) {
  auto dfs = DetectFeature(expr);
  ICHECK(dfs.is_subset_of(fs)) << AsText(expr, false)
                               << "\nhas unsupported feature: " << (dfs - fs).ToString();
}

void CheckFeature(const IRModule& mod, const FeatureSet& fs) {
  for (const auto& f : mod->functions) {
    CheckFeature(f.second, fs);
  }
}

}  // namespace relay
}  // namespace tvm
