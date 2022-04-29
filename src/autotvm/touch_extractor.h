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
 * \file touch_extractor.h
 * \brief Extract feature of touch pattern of axes in lowered IR
 */

#ifndef TVM_AUTOTVM_TOUCH_EXTRACTOR_H_
#define TVM_AUTOTVM_TOUCH_EXTRACTOR_H_

#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <deque>
#include <map>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "feature_visitor.h"

namespace tvm {
namespace autotvm {

using TouchedBuffer = std::string;

// touch pattern buf[(stride * var) % mod) + other]
struct TouchPattern {
  int64_t stride{0};
  int64_t mod{-1};  // -1 for +inf

  int64_t count{1};
  int64_t reuse{1};
  int64_t thread_count{0};  // count when move thread axis into innermost
  int64_t thread_reuse{0};  // reuse ratio move thread axis into innermost
};

// all the feature of an iter var
struct ItervarFeature {
  ItervarFeature(Var var, int64_t extent, int nest, AnnotationType ann_type, int64_t topdown,
                 int counter)
      : length(extent), nest_level(nest), ann(ann_type), topdown_product(topdown), order(counter) {}
  ItervarFeature() {}

  // Axis Attributes
  int64_t length;
  int nest_level;
  AnnotationType ann;        // one-hot axis type
  int64_t topdown_product;   // accumulative product of axis length, in top-down order
  int64_t bottomup_product;  // accumulative product of axis length, in bottom-up order
  // bottomup_product = reuse * count for any touched buffer

  int order;  // used for soring axis

  // Arithmetic feature
  int add_ct{0};
  int mul_ct{0};
  int div_ct{0};

  // Memory Touch Feature
  std::unordered_map<TouchedBuffer, TouchPattern> touch_feature;
};

// extract iter vars and their touch pattern from ir
class TouchExtractor : public FeatureVisitor {
 public:
  void Analyze(const Stmt& stmt) { operator()(stmt); }

  // arithmetic stats
  void VisitExpr_(const AddNode* op) final {
    if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
      itervar_map[itervar_stack_.back()].add_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubNode* op) final {
    if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
      itervar_map[itervar_stack_.back()].add_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MulNode* op) final {
    if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
      itervar_map[itervar_stack_.back()].mul_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const DivNode* op) final {
    if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
      itervar_map[itervar_stack_.back()].div_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ModNode* op) final {
    if (op->dtype.is_float() || op->dtype.is_bfloat16()) {
      itervar_map[itervar_stack_.back()].div_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  std::unordered_map<Var, ItervarFeature, tvm::ObjectPtrHash, tvm::ObjectPtrEqual> itervar_map;

 private:
  bool EnterItervar_(Var var, int64_t length, AnnotationType ann_type);
  void ExitItervar_();
  void EnterMem_(Var buffer_var, PrimExpr index);
  void ExitMem_();

  int64_t topdown_product_{1};
  std::map<std::string, size_t> buffer_counter_;
  size_t itervar_counter_{0};
  std::deque<Var> itervar_stack_;  // use deque instead of stack for indexing
  std::deque<size_t> skip_stack_size_;

  using FeatureVisitor::VisitExpr_;
};

}  // namespace autotvm
}  // namespace tvm

#endif  // TVM_AUTOTVM_TOUCH_EXTRACTOR_H_
