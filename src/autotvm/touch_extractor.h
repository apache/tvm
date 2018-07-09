/*!
 *  Copyright (c) 2018 by Contributors
 * \file touch_extractor.h
 * \brief Extract feature of touch pattern of axes in lowered IR
 */

#ifndef TVM_AUTOTVM_TOUCH_EXTRACTOR_H_
#define TVM_AUTOTVM_TOUCH_EXTRACTOR_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/api_registry.h>
#include <stack>
#include <vector>
#include <map>
#include <string>
#include <deque>
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
  ItervarFeature(VarExpr var,
                 int64_t extent,
                 int nest,
                 AnnotationType ann_type,
                 int64_t topdown,
                 int counter)
      : length(extent), nest_level(nest), ann(ann_type), topdown_product(topdown), order(counter) {}
  ItervarFeature() {}

  // Axis Attributes
  int64_t length;
  int nest_level;
  AnnotationType ann;         // one-hot axis type
  int64_t topdown_product;    // accumulative product of axis length, in top-down order
  int64_t bottomup_product;   // accumulative product of axis length, in bottom-up order
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
  void Analyze(Stmt stmt) {
    this->Visit(stmt);
  }

  // arithmetic stats
  void Visit_(const Add *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].add_ct++;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Sub *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].add_ct++;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Mul *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].mul_ct++;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Div *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].div_ct++;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Mod *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].div_ct++;
    IRVisitor::Visit_(op);
  }

  std::unordered_map<VarExpr, ItervarFeature, tvm::ExprHash, tvm::ExprEqual> itervar_map;

 private:
  bool EnterItervar_(VarExpr var, int64_t length, AnnotationType ann_type);
  void ExitItervar_();
  void EnterMem_(VarExpr buffer_var, Expr index);
  void ExitMem_();

  int64_t topdown_product_{1};
  std::map<std::string, size_t> buffer_counter_;
  size_t itervar_counter_{0};
  std::deque<VarExpr> itervar_stack_;  // use deque instead of stack for indexing
  std::deque<size_t> skip_stack_size_;

  using IRVisitor::Visit_;
};

}  // namespace autotvm
}  // namespace tvm

#endif  // TVM_AUTOTVM_TOUCH_EXTRACTOR_H_
