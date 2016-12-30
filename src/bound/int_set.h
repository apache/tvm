/*!
 *  Copyright (c) 2016 by Contributors
 * \file int_set.h
 * \brief Abstract class for iteration integer sets.
 */
#ifndef TVM_BOUND_INT_SET_H_
#define TVM_BOUND_INT_SET_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>

namespace tvm {
namespace bound {
/*!
 * \brief abstract class of integer set for iteration sets.
 */
class IntSet {
 public:
  // constructor
  IntSet();
  // whether the set is same as range
  bool SameAs(const Range& r) const;
  // make integer set by range
  static IntSet make(Range r);
  // make integer set as a constant value
  static IntSet make(Expr value);
  // upward inference function
  // get the int set of parent given int set of outer and inner
  static void PassUp(const SplitNode* s,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     const IntSet& outer,
                     const IntSet& inner,
                     IntSet* parent);
  // upward inference function
  // get the int set of outer and inner given int set of fused.
  static void PassUp(const FuseNode* s,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     const IntSet& fused,
                     IntSet* outer,
                     IntSet* inner);
};

}  // namespace bound
}  // namespace tvm

#endif  // TVM_BOUND_INT_SET_H_
