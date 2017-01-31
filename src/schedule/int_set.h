/*!
 *  Copyright (c) 2016 by Contributors
 * \file int_set.h
 * \brief Abstraction for all integer set operations.
 */
#ifndef TVM_SCHEDULE_INT_SET_H_
#define TVM_SCHEDULE_INT_SET_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>

namespace tvm {
namespace schedule {

// internal node container of int set.
class IntSetNode;

/*!
 * \brief Integer set class, represent a set of integers in one dimension.
 */
class IntSet : public NodeRef {
 public:
  /*! \brief constructor */
  IntSet() {}
  // constructor from not container.
  explicit IntSet(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IntSetNode* operator->() const;
  /*!
   * \brief Find a range that covers the region.
   * \param max_range The range to be covered.
   * \return The covering range.
   */
  Range cover_range(Range max_range) const;
  /*!
   * \brief find an interval that covers the set.
   * \return The covering interval set.
   */
  IntSet cover_interval() const;
  /*! \return Whether the set represent everything  */
  bool is_everything() const;
  /*! \return Whether the set is a single point */
  bool is_single_point() const;
  /*! \return Whether the set contains everything */
  static IntSet everything();
  /*!
   * \brief construct a point set.
   * \param point The point in the set.
   * \return construct a single point set
   */
  static IntSet single_point(Expr point);
  /*!
   * \brief Construct a set representing a range.
   * \param r The range
   * \return constructed set.
   */
  static IntSet range(Range r);
};

/*!
 * \brief Base class of all IntSet containers.
 */
struct IntSetNode : public Node {
};

/*!
 * \brief Find an symbolic integer set that contains all possible values of
 *  e given the domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(Expr e,
               const Map<IterVar, IntSet>& dom_map);

/*!
 * \brief Find an symbolic integer set that contains is union over
 *  all the possible conditional values in dom_map.
 *
 * \param r The initial range.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values.
 */
IntSet EvalSet(Range r,
               const Map<IterVar, IntSet>& dom_map);

/*!
 * \brief Conditional upward message passing.
 *
 * Get domain of parent, condition on domain of children.
 * Domain is represented as IntSet.
 *
 * \param s The Split relation node.
 * \param dom_map The old domain result from downward message passing.
 *    Contains the domain set if all the children are full set.
 * \param outer domain of outer iteration.
 * \param inner domain of inner iteration.
 * \param parent The result domain of parent.
 */
void PassUp(const SplitNode* s,
            const std::unordered_map<IterVar, Range>& dom_map,
            const IntSet& outer,
            const IntSet& inner,
            IntSet* parent);
/*!
 * \brief Conditional upward message passing.
 *
 * Get domain of parent, condition on domain of children.
 * Domain is represented as IntSet.
 *
 * \param s The Fuse relation node.
 * \param dom_map The old domain result from downward message passing.
 *    Contains the domain set if all the children are full set.
 * \param fused domain of fused iteration.
 * \param outer The result domain of outer iteration.
 * \param inner The result domain of inner iteration.
 */
void PassUp(const FuseNode* s,
            const std::unordered_map<IterVar, Range>& dom_map,
            const IntSet& fused,
            IntSet* outer,
            IntSet* inner);

/*!
 * \brief Conditional upward message passing.
 *
 * Get domain of parent, condition on domain of children.
 * Domain is represented as IntSet.
 *
 * \param s The Fuse relation node.
 * \param dom_map The old domain result from downward message passing.
 *    Contains the domain set if all the children are full set.
 * \param rebased domain of rebased iteration.
 * \param parent The result domain of parent iteration.
 */
void PassUp(const RebaseNode* s,
            const std::unordered_map<IterVar, Range>& dom_map,
            const IntSet& fused,
            IntSet* parent);
/*!
 * \brief Create an union set of all sets
 * \param sets The sets to be unioned
 * \return the set after union
 */
IntSet Union(const Array<IntSet>& sets);

// implementation
inline const IntSetNode* IntSet::operator->() const {
  return static_cast<const IntSetNode*>(node_.get());
}

}  // namespace schedule
}  // namespace tvm

#endif  // TVM_SCHEDULE_INT_SET_H_
