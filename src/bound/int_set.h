/*!
 *  Copyright (c) 2016 by Contributors
 * \file int_set.h
 * \brief Abstraction for all integer set operations.
 */
#ifndef TVM_BOUND_INT_SET_H_
#define TVM_BOUND_INT_SET_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>

namespace tvm {
namespace bound {

// internal node container of int set.
class IntSetNode;

/*!
 * \brief Integer set class, represent a set of integers in one dimension.
 */
class IntSet : public NodeRef {
 public:
  /*! \brief constructor */
  IntSet() {}
  // constructor from not deontainer.
  explicit IntSet(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*! \return whether the set is empty */
  inline bool is_empty() const {
    return !defined();
  }
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IntSetNode* operator->() const;
  /*!
   * \param dom The domain to be created.
   * \return create integer set from existing domain
   */
  static IntSet make(Range dom);
  /*!
   * \return create integer set that represents everything
   */
  static IntSet make_all_set();
};

/*!
 * \brief Find an symbolic integer set that contains all possible values of
 *  e given the domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet Eval(Expr e,
            const std::unordered_map<IterVar, IntSet>& dom_map);
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
 * \brief Create an union set of all sets
 * \param sets The sets to be unioned
 * \return the set after union
 */
IntSet Union(const Array<IntSet>& sets);

}  // namespace bound
}  // namespace tvm

#endif  // TVM_BOUND_INT_SET_H_
