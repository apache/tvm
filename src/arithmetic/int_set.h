/*!
 *  Copyright (c) 2016 by Contributors
 * \file int_set.h
 * \brief Abstraction for all integer set operations.
 */
#ifndef TVM_ARITHMETIC_INT_SET_H_
#define TVM_ARITHMETIC_INT_SET_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>

namespace tvm {
namespace arith {

enum SignType {
  kPositive,
  kNegative,
  kZero,
  kUnknown
};

// internal node container of int set.
struct IntSetNode;

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
  /*! \return Lower bound of the set */
  Expr min() const;
  /*! \return upper bound of the set */
  Expr max() const;
  /*! \return Whether the set represent nothing  */
  bool is_nothing() const;
  /*! \return Whether the set represent everything  */
  bool is_everything() const;
  /*! \return Whether the set is a single point */
  bool is_single_point() const;
  /*! \return Whether the set is proved to be bigger than 0 */
  bool can_prove_positive() const;
  /*! \return Whether the set is proved to be smaller than 0 */
  bool can_prove_negative() const;
  /*! \return The sign of the elements in the integer set */
  SignType sign_type() const;
  /*!
   * \brief The single point value, call only if is_single_point is true
   * \return The point value.
   */
  Expr point_value() const;
  /*!
   * \brief Try to match IntSet with range r.
   *
   * \note It is guanrateed that IntSet::range(r).match_range(r) == true
   * \return true if we can prove they are the same.
   */
  bool match_range(const Range& r) const;
  /*! \return The set contains nothing */
  static IntSet nothing();
  /*! \return The set contains everything */
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
  /*!
   * \brief Construct a set representing a interval.
   * \param min The minimum value of the interval.
   * \param max The maximum value of the interval.
   * \return constructed set.
   */
  static IntSet interval(Expr min, Expr max);
};

/*!
 * \brief Base class of all IntSet containers.
 */
struct IntSetNode : public Node {
  static constexpr const char* _type_key = "IntSet";
  TVM_DECLARE_BASE_NODE_INFO(IntSetNode, Node);
};

using ExprIntSetMap = std::unordered_map<Expr, IntSet,
      Halide::ExprHash, Halide::ExprEqual>;

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
IntSet EvalSet(Expr e,
               const std::unordered_map<const Variable*, IntSet>& dom_map);

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
IntSet EvalSet(Range r,
               const std::unordered_map<const Variable*, IntSet>& dom_map);



/*!
 * \brief Find the integer set of every sub-expression, given the
 *  domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return the map from the expression to its possible value.
 */
ExprIntSetMap EvalSetForEachSubExpr(Expr r,
    const std::unordered_map<const Variable*, IntSet>& dom_map);

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

/*!
 * \brief Deduce the bound of the target variable in a expression,
 *  give the domain of each variables. Return undefined IntSet to
 *  represent failure.
 *
 * \param v The target variable to be deduced.
 * \param cond The conditional expression.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values.
 */
IntSet DeduceBound(Expr v, Expr cond,
                   const Map<Var, IntSet>& dom_map);
IntSet DeduceBound(Expr v, Expr e,
  const std::unordered_map<const Variable*, IntSet> dom_map);


}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITHMETIC_INT_SET_H_
