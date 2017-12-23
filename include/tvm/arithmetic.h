/*!
 *  Copyright (c) 2016 by Contributors
 * \file arithmetic.h
 * \brief Algebra and set operations and simplifications.
 */
#ifndef TVM_ARITHMETIC_H_
#define TVM_ARITHMETIC_H_

#include <vector>
#include <unordered_map>
#include <memory>
#include "./expr.h"

namespace tvm {

class Tensor;

/*! \brief namespace of arithmetic */
namespace arith {
/*!
 * \brief Sign of an expression or set.
 */
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
   * \brief construct a integer set from vector expression.
   * \param vec The vector expression, can also be single point.
   * \return The result set containing the indices in the vector.
   */
  static IntSet vector(Expr vec);
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
 * \brief Range of a linear integer function.
 *  Use to do specify the possible index values.
 *
 *  set = { coeff * x + base | x in Z }
 *
 *  When coeff != 0, it can also be written as
 *  set = { n | n % coeff == base }
 *
 *  This is useful to decide if the index is dividable by certain value.
 *  For example, if index = 0 + 4 x, then we know it can be divided by 4.
 */
struct ModularEntry {
  /*! \brief linear co-efficient */
  int coeff{1};
  /*! \brief The base */
  int base{0};

  /*! \return entry represent everything */
  static ModularEntry everything() {
    // always safe to set 0 + x, so it can be everything.
    ModularEntry e;
    e.coeff = 1;
    e.base = 0;
    return e;
  }
  /*!
   * \brief Add two modular entries together to get a new modular entry.
   * \param a The left operand.
   * \param b The right operand.
   * \return The combined modular entry.
   */
  static ModularEntry Add(const ModularEntry& a,
                          const ModularEntry& b);
};

/*!
 * \brief Base class of all IntSet containers.
 */
struct IntSetNode : public Node {
  static constexpr const char* _type_key = "IntSet";
  TVM_DECLARE_BASE_NODE_INFO(IntSetNode, Node);
};

/*!
 * \brief Detect if e can be rewritten as e = sum_{i=0}^{n-1} var[i] * coeff[i] + coeff[n]
 *  Where coeff[i] and base are invariant of var[j] for all i and j.
 *
 * \param e The expression to be detected.
 * \param vars List of variables to be used in detection.
 * \return [coeff[i]] if it is possible, empty array if it is not.
 */
Array<Expr> DetectLinearEquation(const Expr& e, const Array<Var>& vars);

/*!
 * \brief Detect if expression corresponds to clip bound of the vars
 *
 * \param e The expression to be detected.
 * \param vars List of variables to be used in detection.
 * \return concat([min_value[i], max_value[i]]), None is returned if there is no min or max value
 *          return empty if the e does not match the pattern.
 */
Array<Expr> DetectClipBound(const Expr& e, const Array<Var>& vars);

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
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
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

/*!
 * \brief Find an symbolic integer set that contains is union over
 *  all the possible conditional values in dom_map.
 *
 * \param s The initial set.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values.
 */
IntSet EvalSet(IntSet s,
               const std::unordered_map<const Variable*, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param r The range to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(Range r,
               const std::unordered_map<const Variable*, IntSet>& dom_map);

/*! \brief Map from Expr to IntSet */
using ExprIntSetMap = std::unordered_map<Expr, IntSet, ExprHash, ExprEqual>;
/*!
 * \brief Find the integer set of every sub-expression, given the
 *  domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return the map from the expression to its possible value.
 */
ExprIntSetMap EvalSetForEachSubExpr(
    Expr e,
    const std::unordered_map<const Variable*, IntSet>& dom_map);

/*!
 * \brief Create an union set of all sets
 * \param sets The sets to be unioned
 * \return the set after union
 */
IntSet Union(const Array<IntSet>& sets);

/*!
 * \brief Create an union set of all sets
 * \param sets The sets to be intersected
 * \return the set after intersected
 */
IntSet Intersect(const Array<IntSet>& sets);

/*!
 * \brief Deduce the bound of the target variable in a expression,
 *  give the domain of each variables. Return undefined IntSet to
 *  represent failure.
 *
 * \param v The target variable to be deduced.
 * \param cond The conditional expression.
 * \param hint_map The domain of variable, used to help deduce.
 * \param relax_map The domain of each variable, used to relax the domain,
 *        The deduce bound mush implies e for all value in relax_map
 * \return An integer set that can cover all the possible values.
 */
IntSet DeduceBound(Expr v, Expr cond,
                   const Map<Var, IntSet>& hint_map,
                   const Map<Var, IntSet>& relax_map);
/*!
 * \brief Same as DeduceBound with  unordered_map signature.
 *
 * \param v The target variable to be deduced.
 * \param cond The conditional expression.
 * \param hint_map The domain of variable, used to help deduce.
 * \param relax_map The domain of each variable, used to relax the domain,
 *        The deduce bound mush implies e for all value in relax_map
 * \return An integer set that can cover all the possible values.
 */
IntSet DeduceBound(Expr v, Expr cond,
                   const std::unordered_map<const Variable*, IntSet>& hint_map,
                   const std::unordered_map<const Variable*, IntSet>& relax_map);

/*!
 * \brief Infer a regular domain that covers all the calls or provides within the given statement.
 * \param body The given statement.
 * \param tensor The name of the calls or provides.
 * \param consider_calls If calls (read) are considered.
 * \param consider_provides If provides (write) are considered.
 * \return The domain that covers all the calls or provides within the given statement.
 */
Domain DomainTouched(Stmt body, const Tensor &tensor, bool consider_calls, bool consider_provides);

/*!
 * \brief Evaluate the expression with modular analysis
 * \param e The expression to be evaluated.
 * \param mod_map Map of modular statistics of known variables.
 * \return The ModularEntry covering all possible value of e.
 */
ModularEntry EvalModular(
    const Expr& e,
    const std::unordered_map<const Variable*, ModularEntry>& mod_map);

/*!
 * \brief Same as EvalModular, used by front-end.
 * \param e The expression to be evaluated.
 * \param mod_map Map of modular statistics of known variables.
 * \return A ModularSet covering all possible value of e.
 */
IntSet EvalModular(const Expr& e,
                   const Map<Var, IntSet>& mod_map);
// implementation
inline const IntSetNode* IntSet::operator->() const {
  return static_cast<const IntSetNode*>(node_.get());
}
}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_H_
