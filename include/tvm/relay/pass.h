/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/pass.h
 * \brief The set of Relay passes written in C++.
 */
#ifndef TVM_RELAY_PASS_H_
#define TVM_RELAY_PASS_H_

#include <tvm/lowered_func.h>
#include <tvm/relay/environment.h>
#include <tvm/relay/expr.h>
#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Infer the type of an expression.
 *
 * The result of type checking is a new expression with unambigous
 * type information filled in, as well as it's checked type field
 * populated with the result type.
 *
 * \param expr The expression to type check.
 * \param env The environment used for referencing global functions, can be
 * None.
 *
 * \return A type checked expression with its checked_type field populated.
 */
Expr InferType(const Expr& expr, const Environment& env);
/*!
 * \brief Infer the type of a function as if it is mapped to var in the env.
 *
 * \param f the function.
 * \param env The environment used for referencing global functions.
 * \param var The global variable corresponding to the function.
 *
 * \return A type checked Function with its checked_type field populated.
 * \note this function mutates env and is not thread-safe.
 */
Function InferType(const Function& f, const Environment& env,
                   const GlobalVar& var);

/*!
 * \brief Check that types are well kinded by applying "kinding rules".
 *
 * This pass ensures we do not do things that violate the design of the
 * type system when writing down types.
 *
 * For example tensors are not allowed to contain functions in Relay.
 *
 * We check this by ensuring the `dtype` field of a Tensor always contains
 * a data type such as `int`, `float`, `uint`.
 *
 * \param t The type to check.
 * \param env The global environment.
 *
 * \return true if the rules are satisified otherwise false
 */
bool KindCheck(const Type& t, const Environment& env);

/*! \brief Compare two expressions for structural equivalence.
 *
 * This comparison operator respects scoping and compares
 * expressions without regard to variable choice.
 *
 * For example: `let x = 1 in x` is equal to `let y = 1 in y`.
 *
 *   See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
 *   for more details.
 *
 *   \param e1 The left hand expression.
 *   \param e2 The right hand expression.
 *
 *   \return true if equal, otherwise false
 */
bool AlphaEqual(const Expr& e1, const Expr& e2);

/*! \brief Compare two types for structural equivalence.
 *
 * This comparison operator respects scoping and compares
 * expressions without regard to variable choice.
 *
 * For example: `forall s, Tensor[f32, s]` is equal to
 * `forall w, Tensor[f32, w]`.
 *
 * See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
 * for more details.
 *
 * \param t1 The left hand type.
 * \param t2 The right hand type.
 *
 * \return true if equal, otherwise false
 */
bool AlphaEqual(const Type& t1, const Type& t2);

/*! \brief Check that each Var is only bound once.
 *
 * For example, the expression `let x = 1 in let x = 2 in 3` bound x twice.
 *
 * `let f = (\x -> x) in let g = (\x -> x + 1) in f(g(2))` also bound x twice,
 * although x is not shadowed.
 *
 * \param e the expression to check.
 *
 * \return true iff all Var in e is bound at most once.
 */
bool WellFormed(const Expr& e);

/*! \brief Get free Vars from expr in PostDFS order.
 *
 * Free variables are variables that are not bound by a let or a function
 * parameter in the context.
 *
 * \param e the expression.
 *
 * \return the set of free variable.
 */
tvm::Array<Var> FreeVariables(const Expr& e);

/*! \brief Get free type parameters from expression e.
 *
 * Free variables are variables that are not bound by a
 * let or a function parameter in the context.
 *
 * \param expr the expression.
 *
 * \return List of free vars, in the PostDFS order visited by expr.
 */
tvm::Array<Var> FreeVars(const Expr& expr);

/*! \brief Get free TypeVars from expression expr.
 *
 * Free type parameters are type parameters that are not bound by a function
 * type in the context.
 *
 * \param expr the expression.
 *
 * \return List of free vars, in the PostDFS order visited by expr.
 */
tvm::Array<TypeVar> FreeTypeVars(const Expr& expr);

/*! \brief Remove expressions which does not effect the program result.
 *
 * It will remove let bindings which are not referenced, and branches that will
 * not be entered.
 *
 * For example, this pass should turn `let a = 1 in 2` into `2`, as the value of
 * the expression does not depend on a. Another example is `if (true) then 1
 * else 2` will be optimized into 1.
 *
 * \param e the expression to optimize.
 *
 * \return the optimized expression.
 */
Expr DeadCodeElimination(const Expr& e);

/*! \brief Hash a Relay type.
 *
 * Implements structural hashing of a Relay type.
 *
 *  \param type the type to hash.
 *
 *  \return the hash value.
 */
size_t StructuralHash(const Type& type);

/*! \brief Hash a Relay expression.
 *
 * Implements structural hashing of a Relay expression.
 *
 * \param expr the expression to hash.
 *
 * \return the hash value.
 */
size_t StructuralHash(const Expr& expr);

/*! \brief The hash struct for expressions. */
struct ExprHash {
  size_t operator()(const Expr& a) const { return StructuralHash(a); }
};

/*! \brief The equal comparator for expressions. */
struct ExprEqual {
  bool operator()(const Expr& a, const Expr& b) const {
    return AlphaEqual(a, b);
  }
};

/*! \brief A lowered Relay operation.
 *
 * A lowered operation is a pair containing the "primitive" function used
 * to produce the lowered function as well as the lowered function itself.
 */
class LoweredOp;
/*! \brief Call container. */
class LoweredOpNode : public Node {
 public:
  /*!
   * \brief The primitive function to be lowered.
   *
   * A primitive function consists only of calls to relay::Op which
   * can be fused.
   */
  Function func;

  /*!
   * \brief The lowered function.
   */
  LoweredFunc lowered_func;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("lowered_func", &lowered_func);
  }

  TVM_DLL static LoweredOp make(
      Function func,
      LoweredFunc lowered_func);

  static constexpr const char* _type_key = "relay.LoweredOp";
  TVM_DECLARE_NODE_TYPE_INFO(LoweredOpNode, Node);
};

RELAY_DEFINE_NODE_REF(LoweredOp, LoweredOpNode, NodeRef);

/*!
 * \brief Lower the operations contained in a Relay expression.
 *
 * The lowering pass will only lower functions marked as primitive,
 * the FuseOps pass will provide this behavior, if run before LowerOps.
 *
 * \note This will do a reachability analysis and lower all definitions
 * reachable from the provided expression.
 *
 * \param env  The environment.
 * \param expr The expression with operations to be lowered.
 * \param target The target to lower the functions to.
 *
 * \return The set of lowered operations.
 */
Array<LoweredOp> LowerOps(const Environment& env, const Expr& expr,
                          const std::string& target = "llvm");

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_H_
