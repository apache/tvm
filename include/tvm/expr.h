/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr.h
 * \brief Defines the expressions in AST.
 */
#ifndef TVM_EXPR_H_
#define TVM_EXPR_H_

#include <type_traits>
#include "./base.h"

namespace tvm {
// forward declare Expr
class Expr;

/*!
 * \brief create a constant expression
 * \tparam T the value type
 * \param value The value to the constant.
 * \return The created expression
 */
template<typename T,
         typename = typename std::enable_if<std::is_arithmetic<T>::value>::type >
inline Expr constant(T value);

/*!
 * \brief a expression type, holds a ref to root of an AST
 */
class Expr : public NodeRef {
 public:
  /*! \brief default constructor */
  Expr() = default;
  /*!
   * \brief copy constructor
   * \param other the input
   */
  Expr(const Expr& other) = default;
  /*!
   * \brief move constructor
   * \param other the input
   */
  Expr(Expr&& other) = default;
  /*!
   * \brief assign operator.
   * \param other the input.
   * \return reference to self
   */
  Expr& operator=(const Expr& other) = default;
  /*!
   * \brief assign move operator.
   * \param other the input.
   * \return reference to self
   */
  Expr& operator=(Expr&& other) = default;
  /*!
   * \brief constructor from constant value
   * \param value the constant value
   * \tparam T The constant type
   */
  template<typename T,
           typename = typename std::enable_if<std::is_arithmetic<T>::value>::type >
  Expr(T value) {  // NOLINT(*)
    *this = std::move(constant<T>(value));
  }
  /*!
   * \brief constructor from node pointer
   * \param nptr Another node shared pointer
   */
  explicit Expr(std::shared_ptr<Node>&& nptr) : NodeRef(std::move(nptr)) {
    CHECK(node_.get() != nullptr);
  }
  /*! \return the expression type of the expression  */
  inline DataType dtype() const;
  // print the expression.
  friend std::ostream& operator<<(std::ostream &os, const Expr& e) {  // NOLINT(*)
    e.Print(os);
    return os;
  }

 private:
  // print the expression
  void Print(std::ostream& os) const;  // NOLINT(*)
};

/*! \brief Variable class */
class Var : public Expr {
 public:
  Var(std::string name="", DataType dtype=kInt32);  // NOLINT(*)
};

Expr IntConstant(int64_t value);
Expr FloatConstant(double value);

/*! \brief base of expression node */
class ExprNode : public Node {
 public:
  /*! \brief type of data stored in expression */
  DataType dtype_{kUnknown};
};

// inline implementations
inline DataType Expr::dtype() const {
  return static_cast<const ExprNode*>(node_.get())->dtype_;
}
template<typename T,
         typename = typename std::enable_if<std::is_arithmetic<T>::value>::type >
inline Expr constant(T value) {
  if (std::is_integral<T>::value) {
    return IntConstant(static_cast<int64_t>(value));
  } else {
    return FloatConstant(static_cast<double>(value));
  }
}

}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::Expr> {
  std::size_t operator()(const ::tvm::NodeRef& k) const {
    return k.hash();
  }
};
}  // namespace std
#endif  // TVM_EXPR_H_
