/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief Tiny Infix-Postfix Expr Evaluation
 * \file tiny_expr.h
 */
#ifndef TVM_RUNTIME_GRAPH_TINY_EXPR_H_
#define TVM_RUNTIME_GRAPH_TINY_EXPR_H_

#include <tvm/runtime/node_base.h>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace expr {

/*! \brief Expression node type */
enum ExprType {
  kOperand,
  kOperator,
};

/*! \brief Operator type*/
enum Op {
  kParenthese = 40,
  kPlus = 43,
  kMinus = 45,
  kMul = 42,
  kDiv = 47,
};

// Forward declaration
class ExprBase;
class Operator;
class Operand;

typedef std::shared_ptr<ExprBase> ExprPtr;
typedef std::shared_ptr<Operator> OperatorPtr;
typedef std::shared_ptr<Operand> OperandPtr;

/*! \brief cast Expr pointer to Operator pointer */
inline OperatorPtr ToOperator(ExprPtr ptr) {
  return std::dynamic_pointer_cast<Operator>(ptr);
}

/*! \brief cast Expr pointer to Operand pointer */
inline OperandPtr ToOperand(ExprPtr ptr) {
  return std::dynamic_pointer_cast<Operand>(ptr);
}

/*!
 * \brief Postfix expression base class
 */
class ExprBase {
 public:
  /*!
   * \brief Set the Value object
   * 
   * \param v value
   */
  inline void SetValue(const int32_t v) {
    value_ = v;
  }
  /*!
   * \brief Get the Value object
   * 
   * \return int32_t 
   */
  inline int32_t GetValue() const {
    return value_;
  }

  /*!
   * \brief Get type of the expression
   * 
   * \return ExprType 
   */
  virtual ExprType type() const = 0;

  /*!
   * \brief Destroy the Expr Base object
   * 
   */
  virtual ~ExprBase() {}

 protected:
  /*! \brief value */
  int32_t value_{0};
};

/*!
 * \brief Operand in expression
 * 
 */
class Operand : public ExprBase {
 public:
  /*!
   * \brief Construct a new Operand object
   * 
   * \param v value of the operand
   */
  explicit Operand(const int32_t v) {
    value_ = v;
  }

  /*!
   * \brief return Operand type
   * 
   * \return ExprType kOperand 
   */
  ExprType type() const final {
    return kOperand;
  }
};

/*!
 * \brief Operator in expression
 * 
 */
class Operator : public ExprBase {
 public:
  /*!
   * \brief Construct a new Operator object
   * 
   * \param v value in Op Enum
   */
  explicit Operator(const int32_t v) {
    CHECK(v == 40 || v == 42 || v == 43 || v == 45 || v == 47);
    value_ = v;
  }

  /*!
   * \brief Get the Prec of operators
   * 
   * \return char operator prec
   */
  inline char GetPrec() {
    auto op = static_cast<Op>(value_);
    if (op == kParenthese) return 1;
    else if (op == kPlus || op == kMinus) return 2;
    else if (op == kMul || op == kDiv) return 3;
    else
      return 0;
  }

  /*!
   * \brief Get operator type
   * 
   * \return ExprType kOperator
   */
  ExprType type() const final {
    return kOperator;
  }

  /*!
   * \brief Get operator type
   * 
   * \return Op operator type
   */
  inline Op op() const {
    return static_cast<Op>(value_);
  }
};

/*!
 * \brief Postfix expression class, parse infix expression and evaluate value
 * 
 */
class PostfixExpr {
 public:
  /*!
   * \brief create a expression element 
   * 
   * \tparam T Operator or Operand
   * \param v value of expression
   * \return ExprPtr 
   */
  template<typename T>
  static ExprPtr make(const int32_t v) {
    auto ptr = std::make_shared<T>(v);
    return std::dynamic_pointer_cast<ExprBase>(ptr);
  }

  /*!
   * \brief Parse infix expression in string to postfix in vector<ExprPtr>
   * 
   * \param str_expr infix expression
   * \param expr postfix expression of the infix expression
   * \param var_map variable map name:ExprPtr
   */
  static void ParseExpr(const std::string& str_expr,
                        std::vector<ExprPtr>& expr,
                        const std::unordered_map<std::string, ExprPtr> &var_map) {
    expr.clear();
    std::vector<ExprPtr> op_stack;
    std::string token;
    for (auto c : str_expr) {
      if (isalpha(c) || isdigit(c)) {
        token += c;
      } else {
        if (var_map.count(token)) {
          expr.emplace_back(var_map.at(token));
        } else if (token.size()) {
          auto const_dim = PostfixExpr::make<Operand>(atoi(token.c_str()));
          expr.emplace_back(std::move(const_dim));
        }
        if (c == '(') {
          auto left_par = PostfixExpr::make<Operator>(kParenthese);
          op_stack.emplace_back(std::move(left_par));
        } else if (c == ')') {
          while (ToOperator(op_stack.back())->op() != kParenthese) {
            expr.emplace_back(op_stack.back());
            op_stack.pop_back();
          }
          op_stack.pop_back();
        } else if (c == '+' || c == '-' || c == '*' || c == '/') {
          auto op = PostfixExpr::make<Operator>(c);
          while (op_stack.size() > 0 &&
             (ToOperator(op_stack.back())->GetPrec() > ToOperator(op)->GetPrec())) {
            expr.emplace_back(op_stack.back());
            op_stack.pop_back();
          }
          op_stack.emplace_back(std::move(op));
        }
        token.clear();
      }
    }
    if (token.size()) {
      if (var_map.count(token)) {
        expr.emplace_back(var_map.at(token));
      } else if (token.size()) {
        auto const_dim = PostfixExpr::make<Operand>(atoi(token.c_str()));
        expr.emplace_back(std::move(const_dim));
      }
    }
    while (op_stack.size() > 0) {
      expr.emplace_back(op_stack.back());
      op_stack.pop_back();
    }
  }

  /*!
   * \brief Evaluate value of a given postfix expression
   * 
   * \param expr postfix expression
   * \return int32_t arthic value
   */
  int32_t Eval(const std::vector<ExprPtr>& expr) {
    stack_.clear();
    for (const auto e : expr) {
      if (e->type() == kOperand) {
        stack_.push_back(e);
      } else {
        auto v = PostfixExpr::make<Operand>(0);
        auto a = stack_.back();
        stack_.pop_back();
        auto b = stack_.back();
        stack_.pop_back();
        auto op = ToOperator(e);
        if (op->op() == kPlus) {
          v->SetValue(a->GetValue() + b->GetValue());
        } else if (op->op() == kMinus) {
          v->SetValue(b->GetValue() - a->GetValue());
        } else if (op->op() == kMul) {
          v->SetValue(a->GetValue() * b->GetValue());
        } else if (op->op() == kDiv) {
          v->SetValue(b->GetValue() / a->GetValue());
        }
        stack_.push_back(v);
      }
    }
    return stack_.back()->GetValue();
  }

 private:
  std::vector<ExprPtr> stack_;
};

}  // namespace expr
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_GRAPH_TINY_EXPR_H_
