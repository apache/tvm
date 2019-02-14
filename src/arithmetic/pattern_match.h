/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/arithmetic/pattern_match.h
 *
 * \brief Internal tool for expression-template based pattern matching.
 *
 * It helps to simplify pattern matching and rewrites.
 * All the patterns are generated via expression template during compile time,
 * so the result code should be as efficient as manually written pattern match code.
 *
 * The code below shows how to use the pattern matcher.
 *
 * \code
 *
 *  // max(x + z, y + z) => max(x, y) + z
 *  arith::PVar<Expr> x, y, z;
 *
 *  // The following code tries to match the declared pattern.
 *  // Match will fill the result of match into PVar if successful.
 *  // Note that z occurs twice in the pattern,
 *  // an equality check is performed to ensure each occurance of z
 *  // is equivalent to each other.
 *  if (max(x + z, y + z).Match(expr)) {
 *    // Eval evaluates a pattern with the current matched value.
 *    // The filled value is valid until the next call to Match.
 *    return (max(x, y) + z).Eval();
 *  }
 * \endcode
 *
 * \note The pattern matcher is not threadsafe,
 *       do not use the same PVar in multiple threads.
 *
 *       Please be aware that the filled value in a PVar
 *       can be overriden in the next call to Match.
 */
#ifndef TVM_ARITHMETIC_PATTERN_MATCH_H_
#define TVM_ARITHMETIC_PATTERN_MATCH_H_

#include <tvm/ir_pass.h>
#include <tuple>

namespace tvm {
namespace arith {
/*!
 * \brief Base class of all the patterns.
 *
 * There are two major member functions supported by each pattern.
 * - Match: checks if value matches the pattern.
 * - Eval: construct a new value based on matched values in PVar.
 *
 * We use curiously recurring template pattern.
 * \tparam SubType The type if the child class.
 */
template<typename SubType>
class Pattern {
 public:
  /*!
   * \brief Check if value matches the current pattern.
   *
   * This call also populates the PVars with matched value.
   * The values in PVars are valid until the next call to Match.
   *
   * \return whether value matches the pattern.
   */
  template<typename NodeType>
  bool Match(const NodeType& value) const {
    self().InitMatch_();
    return self().Match_(value);
  }
  /*! \return subtype instance of current class. */
  const SubType& self() const {
    return *static_cast<const SubType*>(this);
  }
};

/*!
 * \brief Default deep equality checker
 * \tparam T the comparison point.
 */
template<typename T>
class PEqualChecker {
 public:
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs == rhs;
  }
};

template<>
class PEqualChecker<Expr> {
 public:
  bool operator()(const Expr& lhs, const Expr& rhs) const {
    if (lhs.same_as(rhs)) return true;
    return ir::Equal(lhs, rhs);
  }
};

/*!
 * \brief Pattern variable container.
 *
 * PVar is used as a "hole" in the pattern that can be matched.
 *
 * \tparam T the type of the hole.
 *
 * \note PVar is not thread safe.
 *       Do not use the same PVar in multiple threads.
 */
template<typename T>
class PVar : public Pattern<PVar<T> > {
 public:
  void InitMatch_() const {
    filled_ = false;
  }

  bool Match_(const T& value) const {
    if (!filled_) {
      value_ = value;
      filled_ = true;
      return true;
    } else {
      return PEqualChecker<T>()(value_, value);
    }
  }

  T Eval() const {
    CHECK(filled_);
    return value_;
  }

 private:
  /*! \brief The matched value */
  mutable T value_;
  /*! \brief whether the variable has been filled */
  mutable bool filled_{false};
};

/*!
 * \brief Constant Pattern variable container.
 *
 * \tparam T the type of the hole.
 */
template<typename T>
class PConst : public Pattern<PConst<T> > {
 public:
  PConst(T value)  // NOLINT(*)
      : value_(value) {}

  void InitMatch_() const {}

  bool Match_(const T& value) const {
    return PEqualChecker<T>()(value_, value);
  }

  T Eval() const {
    return value_;
  }
 private:
  const T value_;
};

/*!
 * \brief Pattern binary expression.
 * \tparam NodeType The AST node type.
 * \tparam TA The pattern type of the first operand.
 * \tparam TB The pattern type of the second operand.
 */
template<typename NodeType, typename TA, typename TB>
class PBinaryExpr :
      public Pattern<PBinaryExpr<NodeType, TA, TB> > {
 public:
  PBinaryExpr(const TA& a, const TB& b) : a_(a), b_(b) {}

  void InitMatch_() const {
    a_.InitMatch_();
    b_.InitMatch_();
  }

  bool Match_(const NodeRef& node) const {
    if (const NodeType* ptr = node.as<NodeType>()) {
      if (!a_.Match_(ptr->a)) return false;
      if (!b_.Match_(ptr->b)) return false;
      return true;
    } else {
      return false;
    }
  }

  Expr Eval() const {
    return NodeType::make(a_.Eval(), b_.Eval());
  }

 private:
  const TA& a_;
  const TB& b_;
};


#define TVM_PATTERN_BINARY_OP(FuncName, NodeName)             \
  template<typename TA, typename TB>                          \
  inline PBinaryExpr<NodeName, TA, TB>                        \
  FuncName(const Pattern<TA>& a, const Pattern<TB>& b) {      \
    return PBinaryExpr<NodeName, TA, TB>(a.self(), b.self()); \
  }

// arithmetic expressions
TVM_PATTERN_BINARY_OP(operator+, ir::Add);
TVM_PATTERN_BINARY_OP(operator-, ir::Sub);
TVM_PATTERN_BINARY_OP(operator*, ir::Mul);
TVM_PATTERN_BINARY_OP(operator/, ir::Div);
TVM_PATTERN_BINARY_OP(operator%, ir::Mod);
TVM_PATTERN_BINARY_OP(min, ir::Min);
TVM_PATTERN_BINARY_OP(max, ir::Max);

// logical expressions
TVM_PATTERN_BINARY_OP(operator>, ir::GT);
TVM_PATTERN_BINARY_OP(operator>=, ir::GE);
TVM_PATTERN_BINARY_OP(operator<, ir::LT);
TVM_PATTERN_BINARY_OP(operator<=, ir::LE);
TVM_PATTERN_BINARY_OP(operator==, ir::EQ);
TVM_PATTERN_BINARY_OP(operator!=, ir::NE);
TVM_PATTERN_BINARY_OP(operator&&, ir::And);
TVM_PATTERN_BINARY_OP(operator||, ir::Or);

/*!
 * \brief Pattern not expression.
 * \tparam TA The pattern type of the true operand.
 */
template<typename TA>
class PNotExpr : public Pattern<PNotExpr<TA> > {
 public:
  explicit PNotExpr(const TA& value)
      : value_(value) {}

  void InitMatch_() const {
    value_.InitMatch_();
  }

  bool Match_(const NodeRef& node) const {
    if (const ir::Not* ptr = node.as<ir::Not>()) {
      if (!value_.Match_(ptr->a)) return false;
      return true;
    } else {
      return false;
    }
  }

  Expr Eval() const {
    return ir::Not::make(value_.Eval());
  }

 private:
  const TA& value_;
};

template<typename TA>
inline PNotExpr<TA> operator!(const Pattern<TA>& value) {
  return PNotExpr<TA>(value.self());
}

// select
/*!
 * \brief Pattern select expression.
 * \tparam TCond The pattern type of the condition.
 * \tparam TA The pattern type of the true operand.
 * \tparam TB The pattern type of the false operand.
 */
template<typename TCond, typename TA, typename TB>
class PSelectExpr :
      public Pattern<PSelectExpr<TCond, TA, TB> > {
 public:
  PSelectExpr(const TCond& condition,
              const TA& true_value,
              const TB& false_value)
      : condition_(condition),
        true_value_(true_value),
        false_value_(false_value) {}

  void InitMatch_() const {
    condition_.InitMatch_();
    true_value_.InitMatch_();
    false_value_.InitMatch_();
  }

  bool Match_(const NodeRef& node) const {
    if (const ir::Select* ptr = node.as<ir::Select>()) {
      if (!condition_.Match_(ptr->condition)) return false;
      if (!true_value_.Match_(ptr->true_value)) return false;
      if (!false_value_.Match_(ptr->false_value)) return false;
      return true;
    } else {
      return false;
    }
  }

  Expr Eval() const {
    return ir::Select::make(
        condition_.Eval(), true_value_.Eval(), false_value_.Eval());
  }

 private:
  const TCond& condition_;
  const TA& true_value_;
  const TB& false_value_;
};

/*!
 * \brief Construct a select pattern.
 *
 * \param condition The condition expression.
 * \param true_value The value when condition is true.
 * \param true_value The value when condition is false.
 *
 * \return The result pattern.
 *
 * \tparam TCond The pattern type of the condition.
 * \tparam TA The pattern type of the true operand.
 * \tparam TB The pattern type of the false operand.
 */
template<typename TCond, typename TA, typename TB>
inline PSelectExpr<TCond, TA, TB>
select(const Pattern<TCond>& condition,
       const Pattern<TA>& true_value,
       const Pattern<TB>& false_value) {
  return PSelectExpr<TCond, TA, TB>(
      condition.self(), true_value.self(), false_value.self());
}

/*!
 * \brief Pattern cast expression.
 * \tparam DType The Pattern type of dtype.
 * \tparam TA The pattern type of the first operand.
 */
template<typename DType, typename TA>
class PCastExpr :
      public Pattern<PCastExpr<DType, TA> > {
 public:
  PCastExpr(const DType& dtype, const TA& value)
      : dtype_(dtype), value_(value) {
  }

  void InitMatch_() const {
    dtype_.InitMatch_();
    value_.InitMatch_();
  }

  bool Match_(const NodeRef& node) const {
    if (const ir::Cast* ptr = node.as<ir::Cast>()) {
      if (!dtype_.Match_(ptr->type)) return false;
      if (!value_.Match_(ptr->value)) return false;
      return true;
    } else {
      return false;
    }
  }

  Expr Eval() const {
    return ir::Cast::make(dtype_.Eval(), value_.Eval());
  }

 private:
  const DType& dtype_;
  const TA& value_;
};

/*!
 * \brief Construct a cast pattern.
 *
 * \param dtype The target data type, can be PVar<Type> or PConst<Type>.
 * \param value The input type.
 *
 * \return The result pattern.
 *
 * \tparam DType The pattern type of type.
 * \tparam TA The pattern type of value.
 */
template<typename DType, typename TA>
inline PCastExpr<DType, TA>
cast(const Pattern<DType>& dtype, const Pattern<TA>& value) {
  return PCastExpr<DType, TA>(dtype.self(), value.self());
}

/*!
 * \brief Pattern ramp expression.
 * \tparam TBase The pattern type of the base.
 * \tparam TStride The pattern type of the stride.
 * \tparam TLanes The pattern type of the lanes.
 */
template<typename TBase, typename TStride, typename TLanes>
class PRampExpr :
      public Pattern<PRampExpr<TBase, TStride, TLanes> > {
 public:
  PRampExpr(const TBase& base,
            const TStride& stride,
            const TLanes& lanes)
      : base_(base), stride_(stride), lanes_(lanes) {
  }

  void InitMatch_() const {
    base_.InitMatch_();
    stride_.InitMatch_();
    lanes_.InitMatch_();
  }

  bool Match_(const NodeRef& node) const {
    if (const ir::Ramp* ptr = node.as<ir::Ramp>()) {
      if (!base_.Match_(ptr->base)) return false;
      if (!stride_.Match_(ptr->stride)) return false;
      if (!lanes_.Match_(ptr->lanes)) return false;
      return true;
    } else {
      return false;
    }
  }

  Expr Eval() const {
    return ir::Ramp::make(base_.Eval(), stride_.Eval(), lanes_.Eval());
  }

 private:
  const TBase& base_;
  const TStride& stride_;
  const TLanes& lanes_;
};

/*!
 * \brief Construct a ramp pattern.
 *
 * \param base The base pattern.
 * \param stride The stride pattern.
 * \param lanes The lanes pattern.
 *
 * \return The result pattern.
 *
 * \tparam TBase The pattern type of the base.
 * \tparam TStride The pattern type of the stride.
 * \tparam TLanes The pattern type of the lanes.
 */
template<typename TBase, typename TStride, typename TLanes>
inline PRampExpr<TBase, TStride, TLanes>
ramp(const Pattern<TBase>& base,
     const Pattern<TStride>& stride,
     const Pattern<TLanes>& lanes) {
  return PRampExpr<TBase, TStride, TLanes>(
      base.self(), stride.self(), lanes.self());
}

/*!
 * \brief Pattern broadcast expression.
 * \tparam TA The pattern type of the value.
 * \tparam TLanes The pattern type of the lanes.
 */
template<typename TA, typename TLanes>
class PBroadcastExpr :
      public Pattern<PBroadcastExpr<TA, TLanes> > {
 public:
  PBroadcastExpr(const TA& value,
                 const TLanes& lanes)
      : value_(value), lanes_(lanes) {
  }

  void InitMatch_() const {
    value_.InitMatch_();
    lanes_.InitMatch_();
  }

  bool Match_(const NodeRef& node) const {
    if (const ir::Broadcast* ptr = node.as<ir::Broadcast>()) {
      if (!value_.Match_(ptr->value)) return false;
      if (!lanes_.Match_(ptr->lanes)) return false;
      return true;
    } else {
      return false;
    }
  }

  Expr Eval() const {
    return ir::Broadcast::make(value_.Eval(), lanes_.Eval());
  }

 private:
  const TA& value_;
  const TLanes& lanes_;
};

/*!
 * \brief Construct a broadcast pattern.
 *
 * \param value The value pattern.
 * \param lanes The lanes pattern.
 *
 * \return The result pattern.
 *
 * \tparam TA The pattern type of the value.
 * \tparam TLanes The pattern type of the lanes.
 */
template<typename TA, typename TLanes>
inline PBroadcastExpr<TA, TLanes>
broadcast(const Pattern<TA>& value, const Pattern<TLanes>& lanes) {
  return PBroadcastExpr<TA, TLanes>(value.self(), lanes.self());
}

// internal namespace
namespace detail {
// implementation details for  CallExpr
template<bool stop, std::size_t I, typename F>
struct tuple_for_each_dispatcher {
  template<typename TTuple>
  static void run(F& f, const TTuple& tuple) { // NOLINT(*)
    f(I, std::get<I>(tuple));
    tuple_for_each_dispatcher<
      (I + 1) == std::tuple_size<TTuple>::value, (I + 1), F>
        ::run(f, tuple);
  }
};

template<std::size_t I, typename F>
struct tuple_for_each_dispatcher<true, I, F> {
  template<typename TTuple>
  static void run(F& f, const TTuple& tuple) {} // NOLINT(*)
};

template<typename F, typename TTuple>
inline void tuple_for_each(F& f, const TTuple& tuple) {  // NOLINT(*)
  tuple_for_each_dispatcher<std::tuple_size<TTuple>::value == 0, 0, F>
      ::run(f, tuple);
}

struct PCallExprInitMatchFunctor {
  template<typename T>
  void operator()(size_t i, const T& pattern) const {
    pattern.InitMatch_();
  }
};

struct PCallExprMatchFunctor {
  const ir::Call* call_;
  bool matched_{true};

  explicit PCallExprMatchFunctor(const ir::Call* call)
      : call_(call) {}

  template<typename T>
  void operator()(size_t i, const T& pattern) {
    matched_ = matched_ && pattern.Match_(call_->args[i]);
  }
};

struct PCallExprEvalArgsFunctor {
  Array<Expr> args_;

  template<typename T>
  void operator()(size_t i, const T& pattern) {
    args_.push_back(pattern.Eval());
  }
};
}  // namespace detail

/*!
 * \brief Pattern CallExpr expression.
 * \tparam Op The operator functor class.
 * \tparam TArgs The arguments.
 * \note Op functor contains the name of the function and
 *          the implementation of Eval.
 */
template<typename Op, typename ...TArgs>
class PCallExpr :
      public Pattern<PCallExpr<Op, TArgs...> > {
 public:
  explicit PCallExpr(const TArgs&... args)
      : args_(args...) {
  }

  void InitMatch_() const {
    detail::PCallExprInitMatchFunctor finit;
    detail::tuple_for_each(finit, args_);
  }

  bool Match_(const NodeRef& node) const {
    if (const ir::Call* ptr = node.as<ir::Call>()) {
      if (ptr->args.size() != sizeof...(TArgs)) return false;
      if (ptr->name != Op::kName) return false;
      detail::PCallExprMatchFunctor fmatch(ptr);
      detail::tuple_for_each(fmatch, args_);
      return fmatch.matched_;
    } else {
      return false;
    }
  }

  Expr Eval() const {
    detail::PCallExprEvalArgsFunctor feval_args;
    detail::tuple_for_each(feval_args, args_);
    return Op::Eval(feval_args.args_);
  }

 private:
  const std::tuple<const TArgs&...> args_;
};

// arithemetic intrinsics
#define TVM_PATTERN_BINARY_INTRIN(FuncName, OpName, IntrinStr)        \
  struct OpName {                                                     \
    static Expr Eval(Array<Expr> args) {                              \
      return ir::Call::make(args[0].type(), kName, args,              \
                            ir::Call::PureIntrinsic);                 \
    }                                                                 \
    static constexpr const char* kName = IntrinStr;                   \
  };                                                                  \
  template<typename TA, typename TB>                                  \
  inline PCallExpr<OpName, TA, TB>                                    \
  FuncName(const Pattern<TA>& a, const Pattern<TB>& b) {              \
    return PCallExpr<OpName, TA, TB>(a.self(), b.self());             \
  }

TVM_PATTERN_BINARY_INTRIN(operator<<, PLeftShiftOp, "shift_left");
TVM_PATTERN_BINARY_INTRIN(operator>>, PRightShiftOp, "shift_right");
TVM_PATTERN_BINARY_INTRIN(operator&, PBitwiseAndOp, "bitwise_and");
TVM_PATTERN_BINARY_INTRIN(operator|, PBitwiseOrOp, "bitwise_or");
TVM_PATTERN_BINARY_INTRIN(operator^, PBitwiseXorOp, "bitwise_xor");

// unary intrinsics
#define TVM_PATTERN_UNARY_INTRIN(FuncName, OpName, IntrinStr)         \
  struct OpName {                                                     \
    static Expr Eval(Array<Expr> args) {                              \
      return ir::Call::make(args[0].type(), kName, args,              \
                            ir::Call::PureIntrinsic);                 \
    }                                                                 \
    static constexpr const char* kName = IntrinStr;                   \
  };                                                                  \
  template<typename TA>                                               \
  inline PCallExpr<OpName, TA>                                        \
  FuncName(const Pattern<TA>& a) {                                    \
    return PCallExpr<OpName, TA>(a.self());                           \
  }

TVM_PATTERN_UNARY_INTRIN(operator~, PBitwiseNotOp, "bitwise_not");

// if_then_else
struct PIfThenElseOp {
  static Expr Eval(Array<Expr> args) {
    return ir::Call::make(
        args[1].type(), kName, args,
        ir::Call::PureIntrinsic);
  }
  static constexpr const char* kName = "tvm_if_then_else";
};

/*!
 * \brief Construct a if_then_else pattern.
 *
 * \param cond The condition expression.
 * \param true_value The value when condition is true.
 * \param true_value The value when condition is false.
 *
 * \return The result pattern.
 *
 * \tparam TCond The pattern type of the condition.
 * \tparam TA The pattern type of the true operand.
 * \tparam TB The pattern type of the false operand.
 */
template<typename TCond, typename TA, typename TB>
inline PCallExpr<PIfThenElseOp, TCond, TA, TB>
if_then_else(const Pattern<TCond>& cond,
             const Pattern<TA>& true_value,
             const Pattern<TB>& false_value) {
  return PCallExpr<PIfThenElseOp, TCond, TA, TB>(
      cond.self(), true_value.self(), false_value.self());
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_PATTERN_MATCH_H_
