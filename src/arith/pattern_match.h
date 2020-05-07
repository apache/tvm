/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
 *
 *  tvm::tir::Var tx, ty;
 *  arith::PVar<IntImm> c;
 *  arith::PVar<Var> v;
 *  // We can match integer and Var, both of which are
 *  // special case container of Expr
 *  CHECK((v * c).Match(tx * 3));
 *  CHECK_EQ(c.Eval()->value, 3);
 *  // cannot match c to ty
 *  CHECK(!(v * c).Match(tx * ty));
 *
 * \endcode
 *
 * \note The pattern matcher is not threadsafe,
 *       do not use the same PVar in multiple threads.
 *
 *       Please be aware that the filled value in a PVar
 *       can be overriden in the next call to Match.
 */
#ifndef TVM_ARITH_PATTERN_MATCH_H_
#define TVM_ARITH_PATTERN_MATCH_H_

#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tuple>
#include "const_fold.h"

namespace tvm {
namespace arith {
/*!
 * \brief Base class of all the patterns.
 *
 * There are two major member functions supported by each pattern.
 * - Match: checks if value matches the pattern.
 * - Eval: construct a new value based on matched values in PVar.
 *
 * We use curiously recurring template pattern to construct
 * expression templates.
 *
 * \tparam Derived The type of the derived class.
 */
template<typename Derived>
class Pattern {
 public:
  /*!
   * \brief Nested storage type in the expression.
   *
   *  Depending on the Derived class,
   *  Nested can be Derived (nest by value) or
   *  const Derived& (nest by reference).
   *
   *  The trick of Nested typedef originates from Eigen.
   *
   * \note We use nest by value for intermediate expressions,
   *       and nest by reference for PVars.
   */
  using Nested = Derived;
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
    derived().InitMatch_();
    return derived().Match_(value);
  }
  /*! \return Derived instance of current class. */
  const Derived& derived() const {
    return *static_cast<const Derived*>(this);
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
class PEqualChecker<PrimExpr> {
 public:
  bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
    if (lhs.same_as(rhs)) return true;
    return tir::ExprDeepEqual()(lhs, rhs);
  }
};

template<>
class PEqualChecker<IntImm> {
 public:
  bool operator()(const IntImm& lhs, const IntImm& rhs) const {
    return lhs->value == rhs->value;
  }
};

template<>
class PEqualChecker<tir::Var> {
 public:
  bool operator()(const tir::Var& lhs, const tir::Var& rhs) const {
    return lhs.same_as(rhs);
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
  // Store PVars by reference in the expression.
  using Nested = const PVar<T>&;

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

  template<typename NodeRefType,
           typename = typename std::enable_if<
             std::is_base_of<NodeRefType, T>::value>::type>
  bool Match_(const NodeRefType& value) const {
    if (const auto* ptr = value.template as<typename T::ContainerType>()) {
      return Match_(GetRef<T>(ptr));
    } else {
      return false;
    }
  }

  T Eval() const {
    CHECK(filled_);
    return value_;
  }

 protected:
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

  bool Match_(const ObjectRef& node) const {
    if (const NodeType* ptr = node.as<NodeType>()) {
      if (!a_.Match_(ptr->a)) return false;
      if (!b_.Match_(ptr->b)) return false;
      return true;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    PrimExpr lhs = a_.Eval();
    PrimExpr rhs = b_.Eval();
    PrimExpr ret = TryConstFold<NodeType>(lhs, rhs);
    if (ret.defined()) return ret;
    return NodeType::make(lhs, rhs);
  }

 private:
  typename TA::Nested a_;
  typename TB::Nested b_;
};

template<typename TA>
class PConstWithTypeLike :
      public Pattern<PConstWithTypeLike<TA> > {
 public:
  PConstWithTypeLike(const TA& ref, int64_t value)
      : ref_(ref), value_(value) {}

  void InitMatch_() const {}

  bool Match_(const ObjectRef& node) const {
    if (const tir::IntImmNode* ptr = node.as<tir::IntImmNode>()) {
      return ptr->value == value_;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    return tir::make_const(ref_.Eval().dtype(), value_);
  }

 private:
  typename TA::Nested ref_;
  int64_t value_;
};


#define TVM_PATTERN_BINARY_OP_EX(FuncName, NodeName, CheckStep)     \
  template<typename TA, typename TB>                                \
  inline PBinaryExpr<NodeName, TA, TB>                              \
  FuncName(const Pattern<TA>& a, const Pattern<TB>& b) {            \
    CheckStep;                                                      \
    return PBinaryExpr<NodeName, TA, TB>(a.derived(), b.derived()); \
  }                                                                 \
  template<typename TA>                                             \
  inline PBinaryExpr<NodeName, TA, PConstWithTypeLike<TA> >         \
  FuncName(const Pattern<TA>& a, int64_t b) {                       \
    CheckStep;                                                      \
    return FuncName(a, PConstWithTypeLike<TA>(a.derived(), b));     \
  }                                                                 \
  template<typename TA>                                             \
  inline PBinaryExpr<NodeName, PConstWithTypeLike<TA>, TA>          \
  FuncName(int64_t b, const Pattern<TA>& a) {                       \
    CheckStep;                                                      \
    return FuncName(PConstWithTypeLike<TA>(a.derived(), b), a);     \
  }

#define TVM_PATTERN_BINARY_OP(FuncName, NodeName) \
  TVM_PATTERN_BINARY_OP_EX(FuncName, NodeName, )


// raise ambiguity error for operator overload of / and %
TVM_PATTERN_BINARY_OP_EX(operator/, tir::DivNode, DivAmbiguityError(a));
TVM_PATTERN_BINARY_OP_EX(operator%, tir::ModNode, DivAmbiguityError(a));

// arithmetic expressions
TVM_PATTERN_BINARY_OP(operator+, tir::AddNode);
TVM_PATTERN_BINARY_OP(operator-, tir::SubNode);
TVM_PATTERN_BINARY_OP(operator*, tir::MulNode);
TVM_PATTERN_BINARY_OP(min, tir::MinNode);
TVM_PATTERN_BINARY_OP(max, tir::MaxNode);
TVM_PATTERN_BINARY_OP(div, tir::DivNode);
TVM_PATTERN_BINARY_OP(truncdiv, tir::DivNode);
TVM_PATTERN_BINARY_OP(truncmod, tir::ModNode);
TVM_PATTERN_BINARY_OP(floordiv, tir::FloorDivNode);
TVM_PATTERN_BINARY_OP(floormod, tir::FloorModNode);

// logical expressions
TVM_PATTERN_BINARY_OP(operator>, tir::GTNode);
TVM_PATTERN_BINARY_OP(operator>=, tir::GENode);
TVM_PATTERN_BINARY_OP(operator<, tir::LTNode);
TVM_PATTERN_BINARY_OP(operator<=, tir::LENode);
TVM_PATTERN_BINARY_OP(operator==, tir::EQNode);
TVM_PATTERN_BINARY_OP(operator!=, tir::NENode);
TVM_PATTERN_BINARY_OP(operator&&, tir::AndNode);
TVM_PATTERN_BINARY_OP(operator||, tir::OrNode);

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

  bool Match_(const ObjectRef& node) const {
    if (const tir::NotNode* ptr = node.as<tir::NotNode>()) {
      if (!value_.Match_(ptr->a)) return false;
      return true;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    return tir::NotNode::make(value_.Eval());
  }

 private:
  typename TA::Nested value_;
};

template<typename TA>
inline PNotExpr<TA> operator!(const Pattern<TA>& value) {
  return PNotExpr<TA>(value.derived());
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

  bool Match_(const ObjectRef& node) const {
    if (const tir::SelectNode* ptr = node.as<tir::SelectNode>()) {
      if (!condition_.Match_(ptr->condition)) return false;
      if (!true_value_.Match_(ptr->true_value)) return false;
      if (!false_value_.Match_(ptr->false_value)) return false;
      return true;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    return tir::SelectNode::make(
        condition_.Eval(), true_value_.Eval(), false_value_.Eval());
  }

 private:
  typename TCond::Nested condition_;
  typename TA::Nested true_value_;
  typename TB::Nested false_value_;
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
      condition.derived(), true_value.derived(), false_value.derived());
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

  bool Match_(const ObjectRef& node) const {
    if (const tir::CastNode* ptr = node.as<tir::CastNode>()) {
      if (!dtype_.Match_(ptr->dtype)) return false;
      if (!value_.Match_(ptr->value)) return false;
      return true;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    return tir::CastNode::make(dtype_.Eval(), value_.Eval());
  }

 private:
  typename DType::Nested dtype_;
  typename TA::Nested value_;
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
  return PCastExpr<DType, TA>(dtype.derived(), value.derived());
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

  bool Match_(const ObjectRef& node) const {
    if (const tir::RampNode* ptr = node.as<tir::RampNode>()) {
      if (!base_.Match_(ptr->base)) return false;
      if (!stride_.Match_(ptr->stride)) return false;
      if (!lanes_.Match_(ptr->lanes)) return false;
      return true;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    return tir::RampNode::make(base_.Eval(), stride_.Eval(), lanes_.Eval());
  }

 private:
  typename TBase::Nested base_;
  typename TStride::Nested stride_;
  typename TLanes::Nested lanes_;
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
      base.derived(), stride.derived(), lanes.derived());
}

template<typename TBase>
inline PRampExpr<TBase, PConstWithTypeLike<TBase>, PConst<int>>
ramp(const Pattern<TBase>& base,
     int stride,
     int lanes) {
  return PRampExpr<TBase, PConstWithTypeLike<TBase>, PConst<int>>(
      base.derived(),
      PConstWithTypeLike<TBase>(base.derived(), stride),
      PConst<int>(lanes));
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

  bool Match_(const ObjectRef& node) const {
    if (const tir::BroadcastNode* ptr = node.as<tir::BroadcastNode>()) {
      if (!value_.Match_(ptr->value)) return false;
      if (!lanes_.Match_(ptr->lanes)) return false;
      return true;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    return tir::BroadcastNode::make(value_.Eval(), lanes_.Eval());
  }

 private:
  typename TA::Nested value_;
  typename TLanes::Nested lanes_;
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
  return PBroadcastExpr<TA, TLanes>(value.derived(), lanes.derived());
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
  const tir::CallNode* call_;
  bool matched_{true};

  explicit PCallExprMatchFunctor(const tir::CallNode* call)
      : call_(call) {}

  template<typename T>
  void operator()(size_t i, const T& pattern) {
    matched_ = matched_ && pattern.Match_(call_->args[i]);
  }
};

struct PCallExprEvalArgsFunctor {
  Array<PrimExpr> args_;

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

  bool Match_(const ObjectRef& node) const {
    if (const tir::CallNode* ptr = node.as<tir::CallNode>()) {
      if (ptr->args.size() != sizeof...(TArgs)) return false;
      if (ptr->name != Op::kName) return false;
      detail::PCallExprMatchFunctor fmatch(ptr);
      detail::tuple_for_each(fmatch, args_);
      return fmatch.matched_;
    } else {
      return false;
    }
  }

  PrimExpr Eval() const {
    detail::PCallExprEvalArgsFunctor feval_args;
    detail::tuple_for_each(feval_args, args_);
    return Op::Eval(feval_args.args_);
  }

 private:
  std::tuple<typename TArgs::Nested...> args_;
};

// arithemetic intrinsics
#define TVM_PATTERN_BINARY_INTRIN(FuncName, OpName, IntrinStr)          \
  struct OpName {                                                       \
    static PrimExpr Eval(Array<PrimExpr> args) {                                \
      return tir::CallNode::make(args[0].dtype(), kName, args,           \
                                tir::CallNode::PureIntrinsic);           \
    }                                                                   \
    static constexpr const char* kName = IntrinStr;                     \
  };                                                                    \
  template<typename TA, typename TB>                                    \
  inline PCallExpr<OpName, TA, TB>                                      \
  FuncName(const Pattern<TA>& a, const Pattern<TB>& b) {                \
    return PCallExpr<OpName, TA, TB>(a.derived(), b.derived());         \
  }

TVM_PATTERN_BINARY_INTRIN(operator<<, PLeftShiftOp, "shift_left");
TVM_PATTERN_BINARY_INTRIN(operator>>, PRightShiftOp, "shift_right");
TVM_PATTERN_BINARY_INTRIN(operator&, PBitwiseAndOp, "bitwise_and");
TVM_PATTERN_BINARY_INTRIN(operator|, PBitwiseOrOp, "bitwise_or");
TVM_PATTERN_BINARY_INTRIN(operator^, PBitwiseXorOp, "bitwise_xor");

// unary intrinsics
#define TVM_PATTERN_UNARY_INTRIN(FuncName, OpName, IntrinStr)           \
  struct OpName {                                                       \
    static PrimExpr Eval(Array<PrimExpr> args) {                                \
      return tir::CallNode::make(args[0].dtype(), kName, args,           \
                                tir::CallNode::PureIntrinsic);           \
    }                                                                   \
    static constexpr const char* kName = IntrinStr;                     \
  };                                                                    \
  template<typename TA>                                                 \
  inline PCallExpr<OpName, TA>                                          \
  FuncName(const Pattern<TA>& a) {                                      \
    return PCallExpr<OpName, TA>(a.derived());                          \
  }

TVM_PATTERN_UNARY_INTRIN(operator~, PBitwiseNotOp, "bitwise_not");

// if_then_else
struct PIfThenElseOp {
  static PrimExpr Eval(Array<PrimExpr> args) {
    return tir::CallNode::make(
        args[1].dtype(), kName, args,
        tir::CallNode::PureIntrinsic);
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
      cond.derived(), true_value.derived(), false_value.derived());
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_PATTERN_MATCH_H_
