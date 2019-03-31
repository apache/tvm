/*!
 *  Copyright (c) 2019 by Contributors
 * \file const_fold.h
 * \brief Centralized location for constant folding.
 */
#ifndef TVM_ARITHMETIC_CONST_FOLD_H_
#define TVM_ARITHMETIC_CONST_FOLD_H_

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <algorithm>

namespace tvm {
namespace arith {

/*!
 * \brief Try to run binary compute with constant folding.
 *
 * \param a The left operand.
 * \param b The right operand.
 * \tparam Op The operator type.
 *
 * \note a and b Must already matched data types with each other.
 * \return nullptr if constant fold fails, otherwise return folded result.
 */
template<typename Op>
inline Expr TryConstFold(Expr a, Expr b) {
  return Expr();
}

/*!
 * \brief Try to run unary compute with constant folding.
 *
 * \param a The left operand.
 * \tparam Op The operator type.
 *
 * \note a and b Must already matched data types with each other.
 * \return nullptr if constant fold fails, otherwise return folded result.
 */
template<typename Op>
inline Expr TryConstFold(Expr a);

/*!
 * \brief Check whether type is used to represent index.
 *
 * Index types are frequently used in shape computation
 * and need to be aggressively constant-folded.
 *
 * \param type The type to represent index.
 * \return the checked result.
 */
inline bool IsIndexType(const Type& type) {
  return type.is_int() && type.lanes() == 1 &&
      (type.bits() == 32 || type.bits() == 64);
}


#define TVM_ARITH_CONST_PROPAGATION(BODY)                               \
  using ir::IntImm;                                                     \
  using ir::UIntImm;                                                    \
  using ir::FloatImm;                                                   \
  const IntImm* pa = a.as<IntImm>();                                    \
  const IntImm* pb = b.as<IntImm>();                                    \
  const FloatImm* fa = a.as<FloatImm>();                                \
  const FloatImm* fb = b.as<FloatImm>();                                \
  BODY;


#define TVM_INDEX_CONST_PROPAGATION(BODY)                               \
  using ir::IntImm;                                                     \
  using ir::UIntImm;                                                    \
  const IntImm* pa = a.as<IntImm>();                                    \
  const IntImm* pb = b.as<IntImm>();                                    \
  const Type& ta = a.type();                                            \
  const Type& tb = b.type();                                            \
  if (arith::IsIndexType(ta) && arith::IsIndexType(tb)) {               \
    BODY;                                                               \
  }                                                                     \


// specialization of constant folders.
template<>
inline Expr TryConstFold<ir::Add>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const Type& rtype = a.type();
      if (pa && pb) return IntImm::make(rtype, pa->value + pb->value);
      if (pa && pa->value == 0) return b;
      if (pb && pb->value == 0) return a;
      if (fa && fb) return FloatImm::make(rtype, fa->value + fb->value);
      if (fa && fa->value == 0) return b;
      if (fb && fb->value == 0) return a;
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Sub>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const Type& rtype = a.type();
      if (pa && pb) return IntImm::make(rtype, pa->value - pb->value);
      if (pb && pb->value == 0) return a;
      if (fa && fb) return FloatImm::make(rtype, fa->value - fb->value);
      if (fb && fb->value == 0) return a;
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Mul>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const Type& rtype = a.type();
      if (pa && pb) return IntImm::make(rtype, pa->value * pb->value);
      if (pa) {
        if (pa->value == 1) return b;
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return a;
        if (pb->value == 0) return b;
      }
      if (fa && fb) return FloatImm::make(rtype, fa->value * fb->value);
      if (fa) {
        if (fa->value == 1) return b;
        if (fa->value == 0) return a;
      }
      if (fb) {
        if (fb->value == 1) return a;
        if (fb->value == 0) return b;
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Div>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const Type& rtype = a.type();
      // due to division and mod can have different modes
      // only constant fold positive number where rule is fixed.
      if (pa && pb && pa->value >= 0 && pb->value > 0) {
        return IntImm::make(rtype, pa->value / pb->value);
      }
      if (pa) {
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return a;
        CHECK_NE(pb->value, 0) << "Divide by zero";
      }
      if (fa && fb && fb->value != 0) {
        return FloatImm::make(rtype, fa->value / fb->value);
      }
      if (fa && fa->value == 0) return a;
      if (fb) {
        if (fb->value == 1) return a;
        CHECK_NE(fb->value, 0) << "Divide by zero";
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Mod>(Expr a, Expr b) {
  TVM_INDEX_CONST_PROPAGATION({
      const Type& rtype = a.type();
      // due to division and mod can have different modes
      // only constant fold positive number where rule is fixed.
      if (pa && pb && pa->value >= 0 && pb->value > 0) {
        return IntImm::make(rtype, pa->value % pb->value);
      }
      if (pa) {
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return make_zero(rtype);
        CHECK_NE(pb->value, 0) << "Divide by zero";
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Min>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const Type& rtype = a.type();
      if (pa && pb) return IntImm::make(rtype, std::min(pa->value, pb->value));
      if (fa && fb) return FloatImm::make(rtype, std::min(fa->value, fb->value));
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Max>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const Type& rtype = a.type();
      if (pa && pb) return IntImm::make(rtype, std::max(pa->value, pb->value));
      if (fa && fb) return FloatImm::make(rtype, std::max(fa->value, fb->value));
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::GT>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImm::make(UInt(1), pa->value > pb->value);
      if (fa && fb) return UIntImm::make(UInt(1), fa->value > fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::GE>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImm::make(UInt(1), pa->value >= pb->value);
      if (fa && fb) return UIntImm::make(UInt(1), fa->value >= fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::LT>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImm::make(UInt(1), pa->value < pb->value);
      if (fa && fb) return UIntImm::make(UInt(1), fa->value < fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::LE>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImm::make(UInt(1), pa->value <= pb->value);
      if (fa && fb) return UIntImm::make(UInt(1), fa->value <= fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::EQ>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImm::make(UInt(1), pa->value == pb->value);
      if (fa && fb) return UIntImm::make(UInt(1), fa->value == fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::NE>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImm::make(UInt(1), pa->value != pb->value);
      if (fa && fb) return UIntImm::make(UInt(1), fa->value != fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::And>(Expr a, Expr b) {
  using ir::UIntImm;
  const UIntImm* pa = a.as<UIntImm>();
  const UIntImm* pb = b.as<UIntImm>();
  if (pa && pa->value) return b;
  if (pa && !pa->value) return a;
  if (pb && pb->value) return a;
  if (pb && !pb->value) return b;
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Or>(Expr a, Expr b) {
  using ir::UIntImm;
  const UIntImm* pa = a.as<UIntImm>();
  const UIntImm* pb = b.as<UIntImm>();
  if (pa && pa->value) return a;
  if (pa && !pa->value) return b;
  if (pb && pb->value) return b;
  if (pb && !pb->value) return a;
  return Expr();
}

template<>
inline Expr TryConstFold<ir::Not>(Expr a) {
  using ir::UIntImm;
  const UIntImm* pa = a.as<UIntImm>();
  if (pa) {
    return UIntImm::make(UInt(1), !(pa->value));
  }
  return Expr();
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_CONST_FOLD_H_
