
#include "posit/posit.hpp"
#include "posit/math/sqrt.hpp"
#include "posit/math/exponent.hpp"
#include <cstdint>

sw::unum::posit<16, 1> Uint16ToPosit(uint16_t in) {
  sw::unum::bitblock<16> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<16, 1>().set(bb);
}

uint16_t PositToUint16(sw::unum::posit<16, 1> in) {
  return static_cast<uint16_t>(in.get().to_ullong());
}

extern "C" float Posit16es1ToFloat(uint16_t in) {
  return Uint16ToPosit(in).operator float();
}

extern "C" uint16_t FloatToPosit16es1(float in) {
  auto posit = sw::unum::posit<16, 1>(in);
  return PositToUint16(posit);
}

// TODO(gus) how wide should the input be?
extern "C" uint16_t IntToPosit16es1(int in) {
  return PositToUint16(sw::unum::posit<16, 1>(in));
}

extern "C" uint16_t Posit16es1Add(uint16_t a, uint16_t b) {
  return PositToUint16(Uint16ToPosit(a) + Uint16ToPosit(b));
}

extern "C" uint16_t Posit16es1Sub(uint16_t a, uint16_t b) {
  return PositToUint16(Uint16ToPosit(a) - Uint16ToPosit(b));
}

extern "C" uint16_t Posit16es1Mul(uint16_t a, uint16_t b) {
  return PositToUint16(Uint16ToPosit(a) * Uint16ToPosit(b));
}

extern "C" uint16_t Posit16es1Div(uint16_t a, uint16_t b) {
  return PositToUint16(Uint16ToPosit(a) / Uint16ToPosit(b));
}

extern "C" uint16_t Posit16es1Max(uint16_t a, uint16_t b) {
  auto a_p = Uint16ToPosit(a);
  auto b_p = Uint16ToPosit(b);
  return PositToUint16(a_p > b_p ? a_p : b_p);
}

extern "C" uint16_t Posit16es1Sqrt(uint16_t a) {
  return PositToUint16(sw::unum::sqrt(Uint16ToPosit(a)));
}

extern "C" uint16_t Posit16es1Exp(uint16_t a) {
  return PositToUint16(sw::unum::exp(Uint16ToPosit(a)));
}
