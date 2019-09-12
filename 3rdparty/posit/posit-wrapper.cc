
#include "posit/posit.hpp"
#include "posit/math/sqrt.hpp"
#include "posit/math/exponent.hpp"
#include <cstdint>

sw::unum::posit<32, 2> Uint32ToPosit(uint32_t in) {
  sw::unum::bitblock<32> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<32, 2>().set(bb);
}

uint32_t PositToUint32(sw::unum::posit<32, 2> in) {
  return static_cast<uint32_t>(in.get().to_ullong());
}

extern "C" float Posit32es2ToFloat(uint32_t in) {
  return Uint32ToPosit(in).operator float();
}

extern "C" uint32_t FloatToPosit32es2(float in) {
  auto posit = sw::unum::posit<32, 2>(in);
  return PositToUint32(posit);
}

// TODO(gus) how wide should the input be?
extern "C" uint32_t IntToPosit32es2(int in) {
  return PositToUint32(sw::unum::posit<32, 2>(in));
}

extern "C" uint32_t Posit32es2Add(uint32_t a, uint32_t b) {
  return PositToUint32(Uint32ToPosit(a) + Uint32ToPosit(b));
}

extern "C" uint32_t Posit32es2Sub(uint32_t a, uint32_t b) {
  return PositToUint32(Uint32ToPosit(a) - Uint32ToPosit(b));
}

extern "C" uint32_t Posit32es2Mul(uint32_t a, uint32_t b) {
  return PositToUint32(Uint32ToPosit(a) * Uint32ToPosit(b));
}

extern "C" uint32_t Posit32es2Div(uint32_t a, uint32_t b) {
  return PositToUint32(Uint32ToPosit(a) / Uint32ToPosit(b));
}

extern "C" uint32_t Posit32es2Max(uint32_t a, uint32_t b) {
  auto a_p = Uint32ToPosit(a);
  auto b_p = Uint32ToPosit(b);
  return PositToUint32(a_p > b_p ? a_p : b_p);
}

extern "C" uint32_t Posit32es2Sqrt(uint32_t a) {
  return PositToUint32(sw::unum::sqrt(Uint32ToPosit(a)));
}

extern "C" uint32_t Posit32es2Exp(uint32_t a) {
  return PositToUint32(sw::unum::exp(Uint32ToPosit(a)));
}
