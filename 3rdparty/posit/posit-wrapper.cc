
#include <tvm/runtime/c_runtime_api.h>

#include <cstdint>

#include "posit/math/exponent.hpp"
#include "posit/math/sqrt.hpp"
#include "posit/posit.hpp"

TVM_DLL sw::unum::posit<8, 0> Uint8ToPosit8es0(uint8_t in) {
  sw::unum::bitblock<8> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<8, 0>().set(bb);
}

TVM_DLL uint8_t Posit8es0toUint8(sw::unum::posit<8, 0> in) {
  return static_cast<uint8_t>(in.get().to_ullong());
}

TVM_DLL extern "C" float Posit8es0ToFloat(uint8_t in) {
  return Uint8ToPosit8es0(in).operator float();
}

TVM_DLL extern "C" uint8_t FloatToPosit8es0(float in) {
  auto posit = sw::unum::posit<8, 0>(in);
  return Posit8es0toUint8(posit);
}

// TODO(gus) how wide should the input be?
TVM_DLL extern "C" uint8_t IntToPosit8es0(int in) {
  return Posit8es0toUint8(sw::unum::posit<8, 0>(in));
}

TVM_DLL extern "C" uint8_t Posit8es0Add(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) + Uint8ToPosit8es0(b));
}

TVM_DLL extern "C" uint8_t Posit8es0Sub(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) - Uint8ToPosit8es0(b));
}

TVM_DLL extern "C" uint8_t Posit8es0Mul(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) * Uint8ToPosit8es0(b));
}

TVM_DLL extern "C" uint8_t Posit8es0Div(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) / Uint8ToPosit8es0(b));
}

TVM_DLL extern "C" uint8_t Posit8es0Max(uint8_t a, uint8_t b) {
  auto a_p = Uint8ToPosit8es0(a);
  auto b_p = Uint8ToPosit8es0(b);
  return Posit8es0toUint8(a_p > b_p ? a_p : b_p);
}

TVM_DLL extern "C" uint8_t Posit8es0Sqrt(uint8_t a) {
  return Posit8es0toUint8(sw::unum::sqrt(Uint8ToPosit8es0(a)));
}

TVM_DLL extern "C" uint8_t Posit8es0Exp(uint8_t a) {
  return Posit8es0toUint8(sw::unum::exp(Uint8ToPosit8es0(a)));
}

TVM_DLL sw::unum::posit<16, 1> Uint16ToPosit16es1(uint16_t in) {
  sw::unum::bitblock<16> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<16, 1>().set(bb);
}

TVM_DLL uint16_t Posit16es1toUint16(sw::unum::posit<16, 1> in) {
  return static_cast<uint16_t>(in.get().to_ullong());
}

TVM_DLL extern "C" float Posit16es1ToFloat(uint16_t in) {
  return Uint16ToPosit16es1(in).operator float();
}

TVM_DLL extern "C" uint16_t FloatToPosit16es1(float in) {
  auto posit = sw::unum::posit<16, 1>(in);
  return Posit16es1toUint16(posit);
}

// TODO(gus) how wide should the input be?
TVM_DLL extern "C" uint16_t IntToPosit16es1(int in) {
  return Posit16es1toUint16(sw::unum::posit<16, 1>(in));
}

TVM_DLL extern "C" uint16_t Posit16es1Add(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) + Uint16ToPosit16es1(b));
}

TVM_DLL extern "C" uint16_t Posit16es1Sub(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) - Uint16ToPosit16es1(b));
}

TVM_DLL extern "C" uint16_t Posit16es1Mul(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) * Uint16ToPosit16es1(b));
}

TVM_DLL extern "C" uint16_t Posit16es1Div(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) / Uint16ToPosit16es1(b));
}

TVM_DLL extern "C" uint16_t Posit16es1Max(uint16_t a, uint16_t b) {
  auto a_p = Uint16ToPosit16es1(a);
  auto b_p = Uint16ToPosit16es1(b);
  return Posit16es1toUint16(a_p > b_p ? a_p : b_p);
}

TVM_DLL extern "C" uint16_t Posit16es1Sqrt(uint16_t a) {
  return Posit16es1toUint16(sw::unum::sqrt(Uint16ToPosit16es1(a)));
}

TVM_DLL extern "C" uint16_t Posit16es1Exp(uint16_t a) {
  return Posit16es1toUint16(sw::unum::exp(Uint16ToPosit16es1(a)));
}

TVM_DLL sw::unum::posit<32, 2> Uint32ToPosit32es2(uint32_t in) {
  sw::unum::bitblock<32> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<32, 2>().set(bb);
}

TVM_DLL uint32_t Posit32es2ToUint32(sw::unum::posit<32, 2> in) {
  return static_cast<uint32_t>(in.get().to_ullong());
}

TVM_DLL extern "C" float Posit32es2ToFloat(uint32_t in) {
  return Uint32ToPosit32es2(in).operator float();
}

TVM_DLL extern "C" uint32_t FloatToPosit32es2(float in) {
  auto posit = sw::unum::posit<32, 2>(in);
  return Posit32es2ToUint32(posit);
}

// TODO(gus) how wide should the input be?
TVM_DLL extern "C" uint32_t IntToPosit32es2(int in) {
  return Posit32es2ToUint32(sw::unum::posit<32, 2>(in));
}

TVM_DLL extern "C" uint32_t Posit32es2Add(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) + Uint32ToPosit32es2(b));
}

TVM_DLL extern "C" uint32_t Posit32es2Sub(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) - Uint32ToPosit32es2(b));
}

TVM_DLL extern "C" uint32_t Posit32es2Mul(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) * Uint32ToPosit32es2(b));
}

TVM_DLL extern "C" uint32_t Posit32es2Div(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) / Uint32ToPosit32es2(b));
}

TVM_DLL extern "C" uint32_t Posit32es2Max(uint32_t a, uint32_t b) {
  auto a_p = Uint32ToPosit32es2(a);
  auto b_p = Uint32ToPosit32es2(b);
  return Posit32es2ToUint32(a_p > b_p ? a_p : b_p);
}

TVM_DLL extern "C" uint32_t Posit32es2Sqrt(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::sqrt(Uint32ToPosit32es2(a)));
}

TVM_DLL extern "C" uint32_t Posit32es2Exp(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::exp(Uint32ToPosit32es2(a)));
}
