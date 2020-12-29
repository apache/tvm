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
 * \file 3rdparty/posit/posit-wrapper.cc
 * \brief Wrapper over the Stillwater Universal library for Bring Your Own Datatypes tests
 *
 * To compile TVM with this file,
 * 1. clone the Stillwater Universal repo from here `https://github.com/stillwater-sc/universal`.
 * 2. set `SET_BYODT_POSIT` ON and `UNIVERSAL_PATH` as the path to the folder containing Stillwater
 * Universal in your cmake file
 *
 * TODO(@gussmith23 @hypercubestart) Link to BYODT docs when they exist?
 */
#include <tvm/runtime/c_runtime_api.h>

#include <cstdint>

#include "universal/posit/posit.hpp"
// must go after posit.hpp
#include "universal/posit/math/exponent.hpp"
#include "universal/posit/math/hyperbolic.hpp"
#include "universal/posit/math/logarithm.hpp"
#include "universal/posit/math/sqrt.hpp"
#include "universal/posit/numeric_limits.hpp"

TVM_DLL sw::unum::posit<8, 2> Uint8ToPosit8es2(uint8_t in) {
  sw::unum::bitblock<8> bb;
  bb = static_cast<uint64_t>(in);
  return sw::unum::posit<8, 2>().set(bb);
}

extern "C" {
TVM_DLL uint8_t Posit8es2toUint8(sw::unum::posit<8, 2> in) {
  return static_cast<uint8_t>(in.get().to_ullong());
}

TVM_DLL uint8_t MinPosit8es2() {
  auto min = std::numeric_limits<sw::unum::posit<8, 2>>::lowest();
  return Posit8es2toUint8(min);
}

TVM_DLL float Posit8es2ToFloat(uint8_t in) { return Uint8ToPosit8es2(in).operator float(); }

TVM_DLL uint8_t FloatToPosit8es2(float in) {
  auto posit = sw::unum::posit<8, 2>(in);
  return Posit8es2toUint8(posit);
}

TVM_DLL uint8_t Posit8es2Add(uint8_t a, uint8_t b) {
  return Posit8es2toUint8(Uint8ToPosit8es2(a) + Uint8ToPosit8es2(b));
}

TVM_DLL uint8_t Posit8es2Sub(uint8_t a, uint8_t b) {
  return Posit8es2toUint8(Uint8ToPosit8es2(a) - Uint8ToPosit8es2(b));
}

TVM_DLL uint8_t Posit8es2Mul(uint8_t a, uint8_t b) {
  return Posit8es2toUint8(Uint8ToPosit8es2(a) * Uint8ToPosit8es2(b));
}

TVM_DLL uint8_t Posit8es2Div(uint8_t a, uint8_t b) {
  return Posit8es2toUint8(Uint8ToPosit8es2(a) / Uint8ToPosit8es2(b));
}

TVM_DLL uint8_t Posit8es2Max(uint8_t a, uint8_t b) {
  auto a_p = Uint8ToPosit8es2(a);
  auto b_p = Uint8ToPosit8es2(b);
  return Posit8es2toUint8(a_p > b_p ? a_p : b_p);
}

TVM_DLL uint8_t Posit8es2Sqrt(uint8_t a) {
  return Posit8es2toUint8(sw::unum::sqrt(Uint8ToPosit8es2(a)));
}

TVM_DLL uint8_t Posit8es2Exp(uint8_t a) {
  return Posit8es2toUint8(sw::unum::exp(Uint8ToPosit8es2(a)));
}

TVM_DLL uint8_t Posit8es2Log(uint8_t a) {
  return Posit8es2toUint8(sw::unum::log(Uint8ToPosit8es2(a)));
}

TVM_DLL uint8_t Posit8es2Sigmoid(uint8_t a) {
  auto posit_one = sw::unum::posit<8, 2>(1);
  return Posit8es2toUint8(posit_one / (sw::unum::exp(-Uint8ToPosit8es2(a)) + posit_one));
}

TVM_DLL uint8_t Posit8es2Tanh(uint8_t a) {
  return Posit8es2toUint8(sw::unum::tanh(Uint8ToPosit8es2(a)));
}
}

TVM_DLL sw::unum::posit<16, 2> Uint16ToPosit16es2(uint16_t in) {
  sw::unum::bitblock<16> bb;
  bb = static_cast<uint64_t>(in);
  return sw::unum::posit<16, 2>().set(bb);
}

extern "C" {
TVM_DLL uint16_t Posit16es2toUint16(sw::unum::posit<16, 2> in) {
  return static_cast<uint16_t>(in.get().to_ullong());
}

TVM_DLL uint8_t MinPosit16es2() {
  auto min = std::numeric_limits<sw::unum::posit<16, 2>>::lowest();
  return Posit16es2toUint16(min);
}

TVM_DLL float Posit16es2ToFloat(uint16_t in) { return Uint16ToPosit16es2(in).operator float(); }

TVM_DLL uint16_t FloatToPosit16es2(float in) {
  auto posit = sw::unum::posit<16, 2>(in);
  return Posit16es2toUint16(posit);
}

TVM_DLL uint16_t Posit16es2Add(uint16_t a, uint16_t b) {
  return Posit16es2toUint16(Uint16ToPosit16es2(a) + Uint16ToPosit16es2(b));
}

TVM_DLL uint16_t Posit16es2Sub(uint16_t a, uint16_t b) {
  return Posit16es2toUint16(Uint16ToPosit16es2(a) - Uint16ToPosit16es2(b));
}

TVM_DLL uint16_t Posit16es2Mul(uint16_t a, uint16_t b) {
  return Posit16es2toUint16(Uint16ToPosit16es2(a) * Uint16ToPosit16es2(b));
}

TVM_DLL uint16_t Posit16es2Div(uint16_t a, uint16_t b) {
  return Posit16es2toUint16(Uint16ToPosit16es2(a) / Uint16ToPosit16es2(b));
}

TVM_DLL uint16_t Posit16es2Max(uint16_t a, uint16_t b) {
  auto a_p = Uint16ToPosit16es2(a);
  auto b_p = Uint16ToPosit16es2(b);
  return Posit16es2toUint16(a_p > b_p ? a_p : b_p);
}

TVM_DLL uint16_t Posit16es2Sqrt(uint16_t a) {
  return Posit16es2toUint16(sw::unum::sqrt(Uint16ToPosit16es2(a)));
}

TVM_DLL uint16_t Posit16es2Exp(uint16_t a) {
  return Posit16es2toUint16(sw::unum::exp(Uint16ToPosit16es2(a)));
}

TVM_DLL uint16_t Posit16es2Log(uint16_t a) {
  return Posit16es2toUint16(sw::unum::log(Uint16ToPosit16es2(a)));
}

TVM_DLL uint16_t Posit16es2Sigmoid(uint16_t a) {
  auto posit_one = sw::unum::posit<16, 2>(1);
  return Posit16es2toUint16(posit_one / (sw::unum::exp(-Uint16ToPosit16es2(a)) + posit_one));
}

TVM_DLL uint16_t Posit16es2Tanh(uint16_t a) {
  return Posit16es2toUint16(sw::unum::tanh(Uint16ToPosit16es2(a)));
}
}

TVM_DLL sw::unum::posit<32, 2> Uint32ToPosit32es2(uint32_t in) {
  sw::unum::bitblock<32> bb;
  bb = static_cast<uint64_t>(in);
  return sw::unum::posit<32, 2>().set(bb);
}

extern "C" {
TVM_DLL uint32_t Posit32es2ToUint32(sw::unum::posit<32, 2> in) {
  return static_cast<uint32_t>(in.get().to_ullong());
}

TVM_DLL uint8_t MinPosit32es2() {
  auto min = std::numeric_limits<sw::unum::posit<32, 2>>::lowest();
  return Posit32es2ToUint32(min);
}

TVM_DLL float Posit32es2ToFloat(uint32_t in) { return Uint32ToPosit32es2(in).operator float(); }

TVM_DLL uint32_t FloatToPosit32es2(float in) {
  auto posit = sw::unum::posit<32, 2>(in);
  return Posit32es2ToUint32(posit);
}

TVM_DLL uint32_t Posit32es2Add(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) + Uint32ToPosit32es2(b));
}

TVM_DLL uint32_t Posit32es2Sub(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) - Uint32ToPosit32es2(b));
}

TVM_DLL uint32_t Posit32es2Mul(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) * Uint32ToPosit32es2(b));
}

TVM_DLL uint32_t Posit32es2Div(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) / Uint32ToPosit32es2(b));
}

TVM_DLL uint32_t Posit32es2Max(uint32_t a, uint32_t b) {
  auto a_p = Uint32ToPosit32es2(a);
  auto b_p = Uint32ToPosit32es2(b);
  return Posit32es2ToUint32(a_p > b_p ? a_p : b_p);
}

TVM_DLL uint32_t Posit32es2Sqrt(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::sqrt(Uint32ToPosit32es2(a)));
}

TVM_DLL uint32_t Posit32es2Exp(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::exp(Uint32ToPosit32es2(a)));
}

TVM_DLL uint32_t Posit32es2Log(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::log(Uint32ToPosit32es2(a)));
}

TVM_DLL uint32_t Posit32es2Sigmoid(uint32_t a) {
  auto posit_one = sw::unum::posit<32, 2>(1);
  return Posit32es2ToUint32(posit_one / (posit_one + sw::unum::exp(-Uint32ToPosit32es2(a))));
}

TVM_DLL uint32_t Posit32es2Tanh(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::tanh(Uint32ToPosit32es2(a)));
}
}
