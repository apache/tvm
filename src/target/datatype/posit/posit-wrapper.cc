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
#include <tvm/runtime/base.h>

#include <cstdint>

#include "universal/number/posit/posit.hpp"
#include "universal/number/posit/math/exponent.hpp"
#include "universal/number/posit/math/hyperbolic.hpp"
#include "universal/number/posit/math/logarithm.hpp"
#include "universal/number/posit/math/sqrt.hpp"
#include "universal/number/posit/numeric_limits.hpp"
#include "universal/number/posit/math/minmax.hpp"

#include <iostream>

extern "C" {

static inline uint16_t posit16_bits(sw::universal::posit<16, 2> p) {
  return static_cast<uint16_t>(p.get().to_ulong());
}
static inline sw::universal::posit<16, 2> posit16_from_bits(uint16_t b) {
  sw::universal::posit<16, 2> p;
  p.setbits(b);
  return p;
}

static inline uint32_t posit32_bits(sw::universal::posit<32, 2> p) {
  return static_cast<uint32_t>(p.get().to_ulong());
}
static inline sw::universal::posit<32, 2> posit32_from_bits(uint32_t b) {
  sw::universal::posit<32, 2> p;
  p.setbits(b);
  return p;
}

// ----- posit16 es2: uint16_t API -----
TVM_DLL uint16_t FloatToPosit16es2(float in) {
  return posit16_bits(sw::universal::posit<16, 2>(in));
}
TVM_DLL float Posit16es2ToFloat(uint16_t in) {
  return static_cast<float>(posit16_from_bits(in));
}
TVM_DLL uint16_t Posit16es2Add(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) + posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2Sub(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) - posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2Mul(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) * posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2Div(uint16_t a, uint16_t b) {
  return posit16_bits(posit16_from_bits(a) / posit16_from_bits(b));
}
TVM_DLL uint16_t Posit16es2Max(uint16_t a, uint16_t b) {
  auto pa = posit16_from_bits(a), pb = posit16_from_bits(b);
  return posit16_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint16_t Posit16es2Min(uint16_t a, uint16_t b) {
  auto pa = posit16_from_bits(a), pb = posit16_from_bits(b);
  return posit16_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint16_t Posit16es2Sqrt(uint16_t a) {
  return posit16_bits(sw::universal::sqrt(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Exp(uint16_t a) {
  return posit16_bits(sw::universal::exp(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Log(uint16_t a) {
  return posit16_bits(sw::universal::log(posit16_from_bits(a)));
}
// Cast: posit32(bits)->posit16(bits) and reverse
TVM_DLL uint16_t Posit32ToPosit16es2(uint32_t in_bits) {
  auto p32 = posit32_from_bits(in_bits);
  sw::universal::posit<16, 2> p16 = p32;
  return posit16_bits(p16);
}
TVM_DLL uint32_t Posit16ToPosit32es2(uint16_t in_bits) {
  auto p16 = posit16_from_bits(in_bits);
  sw::universal::posit<32, 2> p32 = p16;
  return posit32_bits(p32);
}

// ----- posit32 es2: uint32_t API -----
TVM_DLL uint32_t FloatToPosit32es2(float in) {
  return posit32_bits(sw::universal::posit<32, 2>(in));
}
TVM_DLL float Posit32es2ToFloat(uint32_t in) {
  return static_cast<float>(posit32_from_bits(in));
}
TVM_DLL uint32_t Posit32es2Add(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) + posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2Sub(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) - posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2Mul(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) * posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2Div(uint32_t a, uint32_t b) {
  return posit32_bits(posit32_from_bits(a) / posit32_from_bits(b));
}
TVM_DLL uint32_t Posit32es2Max(uint32_t a, uint32_t b) {
  auto pa = posit32_from_bits(a), pb = posit32_from_bits(b);
  return posit32_bits(sw::universal::max(pa, pb));
}
TVM_DLL uint32_t Posit32es2Min(uint32_t a, uint32_t b) {
  auto pa = posit32_from_bits(a), pb = posit32_from_bits(b);
  return posit32_bits(sw::universal::min(pa, pb));
}
TVM_DLL uint32_t Posit32es2Sqrt(uint32_t a) {
  return posit32_bits(sw::universal::sqrt(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Exp(uint32_t a) {
  return posit32_bits(sw::universal::exp(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Log(uint32_t a) {
  return posit32_bits(sw::universal::log(posit32_from_bits(a)));
}

} // extern "C"
