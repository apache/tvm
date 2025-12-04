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
#include "universal/number/posit/math/trigonometry.hpp"
#include "universal/number/posit/math/error_and_gamma.hpp"

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

TVM_DLL uint16_t BoolToPosit16es2(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<16, 2> p = b;
  return posit16_bits(p);
}

TVM_DLL uint32_t BoolToPosit32es2(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<32, 2> p = b;
  return posit32_bits(p);
}

TVM_DLL uint8_t Posit16es2ToBool(uint16_t in_bits) {
  auto p = posit16_from_bits(in_bits);
  return (p == sw::universal::posit<16, 2>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

TVM_DLL uint8_t Posit32es2ToBool(uint32_t in_bits) {
  auto p = posit32_from_bits(in_bits);
  return (p == sw::universal::posit<32, 2>(0.0f)) ? uint8_t{0} : uint8_t{1};
}

TVM_DLL uint16_t IntToPosit16es2(int32_t in) {
  sw::universal::posit<16, 2> p = in;
  return posit16_bits(p);
}

TVM_DLL uint32_t IntToPosit32es2(int32_t in) {
  sw::universal::posit<32, 2> p = in;
  return posit32_bits(p);
}

TVM_DLL uint8_t IntToPosit8es2(int32_t in) {
  sw::universal::posit<8, 2> p = in;
  return static_cast<uint8_t>(p.get().to_ulong());
}

TVM_DLL int32_t Posit32es2ToInt(uint32_t in_bits) {
  auto p = posit32_from_bits(in_bits);
  return static_cast<int32_t>(p);
}

TVM_DLL int32_t Posit16es2ToInt(uint16_t in_bits) {
  auto p = posit16_from_bits(in_bits);
  return static_cast<int32_t>(p);
}

TVM_DLL int32_t Posit8es2ToInt(uint8_t in_bits) {
  sw::universal::posit<8, 2> p;
  p.setbits(in_bits);
  return static_cast<int32_t>(p);
}

// Identity casts (uint <-> posit with same bitwidth): just return the bits unchanged
TVM_DLL uint8_t Uint8ToPosit8es2(uint8_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint16_t Uint16ToPosit16es2(uint16_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint32_t Uint32ToPosit32es2(uint32_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint8_t Posit8es2ToUint8(uint8_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint16_t Posit16es2ToUint16(uint16_t in) {
  return in;  // No conversion needed, same bit representation
}

TVM_DLL uint32_t Posit32es2ToUint32(uint32_t in) {
  return in;  // No conversion needed, same bit representation
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
TVM_DLL uint16_t Posit16es2Pow(uint16_t a, uint16_t b) {
  return posit16_bits(sw::universal::pow(posit16_from_bits(a), posit16_from_bits(b)));
}
TVM_DLL uint16_t Posit16es2Sigmoid(uint16_t a) {
  auto p = posit16_from_bits(a);
  return posit16_bits(sw::universal::posit<16, 2>(1.0) / (sw::universal::posit<16, 2>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint16_t Posit16es2Tanh(uint16_t a) {
  return posit16_bits(sw::universal::tanh(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Cos(uint16_t a) {
  return posit16_bits(sw::universal::cos(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Sin(uint16_t a) {
  return posit16_bits(sw::universal::sin(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Tan(uint16_t a) {
  return posit16_bits(sw::universal::tan(posit16_from_bits(a)));
}
TVM_DLL uint16_t Posit16es2Erf(uint16_t a) {
  return posit16_bits(sw::universal::erf(posit16_from_bits(a)));
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
TVM_DLL uint32_t Posit32es2Pow(uint32_t a, uint32_t b) {
  return posit32_bits(sw::universal::pow(posit32_from_bits(a), posit32_from_bits(b)));
}
TVM_DLL uint32_t Posit32es2Sigmoid(uint32_t a) {
  auto p = posit32_from_bits(a);
  return posit32_bits(sw::universal::posit<32, 2>(1.0) / (sw::universal::posit<32, 2>(1.0) + sw::universal::exp(-p)));
}
TVM_DLL uint32_t Posit32es2Tanh(uint32_t a) {
  return posit32_bits(sw::universal::tanh(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Cos(uint32_t a) {
  return posit32_bits(sw::universal::cos(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Sin(uint32_t a) {
  return posit32_bits(sw::universal::sin(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Tan(uint32_t a) {
  return posit32_bits(sw::universal::tan(posit32_from_bits(a)));
}
TVM_DLL uint32_t Posit32es2Erf(uint32_t a) {
  return posit32_bits(sw::universal::erf(posit32_from_bits(a)));
}

// ----- Comparison operations for posit16 -----
TVM_DLL uint8_t Posit16es2LT(uint16_t a, uint16_t b) {
  return posit16_from_bits(a) < posit16_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit16es2LE(uint16_t a, uint16_t b) {
  return posit16_from_bits(a) <= posit16_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit16es2GT(uint16_t a, uint16_t b) {
  return posit16_from_bits(a) > posit16_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit16es2GE(uint16_t a, uint16_t b) {
  return posit16_from_bits(a) >= posit16_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit16es2EQ(uint16_t a, uint16_t b) {
  return posit16_from_bits(a) == posit16_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit16es2NE(uint16_t a, uint16_t b) {
  return posit16_from_bits(a) != posit16_from_bits(b) ? uint8_t{1} : uint8_t{0};
}

// ----- Comparison operations for posit32 -----
TVM_DLL uint8_t Posit32es2LT(uint32_t a, uint32_t b) {
  return posit32_from_bits(a) < posit32_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit32es2LE(uint32_t a, uint32_t b) {
  return posit32_from_bits(a) <= posit32_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit32es2GT(uint32_t a, uint32_t b) {
  return posit32_from_bits(a) > posit32_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit32es2GE(uint32_t a, uint32_t b) {
  return posit32_from_bits(a) >= posit32_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit32es2EQ(uint32_t a, uint32_t b) {
  return posit32_from_bits(a) == posit32_from_bits(b) ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit32es2NE(uint32_t a, uint32_t b) {
  return posit32_from_bits(a) != posit32_from_bits(b) ? uint8_t{1} : uint8_t{0};
}

// ----- posit8 es2: uint8_t API -----
TVM_DLL uint8_t FloatToPosit8es2(float in) {
  sw::universal::posit<8, 2> p(in);
  return static_cast<uint8_t>(p.get().to_ulong());
}
TVM_DLL float Posit8es2ToFloat(uint8_t in) {
  sw::universal::posit<8, 2> p;
  p.setbits(in);
  return static_cast<float>(p);
}
TVM_DLL uint8_t Posit8es2Add(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return static_cast<uint8_t>((pa + pb).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Sub(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return static_cast<uint8_t>((pa - pb).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Mul(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return static_cast<uint8_t>((pa * pb).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Div(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return static_cast<uint8_t>((pa / pb).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Max(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return static_cast<uint8_t>(sw::universal::max(pa, pb).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Min(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return static_cast<uint8_t>(sw::universal::min(pa, pb).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Sqrt(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::sqrt(p).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Exp(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::exp(p).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Log(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::log(p).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Pow(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return static_cast<uint8_t>(sw::universal::pow(pa, pb).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Sigmoid(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>((sw::universal::posit<8, 2>(1.0) / (sw::universal::posit<8, 2>(1.0) + sw::universal::exp(-p))).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Tanh(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::tanh(p).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Cos(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::cos(p).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Sin(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::sin(p).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Tan(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::tan(p).get().to_ulong());
}
TVM_DLL uint8_t Posit8es2Erf(uint8_t a) {
  sw::universal::posit<8, 2> p;
  p.setbits(a);
  return static_cast<uint8_t>(sw::universal::erf(p).get().to_ulong());
}
TVM_DLL uint8_t BoolToPosit8es2(uint8_t in) {
  bool b = (in != 0);
  sw::universal::posit<8, 2> p = b;
  return static_cast<uint8_t>(p.get().to_ulong());
}
TVM_DLL uint8_t Posit8es2ToBool(uint8_t in_bits) {
  sw::universal::posit<8, 2> p;
  p.setbits(in_bits);
  return (p == sw::universal::posit<8, 2>(0.0f)) ? uint8_t{0} : uint8_t{1};
}
TVM_DLL uint8_t Posit8es2LT(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return pa < pb ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit8es2LE(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return pa <= pb ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit8es2GT(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return pa > pb ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit8es2GE(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return pa >= pb ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit8es2EQ(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return pa == pb ? uint8_t{1} : uint8_t{0};
}
TVM_DLL uint8_t Posit8es2NE(uint8_t a, uint8_t b) {
  sw::universal::posit<8, 2> pa, pb;
  pa.setbits(a); pb.setbits(b);
  return pa != pb ? uint8_t{1} : uint8_t{0};
}

} // extern "C"
