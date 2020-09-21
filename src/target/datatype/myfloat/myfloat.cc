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
 * \file 3rdparty/byodt/my-custom-datatype.cc
 * \brief Example Custom Datatype with the Bring Your Own Datatypes (BYODT) framework.
 * This is a toy example that under the hood simulates floats.
 *
 * Users interested in using the BYODT framework can use this file as a template.
 *
 * TODO(@gussmith23 @hypercubestart) Link to BYODT docs when they exist?
 */
#include <tvm/runtime/c_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>

// Custom datatypes are stored as bits in a uint of the appropriate bit length.
// Thus, when TVM calls these C functions,
// the arguments of are uints that need to reinterpreted as your custom datatype.
//
// When returning, your custom datatype needs to be re-wrapped into a uint,
// which can be thought of as just a wrapper for the raw bits that represent your custom datatype.
template <class T>
TVM_DLL T Uint32ToCustom32(uint32_t in) {
  // This is a helper function to interpret the uint as your custom dataype.
  // The following line should be replaced with the appropriate function
  // that interprets the bits in `in` and returns your custom datatype
  T* custom = reinterpret_cast<T*>(&in);
  return *custom;
}

template <class T>
TVM_DLL uint32_t Custom32ToUint32(T in) {
  // This is a helper function to wrap your custom datatype in a uint.
  // the following line should be replaced with the appropriate function
  // that converts your custom datatype into a uint
  uint32_t* bits = reinterpret_cast<uint32_t*>(&in);
  return *bits;
}

extern "C" {
TVM_DLL uint32_t MinCustom32() {
  // return minimum representable value
  float min = std::numeric_limits<float>::lowest();
  return Custom32ToUint32<float>(min);
}

TVM_DLL float Custom32ToFloat(uint32_t in) {
  // cast from custom datatype to float
  float custom_datatype = Uint32ToCustom32<float>(in);
  // our custom datatype is float, so the following redundant cast to float
  // is to remind users to cast their own custom datatype to float
  return static_cast<float>(custom_datatype);
}

TVM_DLL uint32_t FloatToCustom32(float in) {
  // cast from float to custom datatype
  return Custom32ToUint32<float>(in);
}

TVM_DLL uint32_t Custom32Add(uint32_t a, uint32_t b) {
  // add operation
  float acustom = Uint32ToCustom32<float>(a);
  float bcustom = Uint32ToCustom32<float>(b);
  return Custom32ToUint32<float>(acustom + bcustom);
}

TVM_DLL uint32_t Custom32Sub(uint32_t a, uint32_t b) {
  // subtract
  float acustom = Uint32ToCustom32<float>(a);
  float bcustom = Uint32ToCustom32<float>(b);
  return Custom32ToUint32<float>(acustom - bcustom);
}

TVM_DLL uint32_t Custom32Mul(uint32_t a, uint32_t b) {
  // multiply
  float acustom = Uint32ToCustom32<float>(a);
  float bcustom = Uint32ToCustom32<float>(b);
  return Custom32ToUint32<float>(acustom * bcustom);
}

TVM_DLL uint32_t Custom32Div(uint32_t a, uint32_t b) {
  // divide
  float acustom = Uint32ToCustom32<float>(a);
  float bcustom = Uint32ToCustom32<float>(b);
  return Custom32ToUint32<float>(acustom / bcustom);
}

TVM_DLL uint32_t Custom32Max(uint32_t a, uint32_t b) {
  // max
  float acustom = Uint32ToCustom32<float>(a);
  float bcustom = Uint32ToCustom32<float>(b);
  return Custom32ToUint32<float>(acustom > bcustom ? acustom : bcustom);
}

TVM_DLL uint32_t Custom32Sqrt(uint32_t a) {
  // sqrt
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(sqrt(acustom));
}

TVM_DLL uint32_t Custom32Exp(uint32_t a) {
  // exponential
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(exp(acustom));
}

TVM_DLL uint32_t Custom32Log(uint32_t a) {
  // log
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(log(acustom));
}

TVM_DLL uint32_t Custom32Sigmoid(uint32_t a) {
  // sigmoid
  float acustom = Uint32ToCustom32<float>(a);
  float one = 1.0f;
  return Custom32ToUint32<float>(one / (one + exp(-acustom)));
}

TVM_DLL uint32_t Custom32Tanh(uint32_t a) {
  // tanh
  float acustom = Uint32ToCustom32<float>(a);
  return Custom32ToUint32<float>(tanh(acustom));
}
}
