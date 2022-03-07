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
 * \file ptx_mma.cc
 */

#include "ptx_mma.h"

namespace tvm {
namespace codegen {

std::string ReplaceMMAArgument(std::string asm_code, const std::string& original,
                               const std::string& new_arg) {
  size_t len = original.size();
  size_t new_len = new_arg.size();
  size_t pos = asm_code.find(original);
  while (pos != std::string::npos) {
    asm_code = asm_code.replace(pos, len, new_arg);
    pos = asm_code.find(original, pos + new_len);
  }
  return asm_code;
}

std::string PrintMMAm8n8k4Assembly(const std::string& A_layout, const std::string& B_layout,
                                   const std::string& A_dtype, const std::string& B_dtype,
                                   const std::string& C_dtype, const std::string& a_ref,
                                   const std::string& a_bias, const std::string& b_ref,
                                   const std::string& b_bias, const std::string& c_ref,
                                   const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "fp16") && (B_dtype == "fp16")) ||
         ((A_dtype == "fp64") && (B_dtype == "fp64")));
  ICHECK(saturate == false) << "Saturate is not allowed for m8n8k4 mma.";
  if ((A_dtype == "fp16") && (B_dtype == "fp16")) {
    // A/B multiplicand is fp16, SM 70 Tensor Core instructions
    ICHECK((C_dtype == "fp16") || (C_dtype == "fp32"));
    if (C_dtype == "fp16") {
      // C accumulator is fp16
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((unsigned *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k4.left_layout.right_layout.f16.f16.f16.f16 "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, "
                  "{%8,%9,%10,%11};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // C accumulator is fp32
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((float *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k4.left_layout.right_layout.f32.f16.f16.f32 "
                  "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                  "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
                  : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]),
                    "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), 
                    "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
                    "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
            }
          )";
    }
  } else {
    // A/B multiplicand is fp64, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "fp64");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Fp64 Tensor Core instructions "
        << "with shape m8n8k4 expect A layout is row major and B layout is col major.";
    // C accumulator is fp64
    new_a_ref = "((double *)(" + a_ref + " + " + a_bias + "))";
    new_b_ref = "((double *)(" + b_ref + " + " + b_bias + "))";
    new_c_ref = "((double *)(" + c_ref + " + " + c_bias + "))";
    asm_code = R"(
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0,%1}, {%2}, {%3}, "
                "{%4,%5};\n"
                : "=d"(D[0]), "=d"(D[1])
                : "d"(A[0]), "d"(B[0]), 
                  "d"(C[0]), "d"(C[1]));
          }
        )";
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm16n8k8Assembly(const std::string& A_layout, const std::string& B_layout,
                                    const std::string& A_dtype, const std::string& B_dtype,
                                    const std::string& C_dtype, const std::string& a_ref,
                                    const std::string& a_bias, const std::string& b_ref,
                                    const std::string& b_bias, const std::string& c_ref,
                                    const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "fp16") && (B_dtype == "fp16")) ||
         ((A_dtype == "bf16") && (B_dtype == "bf16")));
  ICHECK(saturate == false) << "Saturate is not allowed for m16n8k8 mma.";
  if ((A_dtype == "fp16") && (B_dtype == "fp16")) {
    // A/B multiplicand is fp16, SM 75 Tensor Core instructions
    ICHECK((C_dtype == "fp16") || (C_dtype == "fp32"));
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m16n8k8 expect A layout is row major and B layout is col major.";
    if (C_dtype == "fp16") {
      // C accumulator is fp16
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((unsigned *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                  "{%0,%1}, {%2,%3}, {%5}, "
                  "{%5,%6};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // C accumulator is fp32
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((float *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
            }
          )";
    }
  } else {
    // A/B multiplicand is bf16, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "fp32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k8 expect A layout is row major and B layout is col major.";
    // C accumulator is fp32
    new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
    new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
    new_c_ref = "((float *)(" + c_ref + " + " + c_bias + "))";
    asm_code = R"(
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                "{%7,%8,%9,%10};\n"
                : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                  "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
          }
        )";
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm8n8k16Assembly(const std::string& A_layout, const std::string& B_layout,
                                    const std::string& A_dtype, const std::string& B_dtype,
                                    const std::string& C_dtype, const std::string& a_ref,
                                    const std::string& a_bias, const std::string& b_ref,
                                    const std::string& b_bias, const std::string& c_ref,
                                    const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "int8") && (B_dtype == "int8")) ||
         ((A_dtype == "uint8") && (B_dtype == "int8")) ||
         ((A_dtype == "int8") && (B_dtype == "uint8")) ||
         ((A_dtype == "uint8") && (B_dtype == "uint8")));
  if ((A_dtype == "int8") && (B_dtype == "int8")) {
    // A/B multiplicand is int8, SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  } else if ((A_dtype == "uint8") && (B_dtype == "int8")) {
    // A multiplicand is uint8, B multiplicand is int8
    // SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  } else if ((A_dtype == "int8") && (B_dtype == "uint8")) {
    // A multiplicand is int8, B multiplicand is uint8
    // SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  } else {
    // A/B multiplicand is uint8, SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm8n8k32Assembly(const std::string& A_layout, const std::string& B_layout,
                                    const std::string& A_dtype, const std::string& B_dtype,
                                    const std::string& C_dtype, const std::string& a_ref,
                                    const std::string& a_bias, const std::string& b_ref,
                                    const std::string& b_bias, const std::string& c_ref,
                                    const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "int4") && (B_dtype == "int4")) ||
         ((A_dtype == "uint4") && (B_dtype == "int4")) ||
         ((A_dtype == "int4") && (B_dtype == "uint4")) ||
         ((A_dtype == "uint4") && (B_dtype == "uint4")));
  if ((A_dtype == "int4") && (B_dtype == "int4")) {
    // A/B multiplicand is int4, SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  } else if ((A_dtype == "uint4") && (B_dtype == "int4")) {
    // A multiplicand is uint4, B multiplicand is int4
    // SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  } else if ((A_dtype == "int4") && (B_dtype == "uint4")) {
    // A multiplicand is int4, B multiplicand is uint4
    // SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  } else {
    // A/B multiplicand is uint4, SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM75 Tensor Core instructions "
        << "with shape m8n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 "
                  "{%0,%1}, {%2}, {%3}, "
                  "{%4,%5};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    }
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm16n8k4Assembly(const std::string& A_layout, const std::string& B_layout,
                                    const std::string& A_dtype, const std::string& B_dtype,
                                    const std::string& C_dtype, const std::string& a_ref,
                                    const std::string& a_bias, const std::string& b_ref,
                                    const std::string& b_bias, const std::string& c_ref,
                                    const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK((A_dtype == "tf32") && (B_dtype == "tf32"));
  ICHECK(saturate == false) << "Saturate is not allowed for m16n8k4 mma.";
  // A/B multiplicand is tf32, SM 80 Tensor Core instructions
  ICHECK(C_dtype == "fp32");
  ICHECK((A_layout == "row") && (B_layout == "col"))
      << "SM80 Tensor Core instructions "
      << "with shape m16n8k4 expect A layout is row major and B layout is col major.";
  // C accumulator is fp32
  new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
  new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
  new_c_ref = "((float *)(" + c_ref + " + " + c_bias + "))";
  asm_code = R"(
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
              "{%10,%11,%12,%13};\n"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "f"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]), 
                "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
        }
      )";
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm16n8k16Assembly(const std::string& A_layout, const std::string& B_layout,
                                     const std::string& A_dtype, const std::string& B_dtype,
                                     const std::string& C_dtype, const std::string& a_ref,
                                     const std::string& a_bias, const std::string& b_ref,
                                     const std::string& b_bias, const std::string& c_ref,
                                     const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "fp16") && (B_dtype == "fp16")) ||
         ((A_dtype == "bf16") && (B_dtype == "bf16")) ||
         ((A_dtype == "int8") && (B_dtype == "int8")) ||
         ((A_dtype == "uint8") && (B_dtype == "int8")) ||
         ((A_dtype == "int8") && (B_dtype == "uint8")) ||
         ((A_dtype == "uint8") && (B_dtype == "uint8")));
  if ((A_dtype == "fp16") && (B_dtype == "fp16")) {
    ICHECK(saturate == false) << "Saturate is not allowed for m16n8k8 fp16 mma.";
    // A/B multiplicand is fp16, SM 80 Tensor Core instructions
    ICHECK((C_dtype == "fp16") || (C_dtype == "fp32"));
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k16 expect A layout is row major and B layout is col major.";
    if (C_dtype == "fp16") {
      // C accumulator is fp16
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((unsigned *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                  "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
                  "{%8,%9};\n"
                  : "=r"(D[0]), "=r"(D[1])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]), 
                    "r"(C[0]), "r"(C[1]));
            }
          )";
    } else {
      // C accumulator is fp32
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((float *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
            }
          )";
    }
  } else if ((A_dtype == "bf16") && (B_dtype == "bf16")) {
    // A/B multiplicand is bf16, SM 80 Tensor Core instructions
    ICHECK(saturate == false) << "Saturate is not allowed for m16n8k8 bf16 mma.";
    ICHECK(C_dtype == "fp32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is fp32
    new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
    new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
    new_c_ref = "((float *)(" + c_ref + " + " + c_bias + "))";
    asm_code = R"(
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
          }
        )";
  } else if ((A_dtype == "int8") && (B_dtype == "int8")) {
    // A/B multiplicand is int8, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else if ((A_dtype == "uint8") && (B_dtype == "int8")) {
    // A multiplicand is uint8, B multiplicand is int8
    // SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else if ((A_dtype == "int8") && (B_dtype == "uint8")) {
    // A multiplicand is int8, B multiplicand is uint8
    // SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else {
    // A/B multiplicand is uint8, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k16 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5}, {%6}, "
                  "{%7,%8,%9,%10};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(B[0]), 
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm16n8k32Assembly(const std::string& A_layout, const std::string& B_layout,
                                     const std::string& A_dtype, const std::string& B_dtype,
                                     const std::string& C_dtype, const std::string& a_ref,
                                     const std::string& a_bias, const std::string& b_ref,
                                     const std::string& b_bias, const std::string& c_ref,
                                     const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "int8") && (B_dtype == "int8")) ||
         ((A_dtype == "uint8") && (B_dtype == "int8")) ||
         ((A_dtype == "int8") && (B_dtype == "uint8")) ||
         ((A_dtype == "uint8") && (B_dtype == "uint8")));
  if ((A_dtype == "int8") && (B_dtype == "int8")) {
    // A/B multiplicand is int8, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else if ((A_dtype == "uint8") && (B_dtype == "int8")) {
    // A multiplicand is uint8, B multiplicand is int8
    // SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else if ((A_dtype == "int8") && (B_dtype == "uint8")) {
    // A multiplicand is int8, B multiplicand is uint8
    // SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else {
    // A/B multiplicand is uint8, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k32 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm16n8k64Assembly(const std::string& A_layout, const std::string& B_layout,
                                     const std::string& A_dtype, const std::string& B_dtype,
                                     const std::string& C_dtype, const std::string& a_ref,
                                     const std::string& a_bias, const std::string& b_ref,
                                     const std::string& b_bias, const std::string& c_ref,
                                     const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "int4") && (B_dtype == "int4")) ||
         ((A_dtype == "uint4") && (B_dtype == "int4")) ||
         ((A_dtype == "int4") && (B_dtype == "uint4")) ||
         ((A_dtype == "uint4") && (B_dtype == "uint4")));
  if ((A_dtype == "int4") && (B_dtype == "int4")) {
    // A/B multiplicand is int4, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k64 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else if ((A_dtype == "uint4") && (B_dtype == "int4")) {
    // A multiplicand is uint4, B multiplicand is int4
    // SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k64 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else if ((A_dtype == "int4") && (B_dtype == "uint4")) {
    // A multiplicand is int4, B multiplicand is uint4
    // SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k64 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  } else {
    // A/B multiplicand is uint4, SM 75 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k64 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    if (!saturate) {
      // no saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    } else {
      // saturate
      new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
      new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
      new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
      asm_code = R"(
            {
              __asm__ __volatile__(
                  "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                  "{%10,%11,%12,%13};\n"
                  : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
            }
          )";
    }
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAm16n8k256Assembly(const std::string& A_layout, const std::string& B_layout,
                                      const std::string& A_dtype, const std::string& B_dtype,
                                      const std::string& C_dtype, const std::string& a_ref,
                                      const std::string& a_bias, const std::string& b_ref,
                                      const std::string& b_bias, const std::string& c_ref,
                                      const std::string& c_bias, bool saturate) {
  std::string asm_code = "";
  std::string new_a_ref = "";
  std::string new_b_ref = "";
  std::string new_c_ref = "";
  ICHECK(((A_dtype == "uint1") && (B_dtype == "uint1")) ||
         ((A_dtype == "int1") && (B_dtype == "int1")));
  if ((A_dtype == "uint1") && (B_dtype == "uint1")) {
    // A/B multiplicand is uint1, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k256 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
    new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
    new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
    asm_code = R"(
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
          }
        )";
  } else {
    // A/B multiplicand is int1, SM 80 Tensor Core instructions
    ICHECK(C_dtype == "int32");
    ICHECK((A_layout == "row") && (B_layout == "col"))
        << "SM80 Tensor Core instructions "
        << "with shape m16n8k256 expect A layout is row major and B layout is col major.";
    // C accumulator is int32
    new_a_ref = "((unsigned *)(" + a_ref + " + " + a_bias + "))";
    new_b_ref = "((unsigned *)(" + b_ref + " + " + b_bias + "))";
    new_c_ref = "((int *)(" + c_ref + " + " + c_bias + "))";
    asm_code = R"(
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
          }
        )";
  }
  asm_code = ReplaceMMAArgument(asm_code, "left_layout", A_layout);
  asm_code = ReplaceMMAArgument(asm_code, "right_layout", B_layout);
  asm_code = ReplaceMMAArgument(asm_code, "A", new_a_ref);
  asm_code = ReplaceMMAArgument(asm_code, "B", new_b_ref);
  asm_code = ReplaceMMAArgument(asm_code, "C", new_c_ref);
  asm_code = ReplaceMMAArgument(asm_code, "D", new_c_ref);
  return asm_code;
}

std::string PrintMMAAssembly(const std::string& shape, const std::string& A_layout,
                             const std::string& B_layout, const std::string& A_dtype,
                             const std::string& B_dtype, const std::string& C_dtype,
                             const std::string& a_ref, const std::string& a_bias,
                             const std::string& b_ref, const std::string& b_bias,
                             const std::string& c_ref, const std::string& c_bias, bool saturate) {
  ICHECK((shape == "m8n8k4") || (shape == "m16n8k8") || (shape == "m8n8k16") ||
         (shape == "m8n8k32") || (shape == "m16n8k4") || (shape == "m16n8k16") ||
         (shape == "m16n8k32") || (shape == "m16n8k64") || (shape == "m16n8k256"));
  ICHECK((A_layout == "row") || (A_layout == "col")) << "Unknown A layout: " << A_layout;
  ICHECK((B_layout == "row") || (B_layout == "col")) << "Unknown B layout: " << B_layout;

  if (shape == "m8n8k4") {
    return PrintMMAm8n8k4Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                  b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m16n8k8") {
    return PrintMMAm16n8k8Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                   b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m8n8k16") {
    return PrintMMAm8n8k16Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                   b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m8n8k32") {
    return PrintMMAm8n8k32Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                   b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m16n8k4") {
    return PrintMMAm16n8k4Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                   b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m16n8k16") {
    return PrintMMAm16n8k16Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                    b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m16n8k32") {
    return PrintMMAm16n8k32Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                    b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m16n8k64") {
    return PrintMMAm16n8k64Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                    b_ref, b_bias, c_ref, c_bias, saturate);
  } else if (shape == "m16n8k256") {
    return PrintMMAm16n8k256Assembly(A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias,
                                     b_ref, b_bias, c_ref, c_bias, saturate);
  }
  /*
   * TODO: add mma.m16n8k128
   */
  throw Error("Unknown PTX mma instructions.");
}

}  // namespace codegen
}  // namespace tvm
