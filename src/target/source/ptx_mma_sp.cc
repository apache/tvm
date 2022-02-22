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
 * \file ptx_mma_sp.cc
 */

#include "ptx_mma_sp.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

namespace ptx {

/*!
 * \brief PTX data type.
 */
enum class DataType : int {
  kInt4 = 0,
  kUInt4 = 1,
  kInt8 = 2,
  kUInt8 = 3,
  kInt16 = 4,
  kUInt16 = 5,
  kInt32 = 6,
  kUInt32 = 7,
  kInt64 = 8,
  kUInt64 = 9,
  kFloat16 = 10,
  kBFloat16 = 11,
  kFloat16x2 = 12,
  kFloat32 = 13,
  kTensorFloat32 = 14,
  kFloat64 = 15,
  kBit1 = 16
};

static const char* dtype_str[] = {".s4",    ".u4",  ".s8",   ".u8",  ".s16", ".u16",
                                  ".s32",   ".u32", ".s64",  ".u64", ".f16", ".bf16",
                                  ".f16x2", ".f32", ".tf32", ".f64", ".b1"};
static uint32_t num_bits[] = {4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 16, 16, 32, 32, 32, 64, 1};

/*!
 * \brief Create PTX data type from string.
 */
inline DataType DTypeFromString(const std::string str) {
  if (str == "int4" || str == ".s4") {
    return DataType::kInt4;
  } else if (str == "uint4" || str == ".u4") {
    return DataType::kUInt4;
  } else if (str == "int8" || str == ".s8") {
    return DataType::kUInt8;
  } else if (str == "uint8" || str == ".u8") {
    return DataType::kUInt8;
  } else if (str == "int16" || str == ".s16") {
    return DataType::kInt16;
  } else if (str == "uint16" || str == ".u16") {
    return DataType::kUInt16;
  } else if (str == "int32" || str == ".s32") {
    return DataType::kInt32;
  } else if (str == "uint32" || str == ".u32") {
    return DataType::kUInt32;
  } else if (str == "int64" || str == ".s64") {
    return DataType::kInt64;
  } else if (str == "uint64" || str == ".u64") {
    return DataType::kUInt64;
  } else if (str == "float16" || str == "fp16" || str == ".f16") {
    return DataType::kFloat16;
  } else if (str == "bfloat16" || str == "bf16") {
    return DataType::kBFloat16;
  } else if (str == ".f16x2") {
    return DataType::kFloat16x2;
  } else if (str == "float32" || str == "fp32" || str == ".f32") {
    return DataType::kFloat32;
  } else if (str == "tf32") {
    return DataType::kTensorFloat32;
  } else if (str == "float64" || str == "fp64" || str == ".f64") {
    return DataType::kFloat64;
  } else if (str == ".b1") {
    return DataType::kBit1;
  } else {
    LOG(FATAL) << "Unrecognized data type " << str << " for PTX.";
    return DataType(0);
  }
}

/*!
 * \brief Get the string representation of given PTX data type.
 */
inline std::string DTypeToString(DataType dtype) { return dtype_str[static_cast<int>(dtype)]; }

/*!
 * \brief Get the number of bits of given PTX data type.
 */
inline uint32_t DTypeBits(DataType dtype) { return num_bits[static_cast<int>(dtype)]; }

/*!
 * \brief Extract the value m, n, k from string m*n*k*
 */
std::tuple<int, int, int> ParseMMAShape(const std::string& str) {
  size_t pos_m = str.find("m"), pos_n = str.find("n"), pos_k = str.find("k");
  CHECK(pos_m != str.npos && pos_n != str.npos && pos_k != str.npos)
      << "Cannot parse MMA shape " << str;
  int m = std::stoi(str.substr(pos_m + 1, pos_n - pos_m - 1)),
      n = std::stoi(str.substr(pos_n + 1, pos_k - pos_n - 1)), k = std::stoi(str.substr(pos_k + 1));
  return {m, n, k};
}

/*!
 * \brief Fragment attributes of given data type.
 * \return the register type in ptx, fragment size, fragment pointer string.
 */
inline std::tuple<char, int, std::string> FragmentAttrs(DataType dtype) {
  switch (dtype) {
    case DataType::kBit1:
    case DataType::kInt4:
    case DataType::kUInt4:
    case DataType::kInt8:
    case DataType::kUInt8:
    case DataType::kFloat16:  // .f16x2 register
    case DataType::kBFloat16:
    case DataType::kTensorFloat32:
      return {'r', 32, "(unsigned *)"};
    case DataType::kInt32:
      return {'r', 32, "(int *)"};
    case DataType::kFloat32:
      return {'f', 32, "(float *)"};
    case DataType::kFloat64:
      return {'d', 64, "(double *)"};
    default:
      LOG(FATAL) << DTypeToString(dtype) << " is not matrix data type in MMA.";
      return {'\0', 0, ""};
  }
}

};  // namespace ptx

/*!
 * \brief Replace patterns with replacement strings.
 * \note should use std::format instead when codebase is ported to C++20.
 */
class Replacer {
 public:
  void register_rule(const std::string& pattern, const std::string& replacement) {
    _rules.emplace_back(pattern, replacement);
  }
  std::string rewrite(std::string str) {
    for (auto&& rule : _rules) {
      std::string pattern, replacement;
      std::tie(pattern, replacement) = rule;
      size_t len = pattern.size();
      size_t new_len = replacement.size();
      size_t pos = str.find(pattern);
      while (pos != std::string::npos) {
        str = str.replace(pos, len, replacement);
        pos = str.find(pattern, pos + new_len);
      }
    }
    return str;
  }
  void empty_rules() { _rules.clear(); }

 private:
  std::vector<std::pair<std::string, std::string>> _rules;
};

/*!
 * \brief Return template string, input operands string and output operands string.
 */
inline std::tuple<std::string, std::string, std::string> get_mma_sp_operands(
    int m, int n, int k, ptx::DataType dtype_a, ptx::DataType dtype_b, ptx::DataType dtype_c) {
  std::stringstream templates, inputs, outputs;
  auto frag_attr_a = ptx::FragmentAttrs(dtype_a), frag_attr_b = ptx::FragmentAttrs(dtype_b),
       frag_attr_c = ptx::FragmentAttrs(dtype_c);
  constexpr int warp_size = 32;
  int num_operands_a, num_operands_b, num_operands_c;
  num_operands_a = (m * k / 2) * ptx::DTypeBits(dtype_a) / std::get<1>(frag_attr_a) / warp_size;
  num_operands_b = (k * n) * ptx::DTypeBits(dtype_b) / std::get<1>(frag_attr_b) / warp_size;
  num_operands_c = (m * n) * ptx::DTypeBits(dtype_c) / std::get<1>(frag_attr_c) / warp_size;

  // generate templates;
  int arg_counter = 0;
  templates << "{"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_c; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_a; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_b; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, {"
            << "%" << arg_counter++;
  for (int i = 1; i < num_operands_c; ++i) {
    templates << ", %" << arg_counter++;
  }
  templates << "}, %" << (arg_counter++) << ", F";

  // generate inputs
  for (int i = 0; i < num_operands_a; ++i) {
    if (i != 0) {
      inputs << ", ";
    }
    inputs << "\"" << std::get<0>(frag_attr_a) << "\"((" << std::get<2>(frag_attr_a) << "(A))[" << i
           << "])";
  }
  for (int i = 0; i < num_operands_b; ++i) {
    inputs << ", \"" << std::get<0>(frag_attr_b) << "\"((" << std::get<2>(frag_attr_b) << "(B))["
           << i << "])";
  }
  for (int i = 0; i < num_operands_c; ++i) {
    inputs << ", \"" << std::get<0>(frag_attr_c) << "\"((" << std::get<2>(frag_attr_c) << "(C))["
           << i << "])";
  }
  inputs << ", \"r\"(E[0])";

  // generate outputs
  for (int i = 0; i < num_operands_c; ++i) {
    if (i != 0) {
      outputs << ",";
    }
    outputs << " \"=" << std::get<0>(frag_attr_c) << "\"((" << std::get<2>(frag_attr_c) << "(D))["
            << i << "])";
  }
  return {templates.str(), inputs.str(), outputs.str()};
}

std::string PrintMMASparseAssembly(const std::string& shape, const std::string& A_layout,
                                   const std::string& B_layout, const std::string& A_dtype,
                                   const std::string& B_dtype, const std::string& C_dtype,
                                   const std::string& a_ref, const std::string& a_offset,
                                   const std::string& b_ref, const std::string& b_offset,
                                   const std::string& c_ref, const std::string& c_offset,
                                   const std::string& metadata,
                                   const std::string& sparsity_selector, bool saturate) {
  ptx::DataType dtype_a = ptx::DTypeFromString(A_dtype), dtype_b = ptx::DTypeFromString(B_dtype),
                dtype_c = ptx::DTypeFromString(C_dtype);
  int m, n, k;
  std::tie(m, n, k) = ptx::ParseMMAShape(shape);
  std::string asm_code = R"(
  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.{shape}.{alayout}.{blayout}{satinite}{dtype}{atype}{btype}{ctype}"
      "{templates};\n"
      : {outputs}
      : {inputs});
  }
)";
  std::string templates_str, inputs_str, outputs_str;
  std::tie(templates_str, inputs_str, outputs_str) =
      get_mma_sp_operands(m, n, k, dtype_a, dtype_b, dtype_c);

  // replace patterns
  Replacer replacer;
  replacer.register_rule("{shape}", shape);
  replacer.register_rule("{satinite}", saturate ? ".satinite" : "");
  replacer.register_rule("{alayout}", A_layout);
  replacer.register_rule("{blayout}", B_layout);
  replacer.register_rule("{atype}", ptx::DTypeToString(dtype_a));
  replacer.register_rule("{btype}", ptx::DTypeToString(dtype_b));
  replacer.register_rule("{ctype}", ptx::DTypeToString(dtype_c));
  replacer.register_rule("{dtype}", ptx::DTypeToString(dtype_c));
  replacer.register_rule("{templates}", templates_str);
  replacer.register_rule("{outputs}", outputs_str);
  replacer.register_rule("{inputs}", inputs_str);
  asm_code = replacer.rewrite(asm_code);
  replacer.empty_rules();
  replacer.register_rule("A", a_ref + " + " + a_offset);
  replacer.register_rule("B", b_ref + " + " + b_offset);
  replacer.register_rule("C", c_ref + " + " + c_offset);
  replacer.register_rule("D", c_ref + " + " + c_offset);
  replacer.register_rule("E", metadata);
  replacer.register_rule("F", sparsity_selector);
  asm_code = replacer.rewrite(asm_code);
  return asm_code;
}

}  // namespace codegen
}  // namespace tvm
