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
 * \file op_table.h
 * \brief A operator table for parsing.
 * Provides symbolic token sequences to map to TVM operators, with a given associativity and arity.
 */

#ifndef TVM_RELAY_PARSER_OP_TABLE_H_
#define TVM_RELAY_PARSER_OP_TABLE_H_

#include <tvm/ir/op.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "./tokenizer.h"

namespace tvm {
namespace relay {

struct Rule {
  std::vector<TokenType> tokens;
  int precedence;
  int arity;
  tvm::Op op;
  bool left_assoc;

  Rule() : tokens(), precedence(0), arity(0), op(tvm::Op()), left_assoc(false) {}

  Rule(std::vector<TokenType> tokens, tvm::Op op, int precedence, int arity = 2,
       bool left_assoc = false)
      : tokens(tokens), precedence(precedence), arity(arity), op(op), left_assoc(left_assoc) {}

  Rule(const Rule& rule) {
    this->tokens = rule.tokens;
    this->op = rule.op;
    this->precedence = rule.precedence;
    this->arity = rule.arity;
    this->left_assoc = rule.left_assoc;
  }
};

struct OperatorTable {
  std::vector<Rule> rules;
  std::unordered_map<std::string, Rule> this_is_a_hack;

  explicit OperatorTable(std::vector<Rule> rules) : rules(rules), this_is_a_hack() {
    for (auto rule : rules) {
      std::stringstream key;
      for (auto token : rule.tokens) {
        key << ToString(token);
      }
      this->this_is_a_hack.insert({key.str(), rule});
    }
  }
};

inline OperatorTable DefaultOpTable() {
  return OperatorTable(
      {Rule({TokenType::kStar}, Op::Get("multiply"), 12, 2, true),
       Rule({TokenType::kDivision}, Op::Get("divide"), 12, 2, true),
       Rule({TokenType::kPlus}, Op::Get("add"), 10, 2, true),
       Rule({TokenType::kMinus}, Op::Get("subtract"), 10, 2, true),
       Rule({TokenType::kLAngle}, Op::Get("less"), 8, 2, true),
       Rule({TokenType::kLAngle, TokenType::kEqual}, Op::Get("less_equal"), 8, 2, true),
       Rule({TokenType::kRAngle}, Op::Get("greater"), 8, 2, true),
       Rule({TokenType::kRAngle, TokenType::kEqual}, Op::Get("greater_equal"), 8, 2, true),
       Rule({TokenType::kEqual, TokenType::kEqual}, Op::Get("equal"), 7, 2, true),
       Rule({TokenType::kBang, TokenType::kEqual}, Op::Get("not_equal"), 7, 2, true)});
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PARSER_OP_TABLE_H_
