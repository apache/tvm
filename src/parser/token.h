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
 * \file token.h
 * \brief The definition of tokens for the TVM parser.
 */

#ifndef TVM_PARSER_TOKEN_H_
#define TVM_PARSER_TOKEN_H_

#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <string>
#include <utility>

namespace tvm {
namespace parser {

using namespace runtime;

enum TokenType {
  CommentStart,
  CommentEnd,
  LineComment,
  Comment,
  Whitespace,
  Newline,
  StringLiteral,
  Identifier,
  Local,
  Global,
  Op,
  Graph,
  OpenParen,
  CloseParen,
  AtSymbol,
  Percent,
  Comma,
  Period,
  Equal,
  Semicolon,
  Colon,
  Integer,
  Float,
  Division,
  Boolean,
  Plus,
  Star,
  Minus,
  RAngle,
  LAngle,
  RCurly,
  LCurly,
  RSquare,
  LSquare,
  Bang,
  At,
  Question,
  If,
  Else,
  Underscore,
  Let,
  Fn,
  Defn,
  TypeDef,
  Extern,
  Match,
  PartialMatch,
  Unknown,
  EndOfFile,
  Null,
};

std::string ToString(const TokenType& token_type) {
  switch (token_type) {
    case TokenType::CommentStart:
      return "CommentStart";
    case TokenType::CommentEnd:
      return "CommentEnd";
    case TokenType::LineComment:
      return "LineComment";
    case TokenType::Comment:
      return "Comment";
    case TokenType::Whitespace:
      return "WhiteSpace";
    case TokenType::Newline:
      return "Newline";
    case TokenType::StringLiteral:
      return "StringLiteral";
    case TokenType::Identifier:
      return "Identifier";
    case TokenType::Local:
      return "Local";
    case TokenType::Global:
      return "Global";
    case TokenType::Graph:
      return "Graph";
    case TokenType::Op:
      return "Op";
    case TokenType::OpenParen:
      return "OpenParen";
    case TokenType::CloseParen:
      return "CloseParen";
    case TokenType::AtSymbol:
      return "AtSymbol";
    case TokenType::Percent:
      return "Percent";
    case TokenType::Comma:
      return "Comma";
    case TokenType::Colon:
      return "Colon";
    case TokenType::Semicolon:
      return "Semicolon";
    case TokenType::Period:
      return "Period";
    case TokenType::Equal:
      return "Equal";
    case TokenType::Integer:
      return "Integer";
    case TokenType::Float:
      return "Float";
    case TokenType::Plus:
      return "Plus";
    case TokenType::Star:
      return "Star";
    case TokenType::Minus:
      return "Minus";
    case TokenType::Division:
      return "Division";
    case TokenType::RAngle:
      return "RAngle";
    case TokenType::LAngle:
      return "LAngle";
    case TokenType::RCurly:
      return "RCurly";
    case TokenType::LCurly:
      return "LCurly";
    case TokenType::RSquare:
      return "RSquare";
    case TokenType::LSquare:
      return "LSquare";
    case TokenType::Bang:
      return "Bang";
    case TokenType::Underscore:
      return "Underscore";
    case TokenType::At:
      return "At";
    case TokenType::Let:
      return "Let";
    case TokenType::If:
      return "If";
    case TokenType::Else:
      return "Else";
    case TokenType::Fn:
      return "Fn";
    case TokenType::Defn:
      return "Defn";
    case TokenType::TypeDef:
      return "TypeDef";
    case TokenType::Extern:
      return "Extern";
    case TokenType::Match:
      return "Match";
    case TokenType::PartialMatch:
      return "PartialMatch";
    case TokenType::Question:
      return "Question";
    case TokenType::Boolean:
      return "Boolean";
    case TokenType::Unknown:
      return "Unknown";
    case TokenType::EndOfFile:
      return "EndOfFile";
    case TokenType::Null:
      return "Null";
    // Older compilers warn even though the above code is exhaustive.
    default:
      LOG(FATAL) << "unreachable code";
      return "";
  }
}

std::string Pretty(const TokenType& token_type) {
  switch (token_type) {
    case TokenType::CommentStart:
      return "`/*`";
    case TokenType::CommentEnd:
      return "`*/`";
    case TokenType::LineComment:
      return "`//`";
    case TokenType::Comment:
      return "comment";
    case TokenType::Whitespace:
      return "whitespace";
    case TokenType::Newline:
      return "newline";
    case TokenType::StringLiteral:
      return "string literal";
    case TokenType::Identifier:
      return "identifier";
    case TokenType::Local:
      return "local variable";
    case TokenType::Global:
      return "global variable";
    case TokenType::Graph:
      return "graph variable";
    case TokenType::Op:
      return "operator";
    case TokenType::OpenParen:
      return "`(`";
    case TokenType::CloseParen:
      return "`)`";
    case TokenType::AtSymbol:
      return "`@`";
    case TokenType::Percent:
      return "`%`";
    case TokenType::Comma:
      return "`,`";
    case TokenType::Colon:
      return "`:`";
    case TokenType::Semicolon:
      return "`;`";
    case TokenType::Period:
      return "`.`";
    case TokenType::Equal:
      return "`=`";
    case TokenType::Integer:
      return "integer";
    case TokenType::Float:
      return "float";
    case TokenType::Plus:
      return "`+`";
    case TokenType::Star:
      return "`*`";
    case TokenType::Minus:
      return "`-`";
    case TokenType::Division:
      return "`/`";
    case TokenType::RAngle:
      return "`<`";
    case TokenType::LAngle:
      return "`>`";
    case TokenType::RCurly:
      return "`}`";
    case TokenType::LCurly:
      return "`{`";
    case TokenType::RSquare:
      return "`]`";
    case TokenType::LSquare:
      return "`[`";
    case TokenType::Bang:
      return "`!`";
    case TokenType::Underscore:
      return "`_`";
    case TokenType::At:
      return "`@`";
    case TokenType::Let:
      return "`let`";
    case TokenType::If:
      return "`if`";
    case TokenType::Else:
      return "`else`";
    case TokenType::Fn:
      return "`fn`";
    case TokenType::Defn:
      return "`def`";
    case TokenType::TypeDef:
      return "`type`";
    case TokenType::Extern:
      return "`extern`";
    case TokenType::Boolean:
      return "boolean";
    case TokenType::Match:
      return "`match`";
    case TokenType::PartialMatch:
      return "`match?`";
    case TokenType::Question:
      return "`?`";
    case TokenType::Unknown:
      return "unknown";
    case TokenType::EndOfFile:
      return "end of file";
    case TokenType::Null:
      return "null";
    // Older compilers warn even though the above code is exhaustive.
    default:
      LOG(FATAL) << "unreachable code";
      return "";
  }
}

class Token;

class TokenNode : public Object {
 public:
  int line;
  int column;
  TokenType token_type;
  mutable runtime::ObjectRef data;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "parser.Token";
  TVM_DECLARE_FINAL_OBJECT_INFO(TokenNode, Object);
};

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TokenNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TokenNode*>(ref.get());
      p->stream << "Token(line=" << node->line << ", column=" << node->column
                << ", token_type=" << ToString(node->token_type) << ", data=" << node->data << ")";
    });

TVM_REGISTER_NODE_TYPE(TokenNode);

class Token : public ObjectRef {
 public:
  TVM_DLL explicit Token(int line, int column, TokenType token_type, ObjectRef data = ObjectRef());

  static Token Null();
  int64_t ToNumber() const;
  std::string ToString() const;
  TVM_DEFINE_OBJECT_REF_METHODS(Token, ObjectRef, TokenNode);
};

Token::Token(int line, int column, TokenType token_type, ObjectRef data) {
  ObjectPtr<TokenNode> n = make_object<TokenNode>();
  n->line = line;
  n->column = column;
  n->token_type = token_type;
  n->data = data;
  data_ = std::move(n);
}

Token Token::Null() { return Token(0, 0, TokenType::Null); }

int64_t Token::ToNumber() const { return Downcast<tvm::Integer>(this->operator->()->data); }

std::string Token::ToString() const { return Downcast<tvm::String>(this->operator->()->data); }

}  // namespace parser
}  // namespace tvm
#endif  // TVM_PARSER_TOKEN_H_
