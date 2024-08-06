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

#ifndef TVM_RELAY_PARSER_TOKEN_H_
#define TVM_RELAY_PARSER_TOKEN_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/source_map.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <string>
#include <utility>

namespace tvm {
namespace relay {

enum class TokenType {
  kCommentStart,
  kCommentEnd,
  kLineComment,
  kComment,
  kWhitespace,
  kNewline,
  kStringLiteral,
  kIdentifier,
  kLocal,
  kGlobal,
  kOp,
  kGraph,
  kOpenParen,
  kCloseParen,
  kAtSymbol,
  kPercent,
  kComma,
  kPeriod,
  kEqual,
  kSemicolon,
  kColon,
  kInteger,
  kFloat,
  kDivision,
  kBoolean,
  kPlus,
  kStar,
  kMinus,
  kRAngle,
  kLAngle,
  kRCurly,
  kLCurly,
  kRSquare,
  kLSquare,
  kBang,
  kAt,
  kQuestion,
  kIf,
  kElse,
  kUnderscore,
  kLet,
  kFn,
  kDefn,
  kTypeDef,
  kExtern,
  kMatch,
  kPartialMatch,
  kMetadata,
  kMetaReference,
  kFreeVar,
  kRef,
  kRefRead,
  kRefWrite,
  kVersion,
  kUnknown,
  kEndOfFile,
  kNull,
};

inline std::string ToString(const TokenType& token_type) {
  switch (token_type) {
    case TokenType::kCommentStart:
      return "CommentStart";
    case TokenType::kCommentEnd:
      return "CommentEnd";
    case TokenType::kLineComment:
      return "LineComment";
    case TokenType::kComment:
      return "Comment";
    case TokenType::kWhitespace:
      return "WhiteSpace";
    case TokenType::kNewline:
      return "Newline";
    case TokenType::kStringLiteral:
      return "StringLiteral";
    case TokenType::kIdentifier:
      return "Identifier";
    case TokenType::kLocal:
      return "Local";
    case TokenType::kGlobal:
      return "Global";
    case TokenType::kGraph:
      return "Graph";
    case TokenType::kOp:
      return "Op";
    case TokenType::kOpenParen:
      return "OpenParen";
    case TokenType::kCloseParen:
      return "CloseParen";
    case TokenType::kAtSymbol:
      return "AtSymbol";
    case TokenType::kPercent:
      return "Percent";
    case TokenType::kComma:
      return "Comma";
    case TokenType::kColon:
      return "Colon";
    case TokenType::kSemicolon:
      return "Semicolon";
    case TokenType::kPeriod:
      return "Period";
    case TokenType::kEqual:
      return "Equal";
    case TokenType::kInteger:
      return "Integer";
    case TokenType::kFloat:
      return "Float";
    case TokenType::kPlus:
      return "Plus";
    case TokenType::kStar:
      return "Star";
    case TokenType::kMinus:
      return "Minus";
    case TokenType::kDivision:
      return "Division";
    case TokenType::kRAngle:
      return "RAngle";
    case TokenType::kLAngle:
      return "LAngle";
    case TokenType::kRCurly:
      return "RCurly";
    case TokenType::kLCurly:
      return "LCurly";
    case TokenType::kRSquare:
      return "RSquare";
    case TokenType::kLSquare:
      return "LSquare";
    case TokenType::kBang:
      return "Bang";
    case TokenType::kUnderscore:
      return "Underscore";
    case TokenType::kAt:
      return "At";
    case TokenType::kLet:
      return "Let";
    case TokenType::kIf:
      return "If";
    case TokenType::kElse:
      return "Else";
    case TokenType::kFn:
      return "Fn";
    case TokenType::kDefn:
      return "Defn";
    case TokenType::kTypeDef:
      return "TypeDef";
    case TokenType::kExtern:
      return "Extern";
    case TokenType::kMatch:
      return "Match";
    case TokenType::kPartialMatch:
      return "PartialMatch";
    case TokenType::kQuestion:
      return "Question";
    case TokenType::kBoolean:
      return "Boolean";
    case TokenType::kMetadata:
      return "Metadata";
    case TokenType::kMetaReference:
      return "MetaReference";
    case TokenType::kFreeVar:
      return "FreeVar";
    case TokenType::kVersion:
      return "Version";
    case TokenType::kRef:
      return "Ref";
    case TokenType::kRefRead:
      return "RefRead";
    case TokenType::kRefWrite:
      return "RefWrite";
    case TokenType::kUnknown:
      return "Unknown";
    case TokenType::kEndOfFile:
      return "EndOfFile";
    case TokenType::kNull:
      return "Null";
    // Older compilers warn even though the above code is exhaustive.
    default:
      LOG(FATAL) << "unreachable code";
  }
}

inline std::string Pretty(const TokenType& token_type) {
  switch (token_type) {
    case TokenType::kCommentStart:
      return "`/*`";
    case TokenType::kCommentEnd:
      return "`*/`";
    case TokenType::kLineComment:
      return "`//`";
    case TokenType::kComment:
      return "comment";
    case TokenType::kWhitespace:
      return "whitespace";
    case TokenType::kNewline:
      return "newline";
    case TokenType::kStringLiteral:
      return "string literal";
    case TokenType::kIdentifier:
      return "identifier";
    case TokenType::kLocal:
      return "local variable";
    case TokenType::kGlobal:
      return "global variable";
    case TokenType::kGraph:
      return "graph variable";
    case TokenType::kOp:
      return "operator";
    case TokenType::kOpenParen:
      return "`(`";
    case TokenType::kCloseParen:
      return "`)`";
    case TokenType::kAtSymbol:
      return "`@`";
    case TokenType::kPercent:
      return "`%`";
    case TokenType::kComma:
      return "`,`";
    case TokenType::kColon:
      return "`:`";
    case TokenType::kSemicolon:
      return "`;`";
    case TokenType::kPeriod:
      return "`.`";
    case TokenType::kEqual:
      return "`=`";
    case TokenType::kInteger:
      return "integer";
    case TokenType::kFloat:
      return "float";
    case TokenType::kPlus:
      return "`+`";
    case TokenType::kStar:
      return "`*`";
    case TokenType::kMinus:
      return "`-`";
    case TokenType::kDivision:
      return "`/`";
    case TokenType::kRAngle:
      return "`<`";
    case TokenType::kLAngle:
      return "`>`";
    case TokenType::kRCurly:
      return "`}`";
    case TokenType::kLCurly:
      return "`{`";
    case TokenType::kRSquare:
      return "`]`";
    case TokenType::kLSquare:
      return "`[`";
    case TokenType::kBang:
      return "`!`";
    case TokenType::kUnderscore:
      return "`_`";
    case TokenType::kAt:
      return "`@`";
    case TokenType::kLet:
      return "`let`";
    case TokenType::kIf:
      return "`if`";
    case TokenType::kElse:
      return "`else`";
    case TokenType::kFn:
      return "`fn`";
    case TokenType::kDefn:
      return "`def`";
    case TokenType::kTypeDef:
      return "`type`";
    case TokenType::kExtern:
      return "`extern`";
    case TokenType::kBoolean:
      return "boolean";
    case TokenType::kMetadata:
      return "metadata section";
    case TokenType::kMetaReference:
      return "`meta`";
    case TokenType::kFreeVar:
      return "`free_var`";
    case TokenType::kMatch:
      return "`match`";
    case TokenType::kPartialMatch:
      return "`match?`";
    case TokenType::kQuestion:
      return "`?`";
    case TokenType::kRef:
      return "`ref`";
    case TokenType::kRefRead:
      return "`ref_read`";
    case TokenType::kRefWrite:
      return "`ref_write`";
    case TokenType::kUnknown:
      return "unknown";
    case TokenType::kEndOfFile:
      return "end of file";
    case TokenType::kNull:
      return "null";
    case TokenType::kVersion:
      return "version attribute";
    // Older compilers warn even though the above code is exhaustive.
    default:
      LOG(FATAL) << "unreachable code";
  }
}

class Token;

class TokenNode : public Object {
 public:
  Span span;
  TokenType token_type;
  mutable runtime::ObjectRef data;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "parser.Token";
  TVM_DECLARE_FINAL_OBJECT_INFO(TokenNode, Object);
};

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TokenNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TokenNode*>(ref.get());
      p->stream << "Token(span=" << node->span << ", token_type=" << ToString(node->token_type)
                << ", data=" << node->data << ")";
    });

TVM_REGISTER_NODE_TYPE(TokenNode);

class Token : public ObjectRef {
 public:
  TVM_DLL explicit Token(Span span, TokenType token_type, ObjectRef data = ObjectRef());

  static Token Null();
  int64_t ToNumber() const;
  std::string ToString() const;
  Map<String, Array<ObjectRef>> ToMetadata() const;
  TVM_DEFINE_OBJECT_REF_METHODS(Token, ObjectRef, TokenNode);
};

inline Token::Token(Span span, TokenType token_type, ObjectRef data) {
  ObjectPtr<TokenNode> n = make_object<TokenNode>();
  n->span = span;
  n->token_type = token_type;
  n->data = data;
  data_ = std::move(n);
}

inline Token Token::Null() { return Token(Span(SourceName(), 0, 0, 0, 0), TokenType::kNull); }

inline int64_t Token::ToNumber() const {
  return Downcast<tvm::Integer>(this->operator->()->data).IntValue();
}

inline std::string Token::ToString() const {
  return Downcast<tvm::String>(this->operator->()->data);
}

inline Map<String, Array<ObjectRef>> Token::ToMetadata() const {
  ObjectRef data = this->operator->()->data;
  if (data.defined()) {
    return Downcast<Map<String, Array<ObjectRef>>>(data);
  } else {
    return Map<String, Array<ObjectRef>>({});
  }
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PARSER_TOKEN_H_
