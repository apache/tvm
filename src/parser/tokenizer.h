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
 * \file parser.h
 * \brief A parser for TVM IR.
 */
#ifndef TVM_PARSER_TOKENIZER_H_
#define TVM_PARSER_TOKENIZER_H_

#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "./token.h"

namespace tvm {
namespace parser {

using namespace runtime;

bool IsDigit(char c) { return '0' <= c && c <= '9'; }

bool IsWhitespace(char c) { return ' ' == c || c == '\t' || c == '\n'; }

bool IsNumeric(char c) {
  return (IsDigit(c) || c == '.' || c == 'e' || c == '-' || c == '+' || c == 'E') &&
         !IsWhitespace(c);
}

bool IsIdentLetter(char c) { return '_' == c || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z'); }

bool IsIdent(char c) { return IsIdentLetter(c) || IsDigit(c); }

static std::unordered_map<std::string, TokenType> KEYWORD_TABLE = {
    {"let", TokenType::Let},     {"fn", TokenType::Fn},        {"def", TokenType::Defn},
    {"if", TokenType::If},       {"else", TokenType::Else},    {"type", TokenType::TypeDef},
    {"match", TokenType::Match}, {"extern", TokenType::Extern}};

struct Tokenizer {
  size_t pos;
  int col;
  int line;
  char next_char;
  const std::string& source;
  std::vector<Token> tokens;

  char Next() {
    char c = this->source.at(this->pos);
    if (c == '\n') {
      this->line += 1;
      this->col = 1;
    } else {
      this->col += 1;
    }
    pos += 1;
    return c;
  }

  bool More() { return this->pos < this->source.size(); }

  char Peek() {
    CHECK(pos < this->source.size());
    return this->source.at(this->pos);
  }

  Token NewToken(TokenType token_type, ObjectRef data = ObjectRef()) {
    return Token(this->line, this->col, token_type, data);
  }

  enum CommentParserState {
    Proceed,
    Forward,
    Backward,
  };

  void MatchComment(std::string* buffer) {
    // We only invoke this after we have matched the first start
    // token assume, we are proceeding the parse forward with
    // nesting = 1.
    //
    // When we are done we should be at nesting zero and be
    // in the stop state.
    CommentParserState state = CommentParserState::Proceed;
    int nesting = 1;

    while (true) {
      switch (state) {
        case CommentParserState::Proceed: {
          if (Peek() == '/') {
            state = CommentParserState::Forward;
          } else if (Peek() == '*') {
            state = CommentParserState::Backward;
          }
          buffer->operator+=(Next());
          continue;
        }
        case CommentParserState::Forward: {
          if (Peek() == '*') {
            nesting += 1;
            buffer->operator+=(Next());
          }
          state = CommentParserState::Proceed;
          continue;
        }
        case CommentParserState::Backward: {
          if (Peek() == '/') {
            nesting -= 1;
            if (nesting == 0) {
              Next();
              buffer->pop_back();
              return;
            } else {
              buffer->operator+=(Next());
              state = CommentParserState::Proceed;
            }
          }
          continue;
        }
      }
    }
  }

  Token ParseNumber(bool is_pos, bool is_float, std::string number) {
    CHECK(number.size() > 0) << "an empty string is an invalid number";

    try {
      if (is_float) {
        throw std::invalid_argument("is_float");
      }
      auto token = NewToken(TokenType::Integer);
      size_t index = 0;
      int value = std::stoi(number, &index);
      if (number.size() > index) {
        throw std::invalid_argument("floating point");
      }
      value = is_pos ? value : -value;
      token->data = tvm::Integer(value);
      return token;
    } catch (const std::invalid_argument& ia) {
      auto token = NewToken(TokenType::Float);

      if (number.back() == 'f') {
        number.pop_back();
      }

      double value = stod(number);
      value = is_pos ? value : -value;
      token->data = tvm::FloatImm(DataType::Float(64), value);
      return token;
    }
  }

  inline Token TokenizeOnce() {
    auto next = Peek();
    if (next == '\n') {
      auto token = NewToken(TokenType::Newline);
      Next();
      return token;
    } else if (next == '\r') {
      Next();
      if (More() && Peek() == '\n') {
        auto token = NewToken(TokenType::Newline);
        return token;
      } else {
        // TODO(@jroesch): have lexer use diagnostic context too.
        LOG(FATAL) << "lexer error";
        return Token();
      }
    } else if (next == '"') {
      LOG(FATAL) << "string not working yet";
      return NewToken(TokenType::Unknown);
    } else if (IsWhitespace(next)) {
      auto token = NewToken(TokenType::Whitespace);
      Next();
      return token;
    } else if (IsDigit(next) || next == '-') {
      int negs = 0;
      while (More() && Peek() == '-') {
        Next();
        negs++;
      }
      // If there isn't a number right after either,
      // this is really slow for lexing, should replace
      // with multi-token return or something.
      if (negs && !IsDigit(Peek())) {
        pos = pos - (negs - 1);
        return NewToken(TokenType::Minus);
      }

      bool is_neg = negs % 2 == 1;
      std::stringstream ss;
      while (More() && IsNumeric(Peek())) {
        ss << Next();
      }

      bool is_float = false;
      // Remove trailing floating point prefix.
      if (More() && Peek() == 'f') {
        Next();
        is_float = true;
      }

      return ParseNumber(!is_neg, is_float, ss.str());
    } else if (next == '.') {
      auto token = NewToken(TokenType::Period);
      Next();
      return token;
    } else if (next == ',') {
      auto token = NewToken(TokenType::Comma);
      Next();
      return token;
    } else if (next == '=') {
      auto token = NewToken(TokenType::Equal);
      Next();
      return token;
    } else if (next == ';') {
      auto token = NewToken(TokenType::Semicolon);
      Next();
      return token;
    } else if (next == ':') {
      auto token = NewToken(TokenType::Colon);
      Next();
      return token;
    } else if (next == '(') {
      auto token = NewToken(TokenType::OpenParen);
      Next();
      return token;
    } else if (next == ')') {
      auto token = NewToken(TokenType::CloseParen);
      Next();
      return token;
    } else if (next == '+') {
      auto token = NewToken(TokenType::Plus);
      Next();
      return token;
    } else if (next == '-') {
      auto token = NewToken(TokenType::Minus);
      Next();
      return token;
    } else if (next == '*') {
      auto token = NewToken(TokenType::Star);
      Next();
      return token;
    } else if (next == '<') {
      auto token = NewToken(TokenType::LAngle);
      Next();
      return token;
    } else if (next == '>') {
      auto token = NewToken(TokenType::RAngle);
      Next();
      return token;
    } else if (next == '{') {
      auto token = NewToken(TokenType::LCurly);
      Next();
      return token;
    } else if (next == '}') {
      auto token = NewToken(TokenType::RCurly);
      Next();
      return token;
    } else if (next == '[') {
      auto token = NewToken(TokenType::LSquare);
      Next();
      return token;
    } else if (next == ']') {
      auto token = NewToken(TokenType::RSquare);
      Next();
      return token;
    } else if (next == '!') {
      auto token = NewToken(TokenType::Bang);
      Next();
      return token;
    } else if (next == '@') {
      auto token = NewToken(TokenType::At);
      Next();
      return token;
    } else if (next == '?') {
      auto token = NewToken(TokenType::Question);
      Next();
      return token;
    } else if (next == '%') {
      auto token = NewToken(TokenType::Percent);
      Next();
      return token;
    } else if (next == '/') {
      Next();
      if (Peek() == '/') {
        auto token = NewToken(TokenType::LineComment);
        // Consume the /
        Next();
        std::stringstream comment;
        while (More() && Peek() != '\n') {
          comment << Next();
        }
        token->data = tvm::String(comment.str());
        return token;
      } else if (Peek() == '*') {
        // Eat the first /* pair before entering the state machine.
        Next();
        std::string comment;
        MatchComment(&comment);
        auto token = NewToken(TokenType::Comment, tvm::String(comment));
        return token;
      } else {
        return NewToken(TokenType::Division);
      }
    } else if (IsIdentLetter(next)) {
      std::stringstream ss;
      // Due the below code we need to patch
      // the line/col info to the start of
      // token.
      int line = this->line;
      int col = this->col;

      while (More() && IsIdent(Peek())) {
        ss << Next();
      }

      std::string keyword = ss.str();
      auto it = KEYWORD_TABLE.find(keyword);

      TokenType token_type;
      if (it != KEYWORD_TABLE.end()) {
        token_type = it->second;

        if (token_type == TokenType::Match) {
          if (More() && Peek() == '?') {
            Next();
            token_type = TokenType::PartialMatch;
          }
        }
      } else {
        token_type = TokenType::Identifier;
      }

      return Token(line, col, token_type, tvm::String(ss.str()));
    } else {
      std::stringstream ss;
      while (More() && !IsWhitespace(Peek())) {
        ss << Next();
      }
      auto token = NewToken(TokenType::Unknown);
      token->data = tvm::String(ss.str());
      return token;
    }
  }

  void Tokenize() {
    while (this->More()) {
      auto token = TokenizeOnce();
      CHECK(token.defined());
      this->tokens.push_back(token);
    }
    this->tokens.push_back(NewToken(TokenType::EndOfFile));
  }

  explicit Tokenizer(std::string& source) : pos(0), col(1), line(1), source(source), tokens() {}
};

std::vector<Token> Condense(const std::vector<Token>& tokens) {
  std::vector<Token> out;

  for (size_t i = 0; i < tokens.size(); i++) {
    auto current = tokens.at(i);
    switch (current->token_type) {
      case TokenType::Percent: {
        auto next = tokens.at(i + 1);
        if (next->token_type == TokenType::Identifier) {
          // Match this token.
          i += 1;
          auto tok = Token(current->line, current->column, TokenType::Local, next->data);
          CHECK(tok.defined());
          out.push_back(tok);
        } else if (next->token_type == TokenType::Integer) {
          i += 1;
          auto tok = Token(current->line, current->column, TokenType::Graph, next->data);
          CHECK(tok.defined());
          out.push_back(tok);
        } else {
          CHECK(current.defined());
          out.push_back(current);
        }
        continue;
      }
      case TokenType::At: {
        auto next = tokens.at(i + 1);
        if (next->token_type == TokenType::Identifier) {
          // Match this token.
          i += 1;
          auto tok = Token(current->line, current->column, TokenType::Global, next->data);
          CHECK(tok.defined());
          out.push_back(tok);
        } else {
          CHECK(current.defined());
          out.push_back(current);
        }
        continue;
      }
      case TokenType::Identifier: {
        std::string str = Downcast<tvm::String>(current->data);
        Token tok;
        if (str == "True") {
          auto data = tvm::Integer(1);
          tok = Token(current->line, current->column, TokenType::Boolean, data);
        } else if (str == "False") {
          auto data = tvm::Integer(0);
          tok = Token(current->line, current->column, TokenType::Boolean, data);
        } else if (str == "_") {
          tok = Token(current->line, current->column, TokenType::Underscore);
        } else {
          tok = current;
        }
        out.push_back(tok);
        continue;
      }
      default: {
        out.push_back(current);
        continue;
      }
    }
  }

  return out;
}

std::vector<Token> Tokenize(std::string source) {
  auto tokenizer = Tokenizer(source);
  tokenizer.Tokenize();
  auto tokens = Condense(tokenizer.tokens);
  for (auto token : tokens) {
    CHECK(token.defined());
  }
  return tokens;
}

}  // namespace parser
}  // namespace tvm

#endif  // TVM_PARSER_TOKENIZER_H_
