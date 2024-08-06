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
 * \file tokenizer.h
 * \brief A parser for TVM IR.
 */
#ifndef TVM_RELAY_PARSER_TOKENIZER_H_
#define TVM_RELAY_PARSER_TOKENIZER_H_

#include <tvm/node/serialization.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../support/scalars.h"
#include "./meta_ref.h"
#include "./token.h"

namespace tvm {
namespace relay {

// trim from start (in place)
static inline void ltrim(std::string& s) {  // NOLINT(*)
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string& s) {  // NOLINT(*)
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(),
          s.end());
}

inline bool IsDigit(char c) { return '0' <= c && c <= '9'; }

inline bool IsWhitespace(char c) { return ' ' == c || c == '\t' || c == '\n'; }

inline bool IsNumeric(char c) {
  return (IsDigit(c) || c == '.' || c == 'e' || c == '-' || c == '+' || c == 'E') &&
         !IsWhitespace(c);
}

inline bool IsIdentLetter(char c) {
  return '_' == c || c == '/' || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

inline bool IsIdent(char c) { return IsIdentLetter(c) || IsDigit(c); }

static std::unordered_map<std::string, TokenType> KEYWORD_TABLE = {
    {"let", TokenType::kLet},          {"fn", TokenType::kFn},
    {"def", TokenType::kDefn},         {"if", TokenType::kIf},
    {"else", TokenType::kElse},        {"type", TokenType::kTypeDef},
    {"match", TokenType::kMatch},      {"extern", TokenType::kExtern},
    {"free_var", TokenType::kFreeVar}, {"ref", TokenType::kRef},
    {"ref_read", TokenType::kRefRead}, {"ref_write", TokenType::kRefWrite}};

struct Tokenizer {
  DiagnosticContext diag_ctx;
  const SourceName& source_name;

  size_t pos;
  int col;
  int line;
  char next_char;
  String source;
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
    ICHECK(pos < this->source.size());
    return this->source.at(this->pos);
  }

  Token NewToken(TokenType token_type, ObjectRef data = ObjectRef(), int lines = 0, int cols = 1) {
    auto span =
        Span(this->source_name, this->line, this->line + lines, this->col, this->col + cols);
    return Token(span, token_type, data);
  }

  Span SpanFrom(int line, int column) {
    int end_line = this->line;
    int end_column = this->col;
    return Span(this->source_name, line, end_line, column, end_column);
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

    while (More()) {
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
            }
          }

          buffer->operator+=(Next());
          state = CommentParserState::Proceed;
          continue;
        }
      }
    }
  }

  Token ParseNumber(bool is_pos, bool is_float, std::string number) {
    ICHECK(number.size() > 0) << "an empty string is an invalid number";

    Token token = NewToken(is_float ? TokenType::kFloat : TokenType::kInteger);
    size_t suffix_pos = number.rfind(is_float ? 'f' : 'i');
    if (suffix_pos == std::string::npos) {
      suffix_pos = number.size();
    }
    std::string literal_text = number.substr(0, suffix_pos);
    std::string suffix;
    if (suffix_pos < number.size()) {
      suffix = number.substr(suffix_pos + 1, number.size() - suffix_pos);
    }
    int width = 32;

    if (suffix.size()) {
      try {
        width = std::stoi(suffix);
      } catch (const std::invalid_argument& err) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid numeric suffix `" << suffix << "`");
      } catch (const std::out_of_range& err) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid numeric suffix `" << suffix << "`");
      }
    }

    if (is_float) {
      double value = 0.0;
      size_t index = 0;
      try {
        value = stod(literal_text, &index);
      } catch (const std::invalid_argument& err) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid floating point number `" << literal_text << "`");
      } catch (const std::out_of_range& err) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid floating point number `" << literal_text << "`");
      }
      if (index < literal_text.size()) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid floating point number `" << literal_text << "`");
      }
      value = is_pos ? value : -value;
      token->data = support::ValueToFloatImm(value, width);
      if (!token->data.defined()) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "floating point number `" << literal_text
                            << "` unrepresentable in width " << width);
        token->data = support::ValueToFloatImm(0.0, width);
      }
    } else {
      int64_t value = 0;
      size_t index = 0;
      try {
        value = std::stoll(literal_text, &index);
      } catch (const std::invalid_argument& err) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid integer number `" << literal_text << "`");
      } catch (const std::out_of_range& err) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid integer number `" << literal_text << "`");
      }
      if (index < literal_text.size()) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "invalid integer number `" << literal_text << "`");
      }
      value = is_pos ? value : -value;
      token->data = support::ValueToIntImm(value, width);
      if (!token->data.defined() && suffix.empty()) {
        // Without any i suffix the legacy behavior was to default to int64 if out of range
        // for int32.
        width = 64;
        token->data = support::ValueToIntImm(value, width);
      }
      if (!token->data.defined()) {
        this->diag_ctx.Emit(Diagnostic::Error(token->span)
                            << "integer number `" << literal_text << "` unrepresentable in width "
                            << width);
        token->data = support::ValueToIntImm(0, width);
      }
    }

    return token;
  }

  Token ParseNumber(bool is_pos) {
    std::stringstream ss;
    while (More() && IsNumeric(Peek())) {
      ss << Next();
    }

    bool is_float = false;
    if (More() && (Peek() == 'f' || Peek() == 'i')) {
      is_float = Peek() == 'f';
      // Capture trailing width suffix
      ss << Next();
      while (More() && IsNumeric(Peek())) {
        ss << Next();
      }
    }
    return ParseNumber(is_pos, is_float, ss.str());
  }

  bool MatchString(const std::string& string) {
    int start = this->pos;

    for (auto c : string) {
      if (Peek() != c) {
        this->pos = start;
        return false;
      } else {
        Next();
      }
    }

    return true;
  }

  Token TokenizeMetaRef() {
    int line = this->line;
    int column = this->col;

    std::stringstream type_key;
    while (More() && Peek() != ']') {
      type_key << Next();
    }
    ICHECK_EQ(Peek(), ']');
    Next();

    ICHECK_EQ(Peek(), '[');
    Next();
    std::stringstream str_index;
    while (More() && Peek() != ']') {
      str_index << Next();
    }
    ICHECK_EQ(Peek(), ']');
    Next();
    // todo: add error handling around bad indices
    auto index = ParseNumber(true, false, str_index.str()).ToNumber();
    auto span = SpanFrom(line, column);
    return Token(span, TokenType::kMetaReference, MetaRef(type_key.str(), index));
  }

  Token TokenizeAttr() {
    int line = this->line;
    int column = this->col;
    Next();
    if (Peek() == '[') {
      Next();
      std::stringstream raw_attribute;

      while (More() && Peek() != ']') {
        raw_attribute << Next();
      }

      ICHECK_EQ(Next(), ']');

      auto attribute = raw_attribute.str();
      // Clean up the white-space on both sides.
      ltrim(attribute);
      rtrim(attribute);

      // Metadata can only appear at the bottom of a file and goes to EOF.
      if (attribute == "metadata") {
        std::stringstream metadata;
        while (More()) {
          metadata << Next();
        }
        ObjectRef metadata_map = tvm::LoadJSON(metadata.str());
        auto span = SpanFrom(line, column);
        return Token(span, TokenType::kMetadata, metadata_map);
      }
      if (attribute.rfind("version", 0) == 0) {
        std::string version = attribute.substr(attribute.find("=") + 1);
        ltrim(version);
        rtrim(version);
        auto span = SpanFrom(line, column);
        return Token(span, TokenType::kVersion, tvm::String(version));
      } else {
        // TOOD(@jroesch): maybe make this a warning an continue parsing?
        auto span = SpanFrom(line, column);
        this->diag_ctx.EmitFatal(Diagnostic::Error(span) << "unsupported attribute " << attribute);
        return Token();
      }
    } else {
      auto span = SpanFrom(line, column);
      this->diag_ctx
          .EmitFatal(Diagnostic::Error(span)
                     << "`#` denotes the start of an attribute can only be followed by `[`"
                     << " found `" << Peek() << "`");
      return Token();
    }
  }

  inline Token TokenizeOnce() {
    int line = this->line;
    int col = this->col;
    auto next = Peek();
    VLOG(9) << "tvm::relay::TokenizeOnce: next=" << next;
    if (next == '\n') {
      auto token = NewToken(TokenType::kNewline);
      Next();
      return token;
    } else if (next == '\r') {
      Next();
      if (More() && Peek() == '\n') {
        auto token = NewToken(TokenType::kNewline);
        return token;
      } else {
        auto span = SpanFrom(line, col);
        this->diag_ctx.EmitFatal(
            Diagnostic::Error(span)
            << "\\r carriage returns must be followed by a \\n in the TVM text format");
        return Token();
      }
    } else if (next == '"') {
      // TODO(@jroesch): Properly tokenize escape sequences in strings.
      // see https://github.com/apache/tvm/issues/6153.
      Next();
      std::stringstream string_content;
      while (More() && Peek() != '"') {
        string_content << Next();
      }
      Next();
      return NewToken(TokenType::kStringLiteral, tvm::String(string_content.str()));
    } else if (IsWhitespace(next)) {
      auto token = NewToken(TokenType::kWhitespace);
      Next();
      return token;
    } else if (next == '-') {
      int negs = 0;
      while (More() && Peek() == '-') {
        Next();
        negs++;
      }
      bool is_neg = negs % 2 == 1;
      if (More() && IsDigit(Peek())) {
        return ParseNumber(!is_neg);
      } else if (More() && MatchString("inff")) {
        return ParseNumber(!is_neg, true, "inff");
      } else {
        // If there isn't a number right after either,
        // this is really slow for lexing, should replace
        // with multi-token return or something.
        pos = pos - (negs - 1);
        return NewToken(TokenType::kMinus);
      }
    } else if (IsDigit(next)) {
      return ParseNumber(true);
    } else if (MatchString("inff")) {
      return ParseNumber(true, true, "inff");
    } else if (next == '.') {
      auto token = NewToken(TokenType::kPeriod);
      Next();
      return token;
    } else if (next == ',') {
      auto token = NewToken(TokenType::kComma);
      Next();
      return token;
    } else if (next == '=') {
      auto token = NewToken(TokenType::kEqual);
      Next();
      return token;
    } else if (next == ';') {
      auto token = NewToken(TokenType::kSemicolon);
      Next();
      return token;
    } else if (next == ':') {
      auto token = NewToken(TokenType::kColon);
      Next();
      return token;
    } else if (next == '(') {
      auto token = NewToken(TokenType::kOpenParen);
      Next();
      return token;
    } else if (next == ')') {
      auto token = NewToken(TokenType::kCloseParen);
      Next();
      return token;
    } else if (next == '+') {
      auto token = NewToken(TokenType::kPlus);
      Next();
      return token;
    } else if (next == '*') {
      auto token = NewToken(TokenType::kStar);
      Next();
      return token;
    } else if (next == '<') {
      auto token = NewToken(TokenType::kLAngle);
      Next();
      return token;
    } else if (next == '>') {
      auto token = NewToken(TokenType::kRAngle);
      Next();
      return token;
    } else if (next == '{') {
      auto token = NewToken(TokenType::kLCurly);
      Next();
      return token;
    } else if (next == '}') {
      auto token = NewToken(TokenType::kRCurly);
      Next();
      return token;
    } else if (next == '[') {
      auto token = NewToken(TokenType::kLSquare);
      Next();
      return token;
    } else if (next == ']') {
      auto token = NewToken(TokenType::kRSquare);
      Next();
      return token;
    } else if (next == '!') {
      auto token = NewToken(TokenType::kBang);
      Next();
      return token;
    } else if (next == '@') {
      auto token = NewToken(TokenType::kAt);
      Next();
      return token;
    } else if (next == '?') {
      auto token = NewToken(TokenType::kQuestion);
      Next();
      return token;
    } else if (MatchString("meta[")) {
      return TokenizeMetaRef();
    } else if (next == '#') {
      return TokenizeAttr();
    } else if (next == '%') {
      auto token = NewToken(TokenType::kPercent);
      Next();

      std::stringstream number;
      while (More() && IsDigit(Peek())) {
        number << Next();
      }

      auto number_str = number.str();
      if (number_str.size()) {
        auto num_tok = ParseNumber(true, false, number_str);
        auto span = SpanFrom(token->span->line, token->span->column);
        token = Token(span, TokenType::kGraph, num_tok->data);
      }

      return token;
    } else if (next == '/') {
      Next();
      if (Peek() == '/') {
        auto token = NewToken(TokenType::kLineComment);
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
        auto token = NewToken(TokenType::kComment, tvm::String(comment));
        return token;
      } else {
        return NewToken(TokenType::kDivision);
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

        if (token_type == TokenType::kMatch) {
          if (More() && Peek() == '?') {
            Next();
            token_type = TokenType::kPartialMatch;
          }
        }
      } else {
        token_type = TokenType::kIdentifier;
      }

      auto span = SpanFrom(line, col);
      return Token(span, token_type, tvm::String(ss.str()));
    } else {
      std::stringstream ss;
      while (More() && !IsWhitespace(Peek())) {
        ss << Next();
      }
      auto token = NewToken(TokenType::kUnknown);
      token->data = tvm::String(ss.str());
      return token;
    }
  }

  void Tokenize() {
    VLOG(9) << "tvm::relay::Tokenize";
    while (this->More()) {
      auto token = TokenizeOnce();
      ICHECK(token.defined());
      this->tokens.push_back(token);
    }
    this->tokens.push_back(NewToken(TokenType::kEndOfFile));
  }

  explicit Tokenizer(const DiagnosticContext& ctx, const Source& source)
      : diag_ctx(ctx),
        source_name(source->source_name),
        pos(0),
        col(1),
        line(1),
        source(source->source),
        tokens() {}
};

inline std::vector<Token> Condense(const std::vector<Token>& tokens, Token* table) {
  std::vector<Token> out;
  bool found_metadata = false;

  for (size_t i = 0; i < tokens.size(); i++) {
    auto current = tokens.at(i);
    switch (current->token_type) {
      case TokenType::kMetadata: {
        if (!found_metadata) {
          found_metadata = true;
          *table = current;
        } else {
          LOG(FATAL) << "duplicate metadata section";
        }
        continue;
      }
      case TokenType::kPercent: {
        auto next = tokens.at(i + 1);
        if (next->token_type == TokenType::kIdentifier) {
          // Match this token.
          i += 1;
          // TODO(@jroesch): merge spans
          auto tok = Token(current->span, TokenType::kLocal, next->data);
          ICHECK(tok.defined());
          out.push_back(tok);
        } else if (next->token_type == TokenType::kInteger) {
          i += 1;
          auto tok = Token(current->span, TokenType::kGraph, next->data);
          ICHECK(tok.defined());
          out.push_back(tok);
        } else {
          ICHECK(current.defined());
          out.push_back(current);
        }
        continue;
      }
      case TokenType::kAt: {
        auto next = tokens.at(i + 1);
        if (next->token_type == TokenType::kIdentifier) {
          // Match this token.
          i += 1;
          // TODO(@jroesch): merge spans
          auto tok = Token(current->span, TokenType::kGlobal, next->data);
          ICHECK(tok.defined());
          out.push_back(tok);
        } else {
          ICHECK(current.defined());
          out.push_back(current);
        }
        continue;
      }
      case TokenType::kIdentifier: {
        std::string str = Downcast<tvm::String>(current->data);
        Token tok;
        // TODO(@jroesch): merge spans
        if (str == "True") {
          auto data = tvm::Integer(1);
          tok = Token(current->span, TokenType::kBoolean, data);
        } else if (str == "False") {
          auto data = tvm::Integer(0);
          tok = Token(current->span, TokenType::kBoolean, data);
        } else if (str == "_") {
          tok = Token(current->span, TokenType::kUnderscore);
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

inline std::pair<std::vector<Token>, Token> Tokenize(const DiagnosticContext& ctx,
                                                     const Source& source) {
  auto tokenizer = Tokenizer(ctx, source);
  tokenizer.Tokenize();
  Token meta_table(Span(), TokenType::kUnknown, ObjectRef());
  auto tokens = Condense(tokenizer.tokens, &meta_table);
  for (auto token : tokens) {
    ICHECK(token.defined());
  }
  return {tokens, meta_table};
}

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PARSER_TOKENIZER_H_
