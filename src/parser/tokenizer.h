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
#include <fstream>
#include <tvm/runtime/object.h>

namespace tvm {
namespace parser {

using namespace runtime;

enum TokenType {
    CommentStart,
    CommentEnd,
    Whitespace,
    Newline,
    StringLiteral,
    Identifier,
    OpenParen,
    CloseParen,
    AtSymbol,
    Percent,
    Keyword,
    Comma,
    Period,
    Equal,
    Semicolon,
    Colon,
    Number,
    Unknown,
    EndOfFile,
};

std::string ToString(const TokenType& token_type) {
    switch (token_type) {
        case TokenType::CommentStart:
            return "CommentStart";
        case TokenType::CommentEnd:
            return "CommentEnd";
        case TokenType::Whitespace:
            return "WhiteSpace";
        case TokenType::Newline:
            return "Newline";
        case TokenType::StringLiteral:
            return "StringLiteral";
        case TokenType::Identifier:
            return "Identifier";
        case TokenType::OpenParen:
            return "OpenParen";
        case TokenType::CloseParen:
            return "CloseParen";
        case TokenType::AtSymbol:
            return "AtSymbol";
        case TokenType::Percent:
            return "Percent";
        case TokenType::Keyword:
            return "Keyword";
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
        case TokenType::Number:
            return "Number";
        case TokenType::Unknown:
            return "Unknown";
        case TokenType::EndOfFile:
            return "EndOfFile";
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
    p->stream << "Token(line=" << node->line << ", column=" << node->column << ", token_type=" << ToString(node->token_type) << ", data=" << node->data << ")";
  });



TVM_REGISTER_NODE_TYPE(TokenNode);

class Token : public ObjectRef {
 public:
  TVM_DLL explicit Token(int line, int column, TokenType token_type, ObjectRef data = ObjectRef());

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


bool IsDigit(char c) {
    return '0' <= c && c <= '9';
}

bool isWhitespace(char c) {
    return ' ' == c || c == '\t' || c == '\n';
}

struct Tokenizer {
    int pos;
    int col;
    int line;
    char next_char;
    const std::string& source;
    std::vector<Token> tokens;

    char Next() {
        char c = this->source.at(this->pos);
        if (c == '\n') {
            this->line += 1;
            this->col += 1;
        } else {
            this->col += 1;
        }
        pos += 1;
        return c;
    }

    bool More() {
        return this->pos < this->source.size();
    }

    char Peek() {
        CHECK(pos < this->source.size());
        return this->source.at(this->pos);
    }

    Token NewToken(TokenType token_type, ObjectRef data = ObjectRef()) {
        return Token(this->line, this->col, token_type, data);
    }

    inline Token TokenizeOnce() {
        auto next = Peek();
        std::cout << next << std::endl;
        if (next == '\n') {
            auto token = NewToken(TokenType::Newline);
            Next();
            return token;
        } else if (this->Peek() == '\r' && this->Peek() == '\n') {
            // fix me
            auto token = NewToken(TokenType::Newline);
            Next();
            return token;
        } else if (next == '"') {
            LOG(FATAL) << "string not working yet";
            return NewToken(TokenType::Unknown);
        } else if (isWhitespace(next)) {
            auto token = NewToken(TokenType::Whitespace);
            Next();
            return token;
        } else if (IsDigit(next)) {
            auto token = NewToken(TokenType::Number);
            std::stringstream ss;
            while (More() && IsDigit(Peek())) {
                ss << Next();
            }
            int value = stoi(ss.str());
            token->data = tvm::Integer(value);
            return token;
        } else if (next == '.') {
            auto token = NewToken(TokenType::Period);
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
        } else if (next == '(') {
            auto token = NewToken(TokenType::OpenParen);
            Next();
            return token;
        } else if (next == ')') {
            auto token = NewToken(TokenType::CloseParen);
            Next();
            return token;
        } else {
            auto token = NewToken(TokenType::Unknown);
            std::stringstream ss;
            while (More() && !isWhitespace(Peek())) {
                ss << Next();
            }
            token->data = tvm::String(ss.str());
            return token;
        }
    }

    void Tokenize() {
        while (this->More()) {
            this->tokens.push_back(TokenizeOnce());
        }
        this->tokens.push_back(NewToken(TokenType::EndOfFile));
    }

    Tokenizer(std::string& source) : pos(0), col(1), line(1), source(source), tokens() {
    }
};

std::vector<Token> Tokenize(std::string source) {
    auto tokenizer = Tokenizer(source);
    tokenizer.Tokenize();
    return tokenizer.tokens;
}

}  // namespace parser
}  // namespace tvm
