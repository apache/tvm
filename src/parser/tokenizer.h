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
#include <tvm/runtime/container.h>

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
    Keyword,
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
    Bang,
    Let,
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
        case TokenType::Bang:
            return "Bang";
        case TokenType::Let:
            return "Let";
        case TokenType::Boolean:
            return "Boolean";
        case TokenType::Unknown:
            return "Unknown";
        case TokenType::EndOfFile:
            return "EndOfFile";
        case TokenType::Null:
            return "Null";
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

  static Token Null();
  int64_t ToNumber() const;
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

Token Token::Null() {
    return Token(0, 0, TokenType::Null);
}

int64_t Token::ToNumber() const {
    return Downcast<tvm::Integer>(this->operator->()->data);
}

bool IsDigit(char c) {
    return '0' <= c && c <= '9';
}

bool IsWhitespace(char c) {
    return ' ' == c || c == '\t' || c == '\n';
}

bool IsNumeric(char c) {
    return (IsDigit(c) || c == '.' || c == 'e' || c == '-' || c == '+' || c == 'E') && !IsWhitespace(c);
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

    enum CommentParserState {
        Proceed,
        Forward,
        Backward,
    };

    void MatchComment(std::string& buffer) {
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
                    buffer += Next();
                    continue;
                }
                case CommentParserState::Forward: {
                    if (Peek() == '*') {
                        nesting += 1;
                        buffer += Next();
                    }
                    state = CommentParserState::Proceed;
                    continue;
                }
                case CommentParserState::Backward: {
                    if (Peek() == '/') {
                        nesting -= 1;
                        if (nesting == 0) {
                            Next();
                            buffer.pop_back();
                            return;
                        } else {
                            buffer += Next();
                            state = CommentParserState::Proceed;
                        };
                    }
                    continue;
                }
            }
        }
    }

    Token ParseNumber(bool is_pos, std::string number) {
        CHECK(number.size() > 0)
            << "an empty string is an invalid number";

        try {
            auto token = NewToken(TokenType::Integer);
            size_t index = 0;
            int value = std::stoi(number, &index);
            std::cout << "Index" << index << std::endl;
            std::cout << "Size" << number.size() << std::endl;
            if (number.size() > index) {
                std::cout << "inside" << std::endl;
                throw std::invalid_argument("floating point");
            }
            value = is_pos ? value : -value;
            token->data = tvm::Integer(value);
            return token;
        } catch (const std::invalid_argument& ia) {
            std::cout << "in float" << std::endl;
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
            std::cout << "Number: " << ss.str() << std::endl;

            // Remove trailing floating point prefix.
            if (More() && Peek() == 'f') {
                Next();
            }

            return ParseNumber(!is_neg, ss.str());
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
        } else if (next == '!') {
            auto token = NewToken(TokenType::Bang);
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
                MatchComment(comment);
                auto token = NewToken(TokenType::Comment, tvm::String(comment));
                return token;
            } else {
                return NewToken(TokenType::Division);
            }
        } else {
            auto token = NewToken(TokenType::Unknown);
            std::stringstream ss;
            while (More() && !IsWhitespace(Peek())) {
                ss << Next();
            }
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

    Tokenizer(std::string& source) : pos(0), col(1), line(1), source(source), tokens() {
    }
};

std::vector<Token> Condense(const std::vector<Token>& tokens) {
    std::vector<Token> out;

    for (auto i = 0; i < tokens.size(); i++) {
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
            case TokenType::Unknown: {
                std::string str = Downcast<tvm::String>(current->data);
                Token tok;
                if (str == "True") {
                    auto data = tvm::Integer(1);
                    tok = Token(current->line, current->column, TokenType::Boolean, data);
                } else if (str == "False") {
                    auto data = tvm::Integer(0);
                    tok = Token(current->line, current->column, TokenType::Boolean, data);
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
