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
 * \file parser.cc
 * \brief A parser for TVM IR.
 */
#include <tvm/ir/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/relay/function.h>
#include <tvm/relay/expr.h>
#include <fstream>

#include "./tokenizer.h"

namespace tvm {
namespace parser {

using namespace relay;
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

/*
 * NOTE: The `USE_ANTLR` option in `config.cmake` must be enabled in order for
 * changes in this file to be reflected by the parser.
 * NOTE: All upper-case rules are *lexer* rules and all camel-case rules are *parser* rules.
 */

// grammar Relay;

// SEMVER: 'v0.0.4' ;

// // Lexing
// // comments
// COMMENT : '/*' (COMMENT|.)*? '*/' -> skip;
// WS : [ \t\n\r]+ -> skip;
// LINE_COMMENT : '//' .*? '\n' -> skip;

// fragment ESCAPED_QUOTE : '\\"';
// QUOTED_STRING :   '"' ( ESCAPED_QUOTE | ~('\n'|'\r') )*? '"';

// // operators
// MUL: '*' ;
// DIV: '/' ;
// ADD: '+' ;
// SUB: '-' ;
// LT: '<' ;
// GT: '>' ;
// LE: '<=' ;
// GE: '>=' ;
// EQ: '==' ;
// NE: '!=' ;

// BOOL_LIT
//   : 'True'
//   | 'False'
//   ;

// CNAME: ('_'|LETTER) ('_'|LETTER|DIGIT)* ('.' CNAME)* ;

// // non-negative floats
// fragment PREFLOAT : NAT ('.' NAT)? EXP?; // 1.35, 1.35E-9, 0.3, 4.5, 1, 1e10 3e4

// FLOAT : PREFLOAT 'f';

// // non-negative ints
// NAT: DIGIT+ ;
// fragment EXP: [eE] [+\-]? NAT ; // \- since - means "range" inside [...]

// fragment LETTER: [a-zA-Z];
// fragment DIGIT: [0-9];

// METADATA: 'METADATA:' .*;
// // Parsing



// // Covers both operator and type idents
// generalIdent: CNAME ('.' CNAME)*;
// globalVar: '@' CNAME ;
// localVar: '%' ('_' | CNAME) ;
// graphVar: '%' NAT ;

// exprList: (expr (',' expr)*)?;
// callList
//   : exprList             # callNoAttr
//   | (expr ',')* attrSeq  # callWithAttr
//   ;

// expr
//   // operators
//   : '(' expr ')'                             # paren
//   // function application
//   | expr '(' callList ')'                    # call
//   | '-' expr                                 # neg
//   | expr op=('*'|'/') expr                   # binOp
//   | expr op=('+'|'-') expr                   # binOp
//   | expr op=('<'|'>'|'<='|'>=') expr         # binOp
//   | expr op=('=='|'!=') expr                 # binOp
//   // function definition
//   | func                                     # funcExpr
//   // tuples and tensors
//   | '(' ')'                                  # tuple
//   | '(' expr ',' ')'                         # tuple
//   | '(' expr (',' expr)+ ')'                 # tuple
//   | '[' (expr (',' expr)*)? ']'              # tensor
//   | 'if' '(' expr ')' body 'else' body       # ifElse
//   | matchType expr '{' matchClauseList? '}'  # match
//   | expr '.' NAT                             # projection
//   // sequencing
//   | 'let' var '=' expr ';' expr              # let
//   // sugar for let %_ = expr; expr
//   | expr ';;' expr                           # let
//   | graphVar '=' expr ';' expr               # graph
//   | ident                                    # identExpr
//   | scalar                                   # scalarExpr
//   | meta                                     # metaExpr
//   | QUOTED_STRING                            # stringExpr
//   ;

// func: 'fn' typeParamList? '(' argList ')' ('->' typeExpr)? body ;
// defn
//   : 'def' globalVar typeParamList? '(' argList ')' ('->' typeExpr)? body  # funcDefn
//   | 'extern' 'type' generalIdent typeParamList?                           # externAdtDefn
//   | 'type' generalIdent typeParamList? '{' adtConsDefnList? '}'           # adtDefn
//   ;

// constructorName: CNAME ;

// adtConsDefnList: adtConsDefn (',' adtConsDefn)* ','? ;
// adtConsDefn: constructorName ('(' typeExpr (',' typeExpr)* ')')? ;
// matchClauseList: matchClause (',' matchClause)* ','? ;
// matchClause: pattern '=>' ('{' expr '}' | expr) ;
// // complete or incomplete match, respectively
// matchType : 'match' | 'match?' ;

// patternList: '(' pattern (',' pattern)* ')';
// pattern
//   : '_'                             # wildcardPattern
//   | localVar (':' typeExpr)?        # varPattern
//   | constructorName patternList?    # constructorPattern
//   | patternList                     # tuplePattern
//   ;

// adtCons: constructorName adtConsParamList? ;
// adtConsParamList: '(' adtConsParam (',' adtConsParam)* ')' ;
// adtConsParam: localVar | constructorName ;

// argList
//   : varList             # argNoAttr
//   | (var ',')* attrSeq  # argWithAttr
//   ;

// varList: (var (',' var)*)? ;
// var: localVar (':' typeExpr)? ;

// attrSeq: attr (',' attr)* ;
// attr: CNAME '=' expr ;

// typeExpr
//   : '(' ')'                                                                # tupleType
//   | '(' typeExpr ')'                                                       # typeParen
//   | '(' typeExpr ',' ')'                                                   # tupleType
//   | '(' typeExpr (',' typeExpr)+ ')'                                       # tupleType
//   | generalIdent typeParamList                                             # typeCallType
//   | generalIdent                                                           # typeIdentType
//   | 'Tensor' '[' shapeList ',' typeExpr ']'                                # tensorType
//   | 'fn' typeParamList? '(' (typeExpr (',' typeExpr)*)? ')' '->' typeExpr  # funcType
//   | '_'                                                                    # incompleteType
//   ;

// typeParamList: '[' typeExpr (',' typeExpr)* ']' ;

// shapeList
//   : '(' ')'
//   | '(' shape (',' shape)+ ')'
//   | shape
//   ;

// meta : 'meta' '[' CNAME ']' '[' NAT ']';

// shape
//   : meta           # metaShape
//   | '(' shape ')'  # parensShape
//   | NAT            # intShape
//   ;

// body: '{' expr '}' ;

// scalar
//   : FLOAT     # scalarFloat
//   | NAT       # scalarInt
//   | BOOL_LIT  # scalarBool
//   ;

// ident
//   : generalIdent
//   | globalVar
//   | localVar
//   | graphVar
//   ;

struct GlobalFunc {
    GlobalVar global;
    Function function;
};

struct Definitions {
    std::vector<GlobalFunc> funcs;
    std::vector<TypeData> types;
};

struct SemVer {
    int major;
    int minor;
    int patch;
};

class MetaRefExpr;
class MetaRefExprNode : public TempExprNode {
 public:
    std::string type_key;
    uint64_t node_index;

  void VisitAttrs(tvm::AttrVisitor* v) {
  }

  Expr Realize() const final {
      return Expr();
  }

  static constexpr const char* _type_key = "relay.MetaRefExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(MetaRefExprNode, TempExprNode);
};

class MetaRefExpr : public TempExpr {
 public:
  /*!
   * \brief The constructor
   * \param expr The original relay expression.
   * \param kind The annotation kind.
   */
  TVM_DLL MetaRefExpr(std::string type_key, uint64_t node_index);

  TVM_DEFINE_OBJECT_REF_METHODS(MetaRefExpr, TempExpr, MetaRefExprNode);
};

MetaRefExpr::MetaRefExpr(std::string type_key, uint64_t node_index) {
  auto rnode = make_object<MetaRefExprNode>();
  rnode->type_key = type_key;
  rnode->node_index = node_index;
  data_ = std::move(rnode);
}

struct Parser {
    int pos;
    std::vector<Token> tokens;
    bool ignore_whitespace;

    std::unordered_map<int, Expr> graph_ctx;

    Parser(std::vector<Token> tokens) : pos(0), tokens(tokens) {}

    void DisplayNextN(int n) {
        std::cout << "remaining tokens: " << std::endl;
        auto bound = std::min(pos + n, (int)tokens.size());
        for (int i = 0; i < bound - pos; i++) {
            std::cout << tokens[pos + i] << std::endl;
        }
    }
    // // A Relay program is a list of global definitions or an expression.
// prog: SEMVER (defn* | expr) METADATA? EOF ;
    Token Peek() {
        while (ignore_whitespace &&
               tokens.at(pos)->token_type == TokenType::Whitespace) {
            pos++;
        }

        return this->tokens.at(pos);
    }

    void Consume(const TokenType& token) {
        std::cout << token << std::endl;
        if (tokens[pos]->token_type != token) {
            throw std::runtime_error("expected a " + ToString(token)
                + " found " + ToString(Peek()->token_type)
                + " at " + std::to_string(tokens[pos]->line) + ":"
                + std::to_string(tokens[pos]->column));
        }
        pos++;
    }

    Token Match(const TokenType& token_type) {
        auto tok = Peek();
        Consume(token_type);
        return tok;
    }

    bool WhenMatch(const TokenType& token_type) {
        if (Peek()->token_type == token_type) {
            Consume(token_type);
            return true;
        } else {
            std::cout << "doesn't match"
                << ToString(Peek()->token_type)
                << ToString(token_type);
            return false;
        }
    }

    void AddGraphBinding(const Token& token, const Expr& expr) {
        auto graph_no = token.ToNumber();
        this->graph_ctx.insert({graph_no, expr});
    }

    Expr LookupGraphBinding(const Token& token) {
        auto graph_no = token.ToNumber();
        std::cout << "graph_no" << graph_no;
        std::cout << this->graph_ctx.size() << std::endl;
        return this->graph_ctx.at(graph_no);
    }

    NDArray NumberToNDArray(const Token& token_type) {
        DLContext ctx({ .device_type = DLDeviceType::kDLCPU, .device_id = 0 });
        auto dtype = String2DLDataType("int32");
        auto data = NDArray::Empty({}, dtype, ctx);
        auto array = reinterpret_cast<int32_t*>(data->data);
        // revisit this, literal node issue.
        int64_t value = Downcast<Integer>(token_type->data);
        array[0] = (int32_t)value;
        return data;
    }

    [[noreturn]] void ParseError(const Token& token, const std::string& msg) {
        throw std::runtime_error(msg);
    }

    IRModule ParseModule() {
        auto version = ParseSemVer();
        auto defs = ParseDefinitions();
        auto metadata = ParseMetadata();
        Consume(TokenType::EndOfFile);
        return IRModule();
    }

    SemVer ParseSemVer() {
        Consume(TokenType::Unknown);
        return SemVer { .major = 0, .minor = 0, .patch = 0 };
    }


    Definitions ParseDefinitions() {
        Definitions defs;

        return defs;
    }

    template<typename R>
    R Bracket(TokenType open, TokenType close, std::function<R()> parser) {
        Consume(open);
        R result;
        if (WhenMatch(close)) {
            return result;
        } else {
            result = parser();
        }
    }

    template<typename R>
    R Parens(std::function<R()> parser) {
        return Bracket(open, close, parser);
    }

    Expr ParseExpr() {
        return ConsumeWhitespace<Expr>([this] {
            auto next = Peek();
            switch (next->token_type) {
            case TokenType::Graph: {
                Consume(TokenType::Graph);
                if (WhenMatch(TokenType::Equal)) {
                    return ParseGraphExpr(next);
                } else {
                    return LookupGraphBinding(next);
                }
            }
            case TokenType::Number: {
                Consume(TokenType::Number);
                auto data = NumberToNDArray(next);
                Expr e = Constant(data);
                return e;
            }
            case TokenType::OpenParen: {
                Consume(TokenType::OpenParen);
                // parse '(' ')'
                if (WhenMatch(TokenType::CloseParen)) {
                    Expr e = Tuple(Array<Expr>());
                    return e;
                } else {
                    auto expr = ParseExpr();
                    // parse '(' expr ')'
                    if (WhenMatch(TokenType::CloseParen)) {
                        return expr;
                    // parse '( expr ',' * ')'
                    } else if (WhenMatch(TokenType::Comma)) {
                        Array<Expr> exprs = { expr };
                        while (true) {
                            if (WhenMatch(TokenType::CloseParen)) {
                                break;
                            } else {
                                auto expr = ParseExpr();
                                WhenMatch(TokenType::Comma);
                                exprs.push_back(expr);
                            }
                        }
                        return static_cast<Expr>(Tuple(exprs));
                    }
                }
            }
            default:
                ParseError(next,
                    "expected an expression found  " + ToString(next->token_type)
                    + std::to_string(next->line) + ":" + std::to_string(next->column));
            }
        });
    }

    Expr ParseGraphExpr(const Token& graph_var) {
        std::cout << "IN HERE";
        DisplayNextN(10);
        auto val = ParseExpr();
        AddGraphBinding(graph_var, val);
        Consume(TokenType::Semicolon);
        auto body = ParseExpr();
        return body;
    }

    template<typename R>
    R ConsumeWhitespace(std::function<R()> func) {
        auto old = this->ignore_whitespace;
        this->ignore_whitespace = true;
        while (tokens[pos]->token_type == TokenType::Whitespace) {
            pos++;
        }
        auto res = func();
        this->ignore_whitespace = old;
        return res;
    }

    ObjectRef ParseMetadata() {
        return ObjectRef();
    }
};

IRModule ParseModule(std::string file_name, std::string file_content) {
    auto tokens = Tokenize(file_content);
    for (auto token : tokens) {
        std::cout << token << std::endl;
    }
    Parser parser(tokens);
    return parser.ParseModule();
}


Expr ParseExpr(std::string file_name, std::string file_content) {
    auto tokens = Tokenize(file_content);
    for (auto token : tokens) {
        std::cout << token << std::endl;
    }
    Parser parser(tokens);
    return parser.ParseExpr();
}

TVM_REGISTER_GLOBAL("parser.ParseModule")
.set_body_typed([](std::string file_name, std::string file_content) {
  return ParseModule(file_name, file_content);
});

TVM_REGISTER_GLOBAL("parser.ParseExpr")
.set_body_typed([](std::string file_name, std::string file_content) {
  return ParseExpr(file_name, file_content);
});



}  // namespace parser
}  // namespace tvm
