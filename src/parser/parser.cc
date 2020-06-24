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
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <fstream>

#include "./tokenizer.h"

namespace tvm {
namespace parser {

using namespace relay;
using Expr = relay::Expr;

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
  GlobalFunc() : global(), function() {}
  GlobalFunc(GlobalVar global, Function function) : global(global), function(function) {}
  GlobalFunc(const GlobalFunc& gfunc) {
    this->global = gfunc.global;
    this->function = gfunc.function;
  }
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

  void VisitAttrs(tvm::AttrVisitor* v) {}

  Expr Realize() const final { return Expr(); }

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

struct Rule {
  std::vector<TokenType> tokens;
  int precedence;
  int arity;
  tvm::Op op;
  bool left_assoc;

  Rule() : tokens(), precedence(0), arity(0), op(tvm::Op()), left_assoc(false) {}

  Rule(std::vector<TokenType> tokens, tvm::Op op, int precedence, int arity = 2, bool left_assoc = false)
      : tokens(tokens), precedence(precedence), arity(arity), op(op), left_assoc(false) {}

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

  OperatorTable(std::vector<Rule> rules) : rules(rules), this_is_a_hack() {
    for (auto rule : rules) {
      std::stringstream key;
      for (auto token : rule.tokens) {
        key << ToString(token);
      }
      this->this_is_a_hack.insert({ key.str(), rule });
    }
  }
};

struct Scope {
  std::unordered_map<std::string, Var> name_map;
  Scope() : name_map() {}
};

struct Parser {
  int pos;
  std::vector<Token> tokens;
  OperatorTable op_table;
  bool ignore_whitespace;

  std::unordered_map<int, Expr> graph_ctx;
  std::vector<Scope> scopes = { Scope() };

  Parser(std::vector<Token> tokens, OperatorTable op_table)
      : pos(0), tokens(tokens), op_table(op_table) {
        DisplayNextN(100);
  }

  void DisplayNextN(int n) {
    std::cout << "remaining tokens: " << std::endl;
    auto bound = std::min(pos + n, (int)tokens.size());
    for (int i = 0; i < bound - pos; i++) {
      std::cout << tokens[pos + i] << std::endl;
    }
  }

  Token Peek() {
    // For now we ignore all whitespace tokens and comments.
    // We can tweak this behavior later to enable white space sensitivity in the parser.
    DisplayNextN(100);
    while (pos < tokens.size() &&
           ignore_whitespace && (tokens.at(pos)->token_type == TokenType::Whitespace ||
                                 tokens.at(pos)->token_type == TokenType::Newline ||
                                 tokens.at(pos)->token_type == TokenType::LineComment ||
                                 tokens.at(pos)->token_type == TokenType::Comment)) {
      std::cout << "pos: " << pos << std::endl;
      std::cout << "tokens: " << tokens.size() << std::endl;
      pos++;
    }

    if (pos < tokens.size()) {
      return Token(this->tokens.at(pos));
    } else {
      return Token::Null();
    }
  }

  void Consume(const TokenType& token) {
    std::cout << token << std::endl;
    if (tokens[pos]->token_type != token) {
      throw std::runtime_error(
          "expected a " + ToString(token) + " found " + ToString(Peek()->token_type) + " at " +
          std::to_string(tokens[pos]->line) + ":" + std::to_string(tokens[pos]->column));
    }
    pos++;
  }

  Token Match(const TokenType& token_type) {
    auto tok = Peek();
    Consume(token_type);
    return tok;
  }

  bool WhenMatch(const TokenType& token_type) {
    std::cout << "token_type: " << token_type << std::endl;
    if (Peek()->token_type == token_type) {
      Consume(token_type);
      return true;
    } else {
      std::cout << "doesn't match" << ToString(Peek()->token_type) << ToString(token_type);
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

  Var BindVar(std::string name, relay::Type type_annotation) {
    auto var = Var(name, type_annotation);
    this->scopes.back().name_map.insert({name, var});
    return var;
  }

  Var LookupVarByString(const std::string& var) {
    for (auto scope = this->scopes.rbegin(); scope != this->scopes.rend(); scope++) {
      auto it = scope->name_map.find(var);
      if (it != scope->name_map.end()) {
        return it->second;
      }
    }
    LOG(FATAL) << "foo";
  }

  void PushScope() {
    this->scopes.push_back(Scope());
  }

  void PopScope(int n) {
    for (int i = 0; i < n; i++) {
      this->scopes.pop_back();
    }
  }

  NDArray NumberToNDArray(const Token& token) {
    if (token->token_type == TokenType::Integer) {
      DLContext ctx({.device_type = DLDeviceType::kDLCPU, .device_id = 0});
      auto dtype = String2DLDataType("int32");
      auto data = NDArray::Empty({}, dtype, ctx);
      auto array = reinterpret_cast<int32_t*>(data->data);
      // revisit this, literal node issue.
      int64_t value = Downcast<tvm::Integer>(token->data);
      array[0] = (int32_t)value;
      return data;
    } else if (token->token_type == TokenType::Float) {
      DLContext ctx({.device_type = DLDeviceType::kDLCPU, .device_id = 0});
      auto dtype = String2DLDataType("float32");
      auto data = NDArray::Empty({}, dtype, ctx);
      auto array = reinterpret_cast<float*>(data->data);
      // revisit this, literal node issue.
      float value = Downcast<tvm::FloatImm>(token->data)->value;
      std::cout << "value: " << value << std::endl;
      array[0] = value;
      return data;
    } else {
      throw "foo";
    }
  }

  NDArray BooleanToNDarray(bool value) {
    DLContext ctx({.device_type = DLDeviceType::kDLCPU, .device_id = 0});
    auto dtype = String2DLDataType("bool");
    auto data = NDArray::Empty({}, dtype, ctx);
    auto array = reinterpret_cast<bool*>(data->data);
    array[0] = value;
    return data;
  }

  [[noreturn]] void ParseError(const Token& token, const std::string& msg) {
    throw std::runtime_error(msg);
  }

  IRModule ParseModule() {
    // Parse the semver header at the top of the module.
    auto _version = ParseSemVer();
    // Parse the definitions.
    auto defs = ParseDefinitions();
    // Parse the metadata section at the end.
    auto metadata = ParseMetadata();
    Match(TokenType::EndOfFile);
    return IRModule();
  }

  SemVer ParseSemVer() {
    // Consume(TokenType::Unknown);
    return SemVer{.major = 0, .minor = 0, .patch = 0};
  }

  Definitions ParseDefinitions() {
    Definitions defs;

    while (true) {
     auto next = Peek();
     switch (next->token_type) {
        case TokenType::Defn: {
          Consume(TokenType::Defn);
          auto tok = Match(TokenType::Global);
          // Todo: add to global parsing context.
          auto global = GlobalVar(tok.ToString());
          auto func = ParseFunctionDef();
          defs.funcs.push_back(GlobalFunc(global, func));
        }
        default:
          return defs;
      }
    }
  }

  template <typename R>
  R Bracket(TokenType open, TokenType close, std::function<R()> parser) {
    Match(open);
    R result = parser();
    Match(close);
    return result;
  }

  template <typename R>
  R Parens(std::function<R()> parser) {
    return Bracket(TokenType::OpenParen, TokenType::CloseParen, parser);
  }

  template <typename R>
  R Block(std::function<R()> parser) {
    return Bracket(TokenType::LCurly, TokenType::RCurly, parser);
  }

  Expr ParseBindingExpr() {
    // We use a loop here so that the stack depth
    // does not grow linearly with a sequence of
    // graph or let bindings.
    //
    // Assuming we start at call depth k, we will
    // enter k + c call frames to parse the RHS
    // of the bindings where `c` is the depth
    // of recursion needed by RHS.
    //
    // If RHS is a call expresssion the c=1.
    //
    // Once we have parsed the RHS we will be
    // back at depth K, and will return to
    // this loop header to parse another
    // graph or let binding.
    //
    // This ensures for n sequential bindings
    // the call depth will be the same before
    // and after parsing the n bindings.
    std::vector<std::pair<Var, Expr>> bindings;
    int scopes = 0;

    while (true) {
      auto next = Peek();
      if (next->token_type == TokenType::Graph) {
        Consume(TokenType::Graph);
        if (WhenMatch(TokenType::Equal)) {
          auto val = this->ParseExpr();
          AddGraphBinding(next, val);
          Consume(TokenType::Semicolon);
        } else {
          // This case is a little weird to put here,
          // if we don't find an equal right after
          // a graph expression it is simply a reference
          // to the graph expression and not a binding
          // but for now we handle the case here.
          return LookupGraphBinding(next);
        }
      } else if (next->token_type == TokenType::Let) {
        // Parse the 'let'.
        Consume(TokenType::Let);

        // Parse the local '%<id>'.
        auto local_tok = Match(TokenType::Local);
        auto string = local_tok.ToString();

        // Parse the optional type annotation (':' <type>).
        Type type;
        if (WhenMatch(TokenType::Colon)) {
          type = ParseType();
        }

        auto var = BindVar(string, type);

        // Parse the '=';
        Match(TokenType::Equal);

        // Parse the body, and the ';'.
        auto val = this->ParseExpr();
        Consume(TokenType::Semicolon);

        // Add the bindings to the local data structure.
        bindings.push_back({ var, val });
        scopes++;
        PushScope();
      } else {
        // This is the only case we will increase the stack
        // depth.
        //
        // If we parse a program which is a sequence of N bindings
        // followed by a single body expression we will end up with
        // a call depth of 3, the first call to ParseExpr, then
        // ParseBindingExpr, then finally ParseExpr once more.
        auto body = this->ParseExpr();

        // Remove the same number of scopes we added.
        PopScope(scopes);

        if (bindings.size() == 0) {
          return body;
        } else {
          // We can now build the let binding up backwards.
          for (auto binding = bindings.rbegin(); binding != bindings.rend(); binding++) {
            body = relay::Let(binding->first, binding->second, body);
          }
          return body;
        }
      }
    }
  }

  std::string HackTokensAsString(int n) {
    std::stringstream key;
    n = std::min((int)(tokens.size() - pos), n);
    for (int i = 0; i < n; i++) {
      key << ToString(tokens.at(pos + i)->token_type);
    }
    return key.str();
  }

  std::vector<Rule> ParseOp() {
    std::vector<Rule> matched;
    Peek();
    for (int i = 4; i > 0; i--) {
      auto key = HackTokensAsString(i);
      auto it = this->op_table.this_is_a_hack.find(key);
      if (it != this->op_table.this_is_a_hack.end()) {
        pos = pos + i;
        matched.push_back(it->second);
      }
    }

    return matched;
  }

  void DebugStack(const std::vector<Expr>& exprs, const std::vector<Rule>& rules) {
      std::cout << "Expr Stack: ";
      for (auto expr : exprs) {
        std::cout << expr << ", ";
      }

      std::cout << std::endl;
      std::cout << "Op Stack: ";
      for (auto rule : rules) {
        std::cout << rule.op << ", ";
      }

      std::cout << std::endl;
  }

  Expr ParseExpr() {
    return ConsumeWhitespace<Expr>([this] {
      // We must parse at least one expression, the default
      // case is that there is no operator and we will fall
      // through.
      std::vector<Expr> exprs;
      exprs.push_back(ParseExprNoOp());

      // Now we parse an optional op.
      std::vector<Rule> ops;

      // We will now parse 0 or more operator occurrences.
      while (true) {
        auto opt_op = ParseOp();

        // If we didn't parse one we done.
        if (opt_op.size() == 0) {
          break;
        }

        // Read the operation we parsed;
        auto op = opt_op[0];

        Expr right = ParseExprNoOp();

        // If the operator stack is empty
        // we parse an operator and expression
        // and push them to stacks, then
        // continue.
        if (ops.size() == 0) {
          ops.push_back(op);
          exprs.push_back(right);
          continue;
        }

        DebugStack(exprs, ops);

        std::cout << "will reduce? " << bool(op.precedence >= ops.back().precedence) << std::endl;

        if (op.precedence > ops.back().precedence ||
              (op.precedence == ops.back().precedence && op.left_assoc == false)) {
          ops.push_back(op);
          exprs.push_back(right);
          continue;
        }

        while (ops.size() && (op.precedence < ops.back().precedence ||
            (op.precedence == ops.back().precedence && op.left_assoc == true))) {
          Rule new_op = ops.back();
          ops.pop_back();
          Expr right = exprs.back();
          exprs.pop_back();
          Expr left = exprs.back();
          exprs.pop_back();
          exprs.push_back(relay::Call(new_op.op, { left, right }));
        }

        exprs.push_back(right);

        ops.push_back(op);
      }

      while (ops.size()) {
        Rule new_op = ops.back();
        ops.pop_back();
        Expr right = exprs.back();
        exprs.pop_back();
        Expr left = exprs.back();
        exprs.pop_back();
        exprs.push_back(relay::Call(new_op.op, {left, right}));
      }

      CHECK_EQ(ops.size(), 0);
      CHECK_EQ(exprs.size(), 1);
      return exprs[0];
    });
  }

  template<typename T>
  Array<T> ParseSequence(TokenType start, TokenType sep, TokenType stop, std::function<T()> parse) {
    Match(start);
    if (WhenMatch(stop)) {
      return Array<T>();
    } else {
      auto data = parse();
      // parse '(' expr ')'
      if (WhenMatch(stop)) {
        return { data };
      // parse '( expr ',' * ')'
      } else if (WhenMatch(sep)) {
        Array<T> elements = { data };
        while (true) {
          if (WhenMatch(TokenType::CloseParen)) {
            break;
          } else {
            auto data = parse();
            WhenMatch(sep);
            elements.push_back(data);
          }
        }
        return elements;
      } else {
        LOG(FATAL) << "issue";
      }
    }
  }

  Type ParseType() {
    auto tok = Peek();
    if (WhenMatch(TokenType::Identifier)) {
      // Need to do better error handling here.
      auto dtype = DataType(String2DLDataType(tok.ToString()));
      return TensorType({}, dtype);
    } if (WhenMatch(TokenType::Underscore)) {
      return IncompleteType();
    } else {
      LOG(FATAL) << "failed to parse type";
      return Type();
    }
  }

  Function ParseFunctionDef() {
    PushScope();

    auto params = ParseSequence<Var>(TokenType::OpenParen, TokenType::Comma, TokenType::CloseParen, [&]() {
      auto token = Match(TokenType::Local);
      auto string = token.ToString();
      Type type;
      if (WhenMatch(TokenType::Colon)) {
        type = ParseType();
      }
      return BindVar(string, type);
    });

    Type ret_type;
    if (WhenMatch(TokenType::Minus)) {
      Match(TokenType::RAngle);
      ret_type = ParseType();
    }

    auto body = Block<Expr>([&]() {
      return ParseExpr();
    });

    std::cout << "After Block" << std::endl;
    DisplayNextN(10);
    PopScope(1);

    return relay::Function(params, body, ret_type, {});
  }

  Expr ParseExprNoOp() {
    return ConsumeWhitespace<Expr>([this] {
      auto next = Peek();
      switch (next->token_type) {
        case TokenType::Integer:
        case TokenType::Float: {
          Consume(next->token_type);
          auto number = NumberToNDArray(next);
          Expr e = Constant(number);
          return e;
        }
        case TokenType::Boolean: {
          Consume(TokenType::Boolean);
          int value = Downcast<tvm::Integer>(next->data);
          auto boolean = BooleanToNDarray(value);
          Expr e = Constant(boolean);
          return e;
        }
        // For graph or let, match first rhs, then invoke ParseBindingExpr
        // ParseBindingExpression then parse_lhs() parse_rhs() ';' continue
        case TokenType::Graph:
        case TokenType::Let:
          return ParseBindingExpr();
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
              Array<Expr> exprs = {expr};
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
        case TokenType::Fn: {
          Consume(TokenType::Fn);
          return Expr(ParseFunctionDef());
        }
        case TokenType::Local: {
          auto string = next.ToString();
          Consume(TokenType::Local);
          return Expr(LookupVarByString(string));
        }
        default:
          ParseError(next, "expected an expression found  " + ToString(next->token_type) +
                               std::to_string(next->line) + ":" + std::to_string(next->column));
      }
    });
  }

  template <typename R>
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

  ObjectRef ParseMetadata() { return ObjectRef(); }
};

OperatorTable DefaultOpTable() {
  return OperatorTable({
      Rule({TokenType::Star}, Op::Get("multiply"), 12, 2, true),
      Rule({TokenType::Division}, Op::Get("divide"), 12, 2, true),
      Rule({TokenType::Plus}, Op::Get("add"), 10, 2, true),
      Rule({TokenType::Minus}, Op::Get("subtract"), 10, 2, true),
      Rule({TokenType::LAngle}, Op::Get("less"), 8, 2, true),
      Rule({TokenType::LAngle, TokenType::Equal}, Op::Get("less_equal"), 8, 2, true),
      Rule({TokenType::RAngle}, Op::Get("greater"), 8, 2, true),
      Rule({TokenType::RAngle, TokenType::Equal}, Op::Get("greater_equal"), 8, 2, true),
      Rule({TokenType::Equal, TokenType::Equal}, Op::Get("equal"), 7, 2, true),
      Rule({TokenType::Bang, TokenType::Equal}, Op::Get("not_equal"), 7, 2, true)
    });
}

IRModule ParseModule(std::string file_name, std::string file_content) {
  auto tokens = Tokenize(file_content);
  for (auto token : tokens) {
    std::cout << token << std::endl;
  }
  Parser parser(tokens, DefaultOpTable());
  return parser.ParseModule();
}

Expr ParseExpr(std::string file_name, std::string file_content) {
  auto tokens = Tokenize(file_content);
  for (auto token : tokens) {
    std::cout << token << std::endl;
  }
  Parser parser(tokens, DefaultOpTable());
  auto expr = parser.ParseExpr();
  parser.Match(TokenType::EndOfFile);
  return expr;
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
