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
#include <tvm/node/reflection.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <fstream>

#include "./diagnostic.h"
#include "./op_table.h"
#include "./tokenizer.h"

namespace tvm {
namespace parser {

using namespace relay;
using Expr = relay::Expr;

/*! \brief A wrapper structure for capturing the result of parsing
 * a global definition *before* we add it to the IRModule.
 *
 * This enables the parser to parse everything in one pass before
 * constructing the IRModule.
 */
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

/*! \brief A wrapper structure for capturing all top-level definitions
 * when parsing a module.
 */
struct Definitions {
  /*! \brief The set of global functions. */
  std::vector<GlobalFunc> funcs;
  /*! \brief The set of type definitions. */
  std::vector<TypeData> types;
  // TODO(@jroesch): contain meta-table below
};

/*! \brief A structure representing the semantic versioning information
 * for a Relay program.
 */
class SemVer {
 public:
  int major_version;
  int minor_version;
  int patch_version;

  SemVer() : major_version(0), minor_version(0), patch_version(0) {}
  SemVer(int major_version, int minor_version, int patch_version)
      : major_version(major_version), minor_version(minor_version), patch_version(patch_version) {}
  SemVer(const SemVer& other)
      : major_version(other.major_version),
        minor_version(other.minor_version),
        patch_version(other.patch_version) {}
};

/*! \brief A reference to a "meta-expression".
 *
 * In the text format we allow referencing metadata which
 * uses a compact serialization that proceeds the main
 * program body.
 *
 * We can reference this table using an expression of
 * the form `meta[Type][index]`.
 *
 * We must later resolve these references to actual in-memory
 * AST nodes but this requires first parsing the full program
 * then expanding these temporary AST nodes into their corresponding
 * nodes.
 *
 * For example the nth large constant will be pretty-printed as meta[relay.Constant][n]
 * with its compact binary serialization residing in the metadata section at the end
 * of the program.
 */
class MetaRefExprNode : public TempExprNode {
 public:
  /*! \brief The type key of the meta expression. */
  std::string type_key;
  /*! \brief The index into the type key's table. */
  uint64_t node_index;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  // TODO(@jroesch): we probably will need to manually
  // expand these with a pass.
  Expr Realize() const final { return Expr(); }

  static constexpr const char* _type_key = "relay.MetaRefExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(MetaRefExprNode, TempExprNode);
};

class MetaRefExpr : public TempExpr {
 public:
  /*!
   * \brief The constructor for MetaRefExpr
   * \param type_key The type key of the object in the meta section.
   * \param kind The index into that subfield.
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

/*! \brief A simple wrapper around a mapping from raw string names
 * to a TVM variable, type variable or other binder type.
 */
template <typename T>
struct Scope {
  /*! \brief The internal map. */
  std::unordered_map<std::string, T> name_map;
};

/*! \brief A stack of scopes.
 *
 * In order to properly handle scoping we must maintain a stack of scopes.
 *
 * A stack allows users to write programs which contain repeated variable
 * names and to properly handle both nested scopes and removal of variables
 * when they go out of scope.
 *
 * This is the classic approach to lexical scoping.
 */
template <typename T>
class ScopeStack {
 private:
  std::vector<Scope<T>> scope_stack;

 public:
  /*! \brief Adds a variable binding to the current scope. */
  void Add(const std::string& name, const T& value) {
    if (!this->scope_stack.size()) {
      LOG(FATAL) << "internal issue";
    }
    this->scope_stack.back().name_map.insert({name, value});
  }

  /*! \brief Looks up a variable name in the scope stack returning the matching variable
   * in most recent scope. */
  T Lookup(const std::string& name) {
    for (auto scope = this->scope_stack.rbegin(); scope != this->scope_stack.rend(); ++scope) {
      auto it = scope->name_map.find(name);
      if (it != scope->name_map.end()) {
        return it->second;
      }
    }
    return T();
  }

  /*! \brief Adds a fresh scope. */
  void PushStack() { this->scope_stack.push_back(Scope<T>()); }

  /*! \brief Removes the most recent scope. */
  void PopStack() { this->scope_stack.pop_back(); }
};

/*! \brief A table of interning strings as global function and type names. */
template <typename T>
struct InternTable {
  /*! \brief The internal table mapping strings to a unique allocation. */
  std::unordered_map<std::string, T> table;

  /*! \brief Add the unique allocation. */
  void Add(const std::string& name, const T& t) {
    auto it = table.find(name);
    if (it != table.end()) {
      LOG(FATAL) << "duplicate name";
    } else {
      table.insert({name, t});
    }
  }

  /*! \brief Return the unique allocation. */
  Optional<T> Get(const std::string& name) {
    auto it = table.find(name);
    if (it != table.end()) {
      return Optional<T>(it->second);
    } else {
      return Optional<T>();
    }
  }
};

/*! \brief The parser class is the main interface to the parser.
 * the parser is not currently exposed beyond this .cc file.
 *
 * The parser is initialized with a diagnostic context, an
 * operator table, and a token stream.
 *
 * The rest of the internal state is used to map the human readable
 * form to in-memory IR representation.
 *
 * The main entry point to the parser are a set of parsing methods
 * such as `ParseModule` and `ParseExpr`.
 *
 * As with traditional recursive descent parsers the parsing methods
 * are factored recursively just as one would do with a formal language
 * grammar.
 *
 * You can view a recursive descent parser as a human friendly way to specify
 * a state machine, and thus this factoring is necessary as the 'state' of this
 * machine is the combination of the current parsing method and the next token.
 *
 * Parsing proceeds by matching a token and then dispatching to the appropriate
 * method to parse the next tokens in the stream.
 *
 * For example if we are parsing a type and encounter a "Tensor" token we switch
 * into a mode for parsing `[`, a shape, a comma, a data type and then a `]`.
 *
 * Certain matches like this are unambiguous and proceed in a straight line fashion
 * once the initial token is found. Other parsing is more complex and requires some
 * tricks to correctly parse.
 *
 * For example when we find a '(' in an expression context, it may be part of
 * a tuple, the arguments to a call, or a parenthesized expression. The below code
 * disambiguate these cases by factoring expression parsing into a series of methods
 * which encode the parsing context and thus how to interpret the parenthesis.
 *
 * For more information one should be able to read the code in order starting with
 * `ParseModule` or `ParseExpr`.
 */
class Parser {
 public:
  /*! \brief The version that the parser is parsing. */
  SemVer version;

  /*! \brief The diagnostic context used for error reporting. */
  DiagnosticContext diag_ctx;

  /*! \brief The current position in the token stream. */
  int pos;

  /*! \brief The token stream for the parser. */
  std::vector<Token> tokens;

  /*! \brief The configured operator table. */
  OperatorTable op_table;

  /*! \brief Configure the whitespace mode, right now we ignore all whitespace. */
  bool ignore_whitespace;

  /*! \brief A global mapping for GlobalVar. */
  InternTable<GlobalVar> global_names;

  /*! \brief A global mapping for type definitions. */
  InternTable<GlobalTypeVar> type_names;

  /*! \brief A global mapping for constructor names. */
  InternTable<Constructor> ctors;

  /*! \brief A mapping from graph variable to expression, i.e., `%0 = expr`. */
  std::unordered_map<int, Expr> graph_ctx;

  /*! \brief The set of type scopes used for generics. */
  ScopeStack<TypeVar> type_scopes;

  /*! \brief The set of expression scopes used for lexical scope. */
  ScopeStack<Var> expr_scopes;

  Parser(std::vector<Token> tokens, OperatorTable op_table, Source source)
      : diag_ctx(source), pos(0), tokens(tokens), op_table(op_table), ignore_whitespace(true) {}

  /*! \brief Examine the next token in the stream, the current parser is configured to be
   * whitespace insensitive so we will skip all whitespace or comment tokens. */
  Token Peek() {
    // For now we ignore all whitespace tokens and comments.
    // We can tweak this behavior later to enable white space sensitivity in the parser.
    while (pos < static_cast<int64_t>(tokens.size()) && ignore_whitespace &&
           (tokens.at(pos)->token_type == TokenType::Whitespace ||
            tokens.at(pos)->token_type == TokenType::Newline ||
            tokens.at(pos)->token_type == TokenType::LineComment ||
            tokens.at(pos)->token_type == TokenType::Comment)) {
      pos++;
    }

    if (pos < static_cast<int64_t>(tokens.size())) {
      return Token(this->tokens.at(pos));
    } else {
      return Token::Null();
    }
  }

  /*! \brief Lookahead by N tokens.
   * \param n The number of tokens to lookahead.
   * \return The Nth token.
   */
  Token Lookahead(int n) {
    CHECK_GE(n, 1) << "lookahead is only valid when n >= 1";

    // We intend to skip n - 1 tokens, then return the nth.
    auto old_pos = pos;
    for (int i = 0; i < n - 1; i++) {
      Peek();
      pos++;
    }

    auto tok = Peek();
    pos = old_pos;
    return tok;
  }

  /*! \brief Consume a token, this method is the lowest level way to consume a token
   * and will not ignore white space or look ahead in anyway.
   *
   * /param token_type The token type to match.
   */
  void Consume(const TokenType& token_type) {
    if (tokens[pos]->token_type != token_type) {
      std::string message =
          "expected a " + Pretty(token_type) + " found " + Pretty(Peek()->token_type);
      this->diag_ctx.Emit({tokens[pos]->line, tokens[pos]->column, message});
      this->diag_ctx.Render(std::cout);
    }
    pos++;
  }

  /*! Match a token in the stream, this will first invoke Peek, ignoring tokens such
   * as whitespace or comments returning the first meaningful token.
   *
   * We then try and consume the requested token, this will trigger an error if the
   * current token does not match the token_type.
   */
  Token Match(const TokenType& token_type) {
    auto tok = Peek();
    Consume(token_type);
    return tok;
  }

  /*! Conditionally consume a token when it matches, this will never trigger an error
   * as we guard against consuming the token before we do.
   *
   * Useful for matching optional tokens, effectively looksahead by one.
   */
  bool WhenMatch(const TokenType& token_type) {
    if (Peek()->token_type == token_type) {
      Consume(token_type);
      return true;
    } else {
      return false;
    }
  }

  /* \brief Add a graph binding to the parsing context
   *
   * For example if we parse %0 = add(...), map 0 -> add(...), etc.
   */
  void AddGraphBinding(const Token& token, const Expr& expr) {
    auto graph_no = token.ToNumber();
    this->graph_ctx.insert({graph_no, expr});
  }

  /* \brief Lookup a previously bound graph variable.
   *
   * Note: we take tokens in all lookup methods so that we
   * that we can do error reporting based on token location.
   */
  Expr LookupGraphBinding(const Token& token) {
    auto graph_no = token.ToNumber();
    return this->graph_ctx.at(graph_no);
  }

  /*! \brief Bind a local variable in the expression scope.
   *
   * "x" -> Var("x"), these are needed to map from the raw string names
   * to unique variable nodes.
   */
  Var BindVar(const std::string& name, const relay::Type& type_annotation) {
    auto var = Var(name, type_annotation);
    this->expr_scopes.Add(name, var);
    return var;
  }

  /*! \brief Bind a type variable in the type scope.
   *
   * "A" -> TypeVar("A", ...), these are needed to map from raw string names
   * to unique type variable nodes.
   */
  TypeVar BindTypeVar(const std::string& name, const TypeKind type_kind) {
    auto type_var = TypeVar(name, type_kind);
    this->type_scopes.Add(name, type_var);
    return type_var;
  }

  /*! \brief Lookup a variable in the expression scope.
   *
   * Note: all lookup methods take tokens intentionally for error reporting information.
   */
  Var LookupLocal(const Token& local) {
    auto var = this->expr_scopes.Lookup(local.ToString());
    if (!var.defined()) {
      diag_ctx.Emit(
          {local->line, local->column, "this local variable has not been previously declared"});
    }
    return var;
  }

  /*! \brief Lookup a variable in the type scope.
   *
   * Note: all lookup methods take tokens intentionally for error reporting information.
   */
  TypeVar LookupTypeVar(const Token& ident) {
    auto var = this->type_scopes.Lookup(ident.ToString());
    if (!var.defined()) {
      diag_ctx.Emit(
          {ident->line, ident->column,
           "this type variable has not been previously declared anywhere, perhaps a typo?"});
    }
    return var;
  }

  /*! \brief Add an expression scope to the scope stack. */
  void PushScope() { this->expr_scopes.PushStack(); }

  /*! \brief Remove N expression scopes from the scope stack. */
  void PopScopes(int n) {
    for (int i = 0; i < n; i++) {
      this->expr_scopes.PopStack();
    }
  }

  /*! \brief Add an type scope to the scope stack. */
  void PushTypeScope() { this->type_scopes.PushStack(); }

  /*! \brief Remove N type scopes from the scope stack. */
  void PopTypeScopes(int n) {
    for (int i = 0; i < n; i++) {
      this->type_scopes.PopStack();
    }
  }

  /*! \brief Convert a numeric token to an NDArray for embedding into the Relay program. */
  NDArray NumberToNDArray(const Token& token) {
    if (token->token_type == TokenType::Integer) {
      DLContext ctx = {DLDeviceType::kDLCPU, 0};
      auto dtype = String2DLDataType("int32");
      auto data = NDArray::Empty({}, dtype, ctx);
      auto array = reinterpret_cast<int32_t*>(data->data);
      // revisit this, literal node issue.
      int64_t value = Downcast<tvm::Integer>(token->data);
      array[0] = (int32_t)value;
      return data;
    } else if (token->token_type == TokenType::Float) {
      DLContext ctx = {DLDeviceType::kDLCPU, 0};
      auto dtype = String2DLDataType("float32");
      auto data = NDArray::Empty({}, dtype, ctx);
      auto array = reinterpret_cast<float*>(data->data);
      // revisit this, literal node issue.
      // TODO(@jroesch): bounds checking
      float value = Downcast<tvm::FloatImm>(token->data)->value;
      array[0] = value;
      return data;
    } else {
      LOG(FATAL) << "internal error: should only call this function on numeric tokens";
      return NDArray();
    }
  }

  /*! \brief Convert a boolean value to an NDArray for embedding into the Relay program. */
  NDArray BooleanToNDarray(bool value) {
    DLContext ctx = {DLDeviceType::kDLCPU, 0};
    auto dtype = String2DLDataType("bool");
    auto data = NDArray::Empty({}, dtype, ctx);
    auto array = reinterpret_cast<bool*>(data->data);
    array[0] = value;
    return data;
  }

  [[noreturn]] void ParseError(const Token& token, const std::string& msg) {
    throw std::runtime_error(msg);
  }

  /*! \brief A parsing helper for a bracketed expression <start> <parser> <stop>. */
  template <typename R>
  R Bracket(TokenType open, TokenType close, std::function<R()> parser) {
    Match(open);
    R result = parser();
    Match(close);
    return result;
  }

  /*! \brief Parse `(` parser() `)`. */
  template <typename R>
  R Parens(std::function<R()> parser) {
    return Bracket(TokenType::OpenParen, TokenType::CloseParen, parser);
  }

  /*! \brief Parse `{` parser() `}`. */
  template <typename R>
  R Block(std::function<R()> parser) {
    return Bracket(TokenType::LCurly, TokenType::RCurly, parser);
  }

  /*! \brief Parses a sequence beginning with a start token, seperated by a seperator token, and
   * ending with a stop token.
   *
   * The simple form being <start> (<parse()> <seperator>)* <stop>.
   *
   * This also provides a fourth argument which is allowed to run when the sequence which matches
   * the inner sequence can not proceed.
   *
   * This is useful for parsing things like attributes which don't match the standard expression
   * parsers but are contained within the stop token.
   */
  template <typename T>
  Array<T> ParseSequence(TokenType start, TokenType sep, TokenType stop, std::function<T()> parse,
                         std::function<void()> before_stop = nullptr) {
    Match(start);
    if (WhenMatch(stop)) {
      return Array<T>();
    } else {
      auto data = parse();
      Array<T> elements = {data};

      // parse '(' expr ')'
      // if we are at the end invoke leftover parser
      if (Peek()->token_type == stop && before_stop) {
        before_stop();
      }
      if (WhenMatch(stop)) {
        return elements;
        // parse '( expr ',' * ')'
      } else if (WhenMatch(sep)) {
        // if we are at the end invoke leftover parser
        if (Peek()->token_type == stop && before_stop) {
          before_stop();
        }
        while (true) {
          if (WhenMatch(stop)) {
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
        return Array<T>(nullptr);
      }
    }
  }

  /*! \brief Parse a full IRModule. */
  IRModule ParseModule() {
    // Parse the semver header at the top of the module.
    this->version = ParseSemVer();
    // Parse the definitions.
    auto defs = ParseDefinitions();
    // Parse the metadata section at the end.
    auto metadata = ParseMetadata();
    Match(TokenType::EndOfFile);
    Map<tvm::GlobalVar, BaseFunc> funcs;
    Map<tvm::GlobalTypeVar, TypeData> types;

    for (auto type_def : defs.types) {
      types.Set(type_def->header, type_def);
    }

    auto mod = IRModule({}, types);

    for (auto func : defs.funcs) {
      mod->Add(func.global, func.function);
    }

    return mod;
  }

  /*! \brief Parse the semantic versioning header. */
  SemVer ParseSemVer() {
    // TODO(@jroesch): convert semver to module level attribute.
    auto id = Peek();
    if (id->token_type == TokenType::Identifier && id.ToString() == "v0") {
      auto id = Match(TokenType::Identifier);
      Consume(TokenType::Period);
      Consume(TokenType::Float);
    }
    // TODO(@jroesch): the current lexing makes it hard to parse this
    // in a way that doesnt feel like a hack.
    //
    // We should move to module level attributes instead
    // so we can tag modules with top-level data.
    //
    // #[text_version = "0.0.4"]
    //
    // For now we only support current version.
    return SemVer(0, 0, 4);
  }

  /*! \brief Parse zero or more Relay definitions. */
  Definitions ParseDefinitions() {
    Definitions defs;

    while (true) {
      auto next = Peek();
      switch (next->token_type) {
        case TokenType::Defn: {
          Consume(TokenType::Defn);
          auto global_name = Match(TokenType::Global).ToString();
          auto global = GlobalVar(global_name);
          global_names.Add(global_name, global);
          auto func = ParseFunctionDef();
          defs.funcs.push_back(GlobalFunc(global, func));
          continue;
        }
        case TokenType::TypeDef: {
          defs.types.push_back(ParseTypeDef());
          continue;
        }
        case TokenType::Extern: {
          Consume(TokenType::Extern);
          auto type_def = ParseTypeDef();
          if (type_def->constructors.size()) {
            diag_ctx.Emit(
                {next->line, next->column, "an external type may not have any constructors"});
          }
          defs.types.push_back(type_def);
        }
        default:
          return defs;
      }
    }
  }

  /*! \brief Parse zero or more Relay type definitions. */
  TypeData ParseTypeDef() {
    // Match the `type` keyword.
    Match(TokenType::TypeDef);
    // Parse the type's identifier.
    auto type_id = Match(TokenType::Identifier).ToString();
    auto type_global = tvm::GlobalTypeVar(type_id, TypeKind::kAdtHandle);
    type_names.Add(type_id, type_global);

    Array<TypeVar> generics;

    bool should_pop = false;
    if (Peek()->token_type == TokenType::LSquare) {
      // If we have generics we need to add a type scope.
      PushTypeScope();
      should_pop = true;
      generics =
          ParseSequence<TypeVar>(TokenType::LSquare, TokenType::Comma, TokenType::RSquare, [&]() {
            auto type_var_name = Match(TokenType::Identifier).ToString();
            return BindTypeVar(type_var_name, TypeKind::kType);
          });
    }

    Array<tvm::Constructor> ctors;
    if (Peek()->token_type == TokenType::LCurly) {
      // Parse the list of constructors.
      ctors = ParseSequence<tvm::Constructor>(
          TokenType::LCurly, TokenType::Comma, TokenType::RCurly, [&]() {
            // First match the name of the constructor.
            auto ctor_name = Match(TokenType::Identifier).ToString();

            Constructor ctor;
            // Match the optional field list.
            if (Peek()->token_type != TokenType::OpenParen) {
              ctor = tvm::Constructor(ctor_name, {}, type_global);
            } else {
              auto arg_types =
                  ParseSequence<Type>(TokenType::OpenParen, TokenType::Comma, TokenType::CloseParen,
                                      [&]() { return ParseType(); });
              ctor = tvm::Constructor(ctor_name, arg_types, type_global);
            }

            CHECK(ctor.defined());

            this->ctors.Add(ctor_name, ctor);

            return ctor;
          });
    }

    // Now pop the type scope.
    if (should_pop) {
      PopTypeScopes(1);
    }

    return TypeData(type_global, generics, ctors);
  }

  std::string HackTokensAsString(int n) {
    std::stringstream key;
    n = std::min(static_cast<int>(tokens.size() - pos), n);
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

  /*! \brief Parse a single Relay expression. */
  Expr ParseExpr() {
    return ConsumeWhitespace<Expr>([this] {
      std::vector<Expr> exprs;

      while (true) {
        auto next = Peek();
        switch (next->token_type) {
          // For graph or let, match first rhs, then invoke ParseBindingExpr
          // ParseBindingExpression then parse_lhs() parse_rhs() ';' continue
          case TokenType::LCurly: {
            // NB: Might need to optimize to remove deep recursion.
            // Stack should only grow proportionally to the number of
            // nested scopes.
            return Bracket<Expr>(TokenType::LCurly, TokenType::RCurly, [&]() {
              PushScope();
              auto expr = ParseExpr();
              PopScopes(1);
              return expr;
            });
          }
          case TokenType::Let:
            exprs.push_back(ParseBindingExpr());
            break;
          case TokenType::Match:
          case TokenType::PartialMatch: {
            bool is_total = next->token_type == TokenType::Match;
            Consume(next->token_type);
            exprs.push_back(ParseMatch(is_total));
            break;
          }
          case TokenType::If: {
            exprs.push_back(ParseIf());
            break;
          }
          case TokenType::Graph:
            if (Lookahead(2)->token_type == TokenType::Equal) {
              exprs.push_back(ParseBindingExpr());
              break;
            }
            // intentional fall through here.
          default: {
            exprs.push_back(ParseExprBinOp());
            break;
          }
        }

        if (!WhenMatch(TokenType::Semicolon)) {
          break;
        }
      }

      CHECK_GE(exprs.size(), 1);

      if (exprs.size() == 1) {
        return exprs[0];
      } else {
        auto body = exprs.back();
        exprs.pop_back();
        while (exprs.size()) {
          auto value = exprs.back();
          exprs.pop_back();
          body = relay::Let(Var("", IncompleteType()), value, body);
        }
        return body;
      }
    });
  }

  /*! \brief Parse a "binding expression"; an expression where
   * a graph or let variable is bound.
   *
   * In order to avoid stack overflow this is implemented in a special
   * iterative way to keep stack depth constant in a long chain of bindings.
   */
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
      if (next->token_type == TokenType::Graph && Lookahead(2)->token_type == TokenType::Equal) {
        Match(TokenType::Graph);
        Match(TokenType::Equal);
        auto val = this->ParseExprBinOp();
        Match(TokenType::Semicolon);
        AddGraphBinding(next, val);
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
        auto val = this->ParseExprBinOp();
        Consume(TokenType::Semicolon);

        // Add the bindings to the local data structure.
        bindings.push_back({var, val});
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
        PopScopes(scopes);

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

  /*! Parse a function definition without a leading keyword or identifier.
   *
   * Handles things of the form [T1, ..., TN](arg1: U1, ..., argN, UN) -> Ret { body }.
   */
  Function ParseFunctionDef() {
    PushScope();
    PushTypeScope();

    Array<TypeVar> generics;
    if (Peek()->token_type == TokenType::LSquare) {
      // If we have generics we need to add a type scope.
      PushTypeScope();
      generics =
          ParseSequence<TypeVar>(TokenType::LSquare, TokenType::Comma, TokenType::RSquare, [&]() {
            auto type_var_name = Match(TokenType::Identifier).ToString();
            return BindTypeVar(type_var_name, TypeKind::kType);
          });
    }

    auto params =
        ParseSequence<Var>(TokenType::OpenParen, TokenType::Comma, TokenType::CloseParen, [&]() {
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

    auto body = Block<Expr>([&]() { return ParseExpr(); });

    PopTypeScopes(1);
    PopScopes(1);

    return relay::Function(params, body, ret_type, generics);
  }

  /*! \brief Parse an if-expression. */
  Expr ParseIf() {
    Consume(TokenType::If);
    auto guard = Parens<Expr>([&] { return ParseExpr(); });

    auto true_branch = Block<Expr>([&] { return ParseExpr(); });

    Match(TokenType::Else);

    auto false_branch = Block<Expr>([&] { return ParseExpr(); });

    return relay::If(guard, true_branch, false_branch);
  }

  /* This factors parsing a list of patterns for both tuples, and constructors. */
  Array<Pattern> ParsePatternList() {
    return ParseSequence<Pattern>(TokenType::OpenParen, TokenType::Comma, TokenType::CloseParen,
                                  [&] { return ParsePattern(); });
  }

  /*! \brief Parses a pattern for a match expression.
   *
   * A pattern is either a wildcard `_`, a local `%name`,
   * a constructor `C(p1, ..., pn)` or tuple `(p1, ..., pn).
   *
   * This function recursively parses a pattern.
   */
  Pattern ParsePattern() {
    auto next = Peek();
    switch (next->token_type) {
      case TokenType::Underscore: {
        Match(TokenType::Underscore);
        return PatternWildcard();
      }
      case TokenType::Local: {
        auto id = Match(TokenType::Local);
        Type type_annotation;
        if (WhenMatch(TokenType::Colon)) {
          type_annotation = ParseType();
        }
        auto var = BindVar(id.ToString(), type_annotation);
        return PatternVar(var);
      }
      case TokenType::Identifier: {
        auto id = Match(TokenType::Identifier);
        auto ctor = ctors.Get(id.ToString());
        CHECK(ctor) << "undefined identifier";
        if (Peek()->token_type == TokenType::OpenParen) {
          auto fields = ParsePatternList();
          return PatternConstructor(ctor.value(), fields);
        } else {
          return PatternConstructor(ctor.value(), {});
        }
      }
      default:
        return PatternTuple(ParsePatternList());
    }
  }

  Clause ParseMatchArm() {
    PushScope();
    auto pattern = ParsePattern();
    Match(TokenType::Equal);
    Consume(TokenType::RAngle);
    auto expr = ParseExpr();
    PopScopes(1);
    return Clause(pattern, expr);
  }

  Expr ParseMatch(bool is_total) {
    Expr scrutinee = ParseExpr();

    Array<Clause> clauses = ParseSequence<Clause>(
        TokenType::LCurly, TokenType::Comma, TokenType::RCurly, [&] { return ParseMatchArm(); });

    return relay::Match(scrutinee, clauses, is_total);
  }

  Expr ParseExprBinOp() {
    return ConsumeWhitespace<Expr>([this] {
      // We must parse at least one expression, the default
      // case is that there is no operator and we will fall
      // through.
      std::vector<Expr> exprs;
      exprs.push_back(ParseCallExpr());

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

        Expr right = ParseCallExpr();

        // If the operator stack is empty
        // we parse an operator and expression
        // and push them to stacks, then
        // continue.
        if (ops.size() == 0) {
          ops.push_back(op);
          exprs.push_back(right);
          continue;
        }

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
          exprs.push_back(relay::Call(new_op.op, {left, right}));
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

  Attrs ParseAttrs(const std::string& type_key) {
    Map<String, ObjectRef> kwargs;
    auto attrs = tvm::ReflectionVTable::Global()->CreateObject(type_key, kwargs);
    LOG(FATAL) << Attrs();
    return Attrs();
  }

  Expr ParseCallArgs(Expr op) {
    Attrs call_attrs;
    if (Peek()->token_type == TokenType::OpenParen) {
      Array<Expr> args = ParseSequence<Expr>(
          TokenType::OpenParen, TokenType::Comma, TokenType::CloseParen,
          [&] { return ParseExpr(); },
          [&] {
            auto is_ident = Lookahead(1)->token_type == TokenType::Identifier;
            auto next_is_equal = Lookahead(2)->token_type == TokenType::Equal;

            if (is_ident && next_is_equal) {
              if (auto op_node = op.as<OpNode>()) {
                call_attrs = ParseAttrs(op_node->attrs_type_key);
              }
            }
          });
      return Expr(Call(op, args, call_attrs, {}));
    } else {
      return Expr();
    }
  }

  Expr ParseCallExpr() {
    return ConsumeWhitespace<Expr>([this] {
      Expr expr = ParseAtomicExpr();
      // Parse as many call args as possible, building up expression
      //
      // NB(@jroesch): this seems like a hack but in order to parse curried functions
      // and avoid complex grammar we will parse multiple call lists in a row.
      while (true) {
        auto new_expr = ParseCallArgs(expr);
        if (new_expr.defined()) {
          expr = new_expr;
        } else {
          break;
        }
      }

      // We need a zero-arity case for constructors.
      if (expr.as<ConstructorNode>()) {
        return Expr(Call(expr, {}));
      } else {
        return expr;
      }
    });
  }

  Expr ParseAtomicExpr() {
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
        case TokenType::Local: {
          Consume(TokenType::Local);
          return Expr(LookupLocal(next));
        }
        case TokenType::Global: {
          auto string = next.ToString();
          Consume(TokenType::Global);
          auto global = global_names.Get(string);
          if (!global) {
            auto global_var = GlobalVar(string);
            global_names.Add(string, global_var);
            return Expr(global_var);
          } else {
            return Expr(global.value());
          }
        }
        case TokenType::Identifier: {
          auto string = next.ToString();
          Consume(TokenType::Identifier);
          auto ctor = ctors.Get(string);
          if (ctor) {
            return Expr(ctor.value());
          } else {
            return Expr(Op::Get(string));
          }
        }
        case TokenType::Graph: {
          Consume(TokenType::Graph);
          return LookupGraphBinding(next);
        }
        case TokenType::Fn: {
          Consume(TokenType::Fn);
          return Expr(ParseFunctionDef());
        }
        case TokenType::OpenParen: {
          Consume(TokenType::OpenParen);
          // parse '(' ')'
          if (WhenMatch(TokenType::CloseParen)) {
            return Expr(Tuple(Array<Expr>()));
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
        default: {
          std::stringstream msg;
          msg << "expected an expression found  " << Pretty(next->token_type);
          diag_ctx.Emit({next->line, next->column, msg.str()});
          diag_ctx.Render(std::cout);
          return Expr();
        }
      }
    });
  }

  /*! \brief Parse a shape. */
  Array<tvm::PrimExpr> ParseShape() {
    auto dims = ParseSequence<tvm::PrimExpr>(TokenType::OpenParen, TokenType::Comma,
                                             TokenType::CloseParen, [&]() {
                                               auto tok = Match(TokenType::Integer);
                                               return Downcast<tvm::PrimExpr>(tok->data);
                                             });
    return dims;
  }

  /*! \brief Parse a function type. */
  Type ParseFunctionType() {
    auto ty_params = ParseSequence<Type>(TokenType::OpenParen, TokenType::Comma,
                                         TokenType::CloseParen, [&]() { return ParseType(); });

    Match(TokenType::Minus);
    Match(TokenType::RAngle);
    auto ret_type = ParseType();

    return relay::FuncType(ty_params, ret_type, {}, {});
  }

  // Parses a user defined ADT or type variable.
  Type ParseNonPrimitiveType(const Token& tok) {
    auto name = tok.ToString();
    Type head_type;
    auto global_type = type_names.Get(name);

    if (!global_type) {
      head_type = LookupTypeVar(tok);
    } else {
      head_type = global_type.value();
    }

    CHECK(head_type.defined()) << "internal error: head type must be defined";

    Array<Type> arg_types;
    if (Peek()->token_type == TokenType::LSquare) {
      arg_types = ParseSequence<Type>(TokenType::LSquare, TokenType::Comma, TokenType::RSquare,
                                      [&]() { return ParseType(); });
    }

    if (arg_types.size()) {
      return TypeCall(head_type, arg_types);
    } else {
      return head_type;
    }
  }

  /*! \brief Parses a TVM type.
   *
   * This matches either a `Tensor[shape, dtype]`, a user defined ADT, a tuple type,
   * a scalar type or an incomplete type `_`.
   */
  Type ParseType() {
    auto tok = Peek();

    if (tok->token_type == TokenType::OpenParen) {
      auto tys = ParseSequence<relay::Type>(TokenType::OpenParen, TokenType::Comma,
                                            TokenType::CloseParen, [&]() { return ParseType(); });
      return relay::TupleType(tys);
    } else if (WhenMatch(TokenType::Fn)) {
      return ParseFunctionType();
    } else if (WhenMatch(TokenType::Identifier)) {
      auto id = tok.ToString();
      if (id == "Tensor") {
        Match(TokenType::LSquare);
        auto shape = ParseShape();
        Match(TokenType::Comma);
        auto dtype_tok = Match(TokenType::Identifier);
        auto dtype = DataType(String2DLDataType(dtype_tok.ToString()));
        Match(TokenType::RSquare);
        return TensorType(shape, dtype);
      } else {
        auto ty = tok.ToString();
        if (ty.rfind("int", 0) == 0 || ty.find("float", 0) == 0 || ty.find("uint", 0) == 0 ||
            ty.find("bool", 0) == 0) {
          // Need to do better error handling here.
          auto dtype = DataType(String2DLDataType(tok.ToString()));
          return TensorType({}, dtype);
        } else {
          return ParseNonPrimitiveType(tok);
        }
      }
    }
    if (WhenMatch(TokenType::Underscore)) {
      return IncompleteType();
    } else {
      std::stringstream msg;
      msg << "failed to parse type found ";
      msg << tok;
      diag_ctx.Emit({tok->line, tok->column, msg.str()});
      diag_ctx.Render(std::cout);
      return Type();
    }
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

  // TODO(@jroesch): this is the final remaining feature.
  ObjectRef ParseMetadata() { return ObjectRef(); }

  /*! \brief A helper for debugging the parser, displays the next N tokens in the token stream. */
  void DisplayNextN(int n) {
    std::cout << "remaining tokens: " << std::endl;
    auto bound = std::min(pos + n, static_cast<int>(tokens.size()));
    for (int i = 0; i < bound - pos; i++) {
      std::cout << tokens[pos + i] << std::endl;
    }
  }

  // A function for debugging the operator parser.
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
};

IRModule ParseModule(std::string file_name, std::string file_content) {
  auto tokens = Tokenize(file_content);
  Parser parser(tokens, DefaultOpTable(), Source(file_content));
  return parser.ParseModule();
}

Expr ParseExpr(std::string file_name, std::string file_content) {
  auto tokens = Tokenize(file_content);
  Parser parser(tokens, DefaultOpTable(), Source(file_content));
  parser.PushScope();
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
