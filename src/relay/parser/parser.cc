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
#include <tvm/relay/parser.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/builtin_fp16.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/virtual_device.h>

#include <fstream>

#include "../../support/scalars.h"
#include "./meta_ref.h"
#include "./op_table.h"
#include "./span_check.h"
#include "./tokenizer.h"

namespace tvm {
namespace relay {

/*! \brief The meta table maps from type key to a sequence of objects. */
using MetaTable = Map<String, Array<ObjectRef>>;

using tvm::runtime::NDArray;
using tvm::runtime::String2DLDataType;
using tvm::transform::CreateModulePass;
using tvm::transform::PassContext;

/*! \brief A helper for passing around spans with data structures with
 * no span field.
 */
template <typename T>
struct Spanned {
  T data;
  Span span;

  Spanned() = default;
  Spanned(const Spanned<T>& other) = default;
  Spanned(T data, Span span) : data(data), span(span) {}
};

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
  std::unordered_map<std::string, T> free_vars;

 public:
  /*! \brief Adds a variable binding to the current scope. */
  void Add(const std::string& name, const T& value) {
    if (!this->scope_stack.size()) {
      LOG(FATAL) << "internal issue";
    }
    this->scope_stack.back().name_map.insert({name, value});
  }

  void AddFreeVar(const std::string& name, const T& value) { free_vars.insert({name, value}); }

  /*! \brief Looks up a variable name in the scope stack returning the matching variable
   * in most recent scope. */
  T Lookup(const std::string& name) {
    for (auto scope = this->scope_stack.rbegin(); scope != this->scope_stack.rend(); ++scope) {
      auto it = scope->name_map.find(name);
      if (it != scope->name_map.end()) {
        return it->second;
      }
    }

    // Check if we bound a free variable declaration.
    auto it = free_vars.find(name);
    if (it != free_vars.end()) {
      return it->second;
    }

    return T();
  }

  /*! \brief Adds a fresh scope. */
  void PushStack() { this->scope_stack.push_back(Scope<T>()); }

  /*! \brief Removes the most recent scope. */
  void PopStack() { this->scope_stack.pop_back(); }
};

struct DuplicateKeyError : public Error {
  explicit DuplicateKeyError(const std::string& msg) : Error(msg) {}
};

/*! \brief A table of interning strings as global function and type names. */
template <typename T>
struct InternTable {
  /*! \brief The internal table mapping strings to a unique allocation. */
  std::unordered_map<std::string, T> table;
  DiagnosticContext* ctx;

  /*! \brief Add the unique allocation. */
  void Add(const std::string& name, const T& t) {
    auto it = table.find(name);
    if (it != table.end()) {
      throw DuplicateKeyError("duplicate key name in intern table");
    } else {
      table.insert({name, t});
    }
  }

  /*! \brief Return the unique allocation. */
  Optional<T> Get(const std::string& name) const {
    auto it = table.find(name);
    if (it != table.end()) {
      return Optional<T>(it->second);
    } else {
      return Optional<T>();
    }
  }
};

GlobalVar AddOrGet(InternTable<GlobalVar>* table, const std::string& name) {
  auto var = table->Get(name);
  if (var) {
    return var.value();
  } else {
    auto gvar = GlobalVar(name);
    table->Add(name, gvar);
    return gvar;
  }
}

GlobalTypeVar AddOrGet(InternTable<GlobalTypeVar>* table, const std::string& name,
                       TypeKind kind = TypeKind::kType) {
  auto var = table->Get(name);
  if (var) {
    auto tvar = var.value();
    TypeKind& tvar_kind = const_cast<TypeKind&>(tvar->kind);
    tvar_kind = kind;
    return tvar;
  } else {
    auto gvar = GlobalTypeVar(name, kind);
    table->Add(name, gvar);
    return gvar;
  }
}

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

  /*! \brief The IRModule we are building. */
  IRModule module;

  /*! \brief The diagnostic context used for error reporting. */
  DiagnosticContext diag_ctx;

  const Source& source;

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

  /*! \brief The metadata section. */
  MetaTable meta_table;

  Parser(IRModule module, DiagnosticContext ctx, const Source& source, std::vector<Token> tokens,
         OperatorTable op_table, MetaTable table)
      : module(module),
        diag_ctx(ctx),
        source(source),
        pos(0),
        tokens(tokens),
        op_table(op_table),
        ignore_whitespace(true),
        meta_table(table) {
    InitializeGlobals();
    InitializeTypeDefs();
  }

  /*! If we are parsing into a module with previously loaded data types we need to
   * map constructor names and variable names in the global tables.
   */
  void InitializeTypeDefs() {
    for (auto pair : this->module->type_definitions) {
      type_names.Add(pair.first->name_hint, pair.first);
      for (auto ctor : pair.second->constructors) {
        ctors.Add(ctor->name_hint, ctor);
      }
    }
  }

  void InitializeGlobals() {
    for (auto pair : this->module->functions) {
      global_names.Add(pair.first->name_hint, pair.first);
    }
  }

  /*! \brief Examine the next token in the stream, the current parser is configured to be
   * whitespace insensitive so we will skip all whitespace or comment tokens. */
  Token Peek() {
    // For now we ignore all whitespace tokens and comments.
    // We can tweak this behavior later to enable white space sensitivity in the parser.
    while (pos < static_cast<int64_t>(tokens.size()) && ignore_whitespace &&
           (tokens.at(pos)->token_type == TokenType::kWhitespace ||
            tokens.at(pos)->token_type == TokenType::kNewline ||
            tokens.at(pos)->token_type == TokenType::kLineComment ||
            tokens.at(pos)->token_type == TokenType::kComment)) {
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
    ICHECK_GE(n, 1) << "lookahead is only valid when n >= 1";

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
      this->diag_ctx.EmitFatal(Diagnostic::Error(tokens[pos]->span)
                               << "expected a " << Pretty(token_type) << " found "
                               << Pretty(Peek()->token_type));
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
    VLOG(9) << "Parser::WhenMatch: Peek() == " << Peek();
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
    auto it = this->graph_ctx.find(graph_no);
    if (it != this->graph_ctx.end()) {
      return it->second;
    } else {
      LOG(FATAL) << "Local variable %" << graph_no << " has not yet been defined";
      throw;
    }
  }

  /*! \brief Bind a local variable in the expression scope.
   *
   * "x" -> Var("x"), these are needed to map from the raw string names
   * to unique variable nodes.
   * If a virtual device is specified, sets the virtual device of the variable.
   */
  Var BindVar(const std::string& name, const relay::Type& type_annotation,
              Optional<VirtualDevice> virtual_device = Optional<VirtualDevice>()) {
    auto var = Var(name, type_annotation);
    var->virtual_device_ = virtual_device.value_or(VirtualDevice::FullyUnconstrained());
    VLOG(1) << "Binding var named " << name << " to variable node " << PrettyPrint(var);
    this->expr_scopes.Add(name, var);
    return var;
  }

  /*! \brief Bind a local variable in the expression scope.
   *
   * "x" -> Var("x"), these are needed to map from the raw string names
   * to unique variable nodes.
   */
  Var BindFreeVar(const std::string& name, const relay::Type& type_annotation) {
    auto var = Var(name, type_annotation);
    this->expr_scopes.AddFreeVar(name, var);
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
      diag_ctx.Emit(Diagnostic::Error(local->span)
                    << "this local variable has not been previously declared");
    }
    return var;
  }

  /*! \brief Lookup a variable in the type scope.
   *
   * Note: all lookup methods take tokens intentionally for error reporting information.
   */
  TypeVar LookupTypeVar(const Token& ident) {
    auto var = this->type_scopes.Lookup(ident.ToString());
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
    if (token->token_type == TokenType::kInteger) {
      return support::IntImmToNDArray(Downcast<tvm::IntImm>(token->data));
    } else if (token->token_type == TokenType::kFloat) {
      return support::FloatImmToNDArray(Downcast<tvm::FloatImm>(token->data));
    } else {
      LOG(FATAL) << "internal error: should only call this function on numeric tokens";
    }
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
    return Bracket(TokenType::kOpenParen, TokenType::kCloseParen, parser);
  }

  /*! \brief Parse `{` parser() `}`. */
  template <typename R>
  R Block(std::function<R()> parser) {
    return Bracket(TokenType::kLCurly, TokenType::kRCurly, parser);
  }

  template <typename R>
  R WithSpan(std::function<R()> parser) {
    auto start_span = Peek()->span;
    VLOG(9) << "WithSpan: start_span = " << start_span;
    R ast = parser();
    if (ast.defined()) {
      // The token at the head of the stream is now 1 past where we parsed. So we find its start
      // position as its start and end, so that when we merge we only grow the spanned region
      // to the start of the current stream.
      auto span_pos = pos - 1;
      while ((tokens.at(span_pos)->token_type == TokenType::kWhitespace ||
              tokens.at(span_pos)->token_type == TokenType::kNewline ||
              tokens.at(span_pos)->token_type == TokenType::kLineComment ||
              tokens.at(span_pos)->token_type == TokenType::kComment)) {
        span_pos--;
      }
      auto end_token = tokens.at(span_pos);
      VLOG(9) << "WithSpan: end_span = " << end_token->span;
      ast->span = start_span.Merge(end_token->span);
    }
    return ast;
  }

  struct MetaRef {
    std::string type_key;
    uint64_t node_index;
    Span span;
    MetaRef(std::string type_key, uint64_t node_index, Span span)
        : type_key(type_key), node_index(node_index), span(span) {}
  };

  MetaRef MetaRefFromToken(const Token& tok) {
    Call ref = Downcast<Call>(tok->data);
    auto attrs = ref->attrs.as<MetaRefAttrs>();
    auto type_key = attrs->node_type_key;
    auto index = attrs->node_index;
    return MetaRef(type_key, index, ref->span);
  }

  /*! \brief Parse a meta reference of the form `meta[type_key][node_index]`.
   * For example `meta[relay.Constant][0]` references the first constant, `meta[relay.Constant][1]`
   * the second, and so on.
   */
  ObjectRef ParseMetaRef() {
    auto meta_ref_tok = Match(TokenType::kMetaReference);
    auto meta_ref = MetaRefFromToken(meta_ref_tok);
    auto it = this->meta_table.find(meta_ref.type_key);
    if (it != this->meta_table.end()) {
      auto nodes = (*it).second;
      if (meta_ref.node_index < nodes.size()) {
        return nodes[meta_ref.node_index];
      } else {
        this->diag_ctx.Emit(Diagnostic::Error(meta_ref.span)
                            << "the node index `" << meta_ref.node_index
                            << "` is out of bounds for `" << meta_ref.type_key << "`");
        return ObjectRef();
      }
    } else {
      this->diag_ctx.Emit(Diagnostic::Error(meta_ref.span)
                          << "no entry in the meta table for `" << meta_ref.type_key << "`");
      return ObjectRef();
    }
  }
  /*! \brief Parses a sequence beginning with a start token, separated by a seperator token, and
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
                         std::function<bool()> before_stop = nullptr) {
    VLOG(9) << "Parser::ParseSequence: start=" << ToString(start) << " sep=" << ToString(sep)
            << " stop=" << ToString(stop);
    Match(start);

    // This is for the empty arguments list case, if we have <start> <leftovers> <stop> token stream
    // we must parse leftovers, then match a stop token.
    if (before_stop) {
      auto did_parse = before_stop();
      if (did_parse) {
        Match(stop);
        return {};
      }
    }

    // This is the case in which we find an empty arguments lists and no leftovers.
    if (WhenMatch(stop)) {
      return Array<T>();
    } else {
      VLOG(9) << "Parser::ParseSequence: parse first";
      auto data = parse();
      Array<T> elements = {data};

      if (WhenMatch(stop)) {
        return elements;
        // parse '( expr ',' * ')'
      } else if (WhenMatch(sep)) {
        while (true) {
          VLOG(9) << "Parser::ParseSequence: parse element";
          if (WhenMatch(stop)) {
            break;
          } else {
            // If before stop is
            if (before_stop) {
              auto did_parse = before_stop();
              if (did_parse) {
                Match(stop);
                return elements;
              }
            }
            auto data = parse();
            WhenMatch(sep);
            elements.push_back(data);
          }
        }
        return elements;
      } else {
        auto next = Peek();
        this->diag_ctx.EmitFatal(Diagnostic::Error(next->span)
                                 << "expected a " << Pretty(stop) << " found  "
                                 << Pretty(next->token_type));
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

    Match(TokenType::kEndOfFile);

    for (auto type_def : defs.types) {
      module->AddTypeDef(type_def->header, type_def);
    }

    for (auto func : defs.funcs) {
      module->Add(func.global, func.function, true);
    }

    return module;
  }

  /*! \brief Parse the semantic versioning header. */
  SemVer ParseSemVer(bool required = true) {
    if (Peek()->token_type == TokenType::kVersion) {
      auto version = Match(TokenType::kVersion);
      // TODO(@jroesch): we currently only support 0.0.5.
      if (version.ToString() != "\"0.0.5\"") {
        this->diag_ctx.Emit(Diagnostic::Error(version->span)
                            << "invalid semantic version `" << version.ToString() << "`");
      }
    } else if (required) {
      this->diag_ctx.Emit(Diagnostic::Error(Peek()->span)
                          << "expected text format semantic version, found a  "
                          << PrettyPrint(Peek()));

      this->diag_ctx.Emit(Diagnostic::Help(Peek()->span)
                          << "you can annotate it as #[version = \"0.0.5\"]");
    }
    return SemVer(0, 0, 5);
  }

  /*! \brief Parse zero or more Relay definitions. */
  Definitions ParseDefinitions() {
    Definitions defs;

    while (true) {
      auto next = Peek();
      switch (next->token_type) {
        case TokenType::kDefn: {
          Consume(TokenType::kDefn);
          auto global_tok = Match(TokenType::kGlobal);
          auto global_name = global_tok.ToString();
          auto global = AddOrGet(&global_names, global_name);
          auto func = WithSpan<relay::Function>([&]() { return ParseFunctionDef(); });
          ICHECK(func->span.defined()) << "spans must be set in parser";
          defs.funcs.push_back(GlobalFunc(global, func));
          continue;
        }
        case TokenType::kTypeDef: {
          defs.types.push_back(ParseTypeDef());
          continue;
        }
        case TokenType::kExtern: {
          Consume(TokenType::kExtern);
          auto type_def = ParseTypeDef();
          if (type_def->constructors.size()) {
            diag_ctx.Emit(Diagnostic::Error(next->span)
                          << "an external type may not have any constructors");
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
    Match(TokenType::kTypeDef);
    // Parse the type's identifier.
    auto type_tok = Match(TokenType::kIdentifier);
    auto type_id = type_tok.ToString();
    auto type_global = AddOrGet(&type_names, type_id, TypeKind::kAdtHandle);

    Array<TypeVar> generics;

    bool should_pop = false;
    if (Peek()->token_type == TokenType::kLSquare) {
      // If we have generics we need to add a type scope.
      PushTypeScope();
      should_pop = true;
      generics = ParseSequence<TypeVar>(
          TokenType::kLSquare, TokenType::kComma, TokenType::kRSquare, [&]() {
            auto type_var_name = Match(TokenType::kIdentifier).ToString();
            return BindTypeVar(type_var_name, TypeKind::kType);
          });
    }

    Array<tvm::Constructor> ctors;
    if (Peek()->token_type == TokenType::kLCurly) {
      // Parse the list of constructors.
      ctors = ParseSequence<tvm::Constructor>(
          TokenType::kLCurly, TokenType::kComma, TokenType::kRCurly, [&]() {
            // First match the name of the constructor.
            auto ctor_tok = Match(TokenType::kIdentifier);
            auto ctor_name = ctor_tok.ToString();

            Constructor ctor;
            // Match the optional field list.
            if (Peek()->token_type != TokenType::kOpenParen) {
              ctor = tvm::Constructor(ctor_name, {}, type_global);
            } else {
              auto arg_types =
                  ParseSequence<Type>(TokenType::kOpenParen, TokenType::kComma,
                                      TokenType::kCloseParen, [&]() { return ParseType(); });
              ctor = tvm::Constructor(ctor_name, arg_types, type_global);
            }

            ICHECK(ctor.defined());

            try {
              this->ctors.Add(ctor_name, ctor);
            } catch (const DuplicateKeyError& e) {
              this->diag_ctx.EmitFatal(Diagnostic::Error(ctor_tok->span)
                                       << "a constructor with the name "
                                       << "`" << ctor_name << "` "
                                       << "was previously defined");
            }

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
    VLOG(9) << "Parser::ParseExpr";
    return WithSpan<Expr>([this] {
      std::vector<Expr> exprs;

      while (true) {
        VLOG(9) << "Parser::ParseExpr: parsing a single expression";
        auto next = Peek();
        switch (next->token_type) {
          // For graph or let, match first rhs, then invoke ParseBindingExpr
          // ParseBindingExpression then parse_lhs() parse_rhs() ';' continue
          case TokenType::kLCurly: {
            // NB: Might need to optimize to remove deep recursion.
            // Stack should only grow proportionally to the number of
            // nested scopes.
            // Parses `{` expression `}`.
            auto block = WithSpan<Expr>([&]() {
              return Bracket<Expr>(TokenType::kLCurly, TokenType::kRCurly, [&]() {
                PushScope();
                auto expr = ParseExpr();
                PopScopes(1);
                return expr;
              });
            });
            exprs.push_back(block);
            break;
          }
          case TokenType::kFreeVar: {
            Consume(TokenType::kFreeVar);
            auto var_token = Match(TokenType::kLocal);

            Type type;
            if (WhenMatch(TokenType::kColon)) {
              type = ParseType();
            } else {
              type = IncompleteType();
            }

            BindFreeVar(var_token.ToString(), type);
            break;
          }
          // Parses `let ...`;
          case TokenType::kLet:
            exprs.push_back(ParseBindingExpr());
            break;
          case TokenType::kMatch:
          case TokenType::kPartialMatch: {
            bool is_total = next->token_type == TokenType::kMatch;
            Consume(next->token_type);
            exprs.push_back(ParseMatch(is_total));
            break;
          }

          // %x ...
          case TokenType::kGraph:
            if (Lookahead(2)->token_type == TokenType::kEqual) {
              exprs.push_back(ParseBindingExpr());
              break;
            }
            // intentional fall through here.
          default: {
            exprs.push_back(ParseExprBinOp());
            break;
          }
        }

        if (!WhenMatch(TokenType::kSemicolon)) {
          break;
        }
      }

      ICHECK_GE(exprs.size(), 1);

      if (exprs.size() == 1) {
        // ICHECK(exprs[0].defined() && exprs[0]->span.defined())
        //   << "parser must set expression spans.\n"
        //   << exprs[0];
        return exprs[0];
      } else {
        auto body = exprs.back();
        exprs.pop_back();
        while (exprs.size()) {
          auto value = exprs.back();
          ICHECK(value->span.defined()) << "parser must set expression spans.";
          exprs.pop_back();
          body = relay::Let(Var("", IncompleteType()), value, body, value->span.Merge(body->span));
        }
        ICHECK(body->span.defined()) << "parser must set expression spans.";
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
    VLOG(9) << "Parser::ParseBindingExpr";
    std::vector<std::tuple<Var, Expr, Span>> bindings;
    int scopes = 0;

    while (true) {
      auto next = Peek();
      if (next->token_type == TokenType::kGraph && Lookahead(2)->token_type == TokenType::kEqual) {
        Match(TokenType::kGraph);
        Match(TokenType::kEqual);
        auto val = this->ParseExprBinOp();
        Match(TokenType::kSemicolon);
        AddGraphBinding(next, val);
      } else if (next->token_type == TokenType::kLet) {
        auto span = next->span;
        // Parse the 'let'.
        Consume(TokenType::kLet);

        // Parse the local '%<id>'.
        auto local_tok = Match(TokenType::kLocal);
        auto string = local_tok.ToString();

        // Parse the optional type annotation (':' <type>).
        Type type;
        if (WhenMatch(TokenType::kColon)) {
          type = ParseType();
        }

        auto var = BindVar(string, type);

        // Parse the '=';
        Match(TokenType::kEqual);

        // Parse the body, and the ';'.
        auto val = this->ParseExprBinOp();
        Consume(TokenType::kSemicolon);

        // Add the bindings to the local data structure.
        std::tuple<relay::Var, relay::Expr, Span> tuple(var, val, span);
        bindings.push_back(tuple);
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
            auto span = body->span.Merge(std::get<2>(*binding));
            body = relay::Let(std::get<0>(*binding), std::get<1>(*binding), body, span);
          }
          return body;
        }
      }
    }
  }

  /*! Parse a function definition without a leading keyword or identifier.
   *
   * Handles things of the form [T1, ..., TN](arg1: U1, ..., argN : UN) -> Ret { body }.
   */
  Function ParseFunctionDef() {
    VLOG(9) << "Parser::ParseFunctionDef";
    return WithSpan<Function>([&]() {
      PushScope();
      PushTypeScope();

      Array<TypeVar> generics;
      if (Peek()->token_type == TokenType::kLSquare) {
        generics = ParseSequence<TypeVar>(
            TokenType::kLSquare, TokenType::kComma, TokenType::kRSquare, [&]() {
              auto type_var_name = Match(TokenType::kIdentifier).ToString();
              return BindTypeVar(type_var_name, TypeKind::kType);
            });
      }

      Map<String, ObjectRef> raw_attrs;

      auto params = ParseSequence<Var>(
          TokenType::kOpenParen, TokenType::kComma, TokenType::kCloseParen,
          [&]() {
            auto token = Match(TokenType::kLocal);
            auto string = token.ToString();

            // The fake attributes where the virtual device is specified.
            VirtualDevice virtual_device;
            if (WhenMatch(TokenType::kLCurly)) {
              Map<String, ObjectRef> fake_attrs = ParseAttrs();
              VLOG(9) << "Fake attributes for function parameter: " << fake_attrs;
              Match(TokenType::kRCurly);
              if (fake_attrs.size() == 1 && fake_attrs.count(kVirtualDevice)) {
                ICHECK(fake_attrs[kVirtualDevice].as<VirtualDeviceNode>())
                    << "Expected the " << kVirtualDevice
                    << " to have type VirtualDeviceNode, but got " << virtual_device->GetTypeKey();
                virtual_device = Downcast<VirtualDevice>(fake_attrs[kVirtualDevice]);
              }
            }

            Type type;
            if (WhenMatch(TokenType::kColon)) {
              type = ParseType();
            }
            return BindVar(string, type, virtual_device);
          },
          [&] {
            auto is_ident = Lookahead(1)->token_type == TokenType::kIdentifier;
            auto next_is_equal = Lookahead(2)->token_type == TokenType::kEqual;

            if (is_ident && next_is_equal) {
              raw_attrs = ParseAttrs();
              return true;
            }

            return false;
          });

      Type ret_type;
      if (WhenMatch(TokenType::kMinus)) {
        Match(TokenType::kRAngle);
        ret_type = ParseType();
      }

      auto body = Block<Expr>([&]() { return ParseExpr(); });

      PopTypeScopes(1);
      PopScopes(1);

      // TODO(@jroesch): attributes should never be null, they should always be empty.
      if (raw_attrs.size()) {
        // Promote kVirtualDevice to first-class
        if (raw_attrs.count(kVirtualDevice)) {
          ObjectRef vid = raw_attrs.at(kVirtualDevice);
          ICHECK(vid.as<VirtualDeviceNode>())
              << "Expected the " << kVirtualDevice << " to have type VirtualDeviceNode, but got "
              << vid->GetTypeKey();

          DictAttrs attrs;
          // Don't fill the raw_attrs in if there's nothing other than kVirtualDevice in the
          // attributes
          if (raw_attrs.size() > 1) {
            raw_attrs.erase(kVirtualDevice);
            attrs = DictAttrs(raw_attrs);
          }
          Function func = relay::Function(params, body, ret_type, generics, attrs);
          func->virtual_device_ = vid;
          return func;
        } else {
          return relay::Function(params, body, ret_type, generics, DictAttrs(raw_attrs));
        }
      } else {
        return relay::Function(params, body, ret_type, generics, tvm::DictAttrs());
      }
    });
  }

  /*! \brief Parse an if-expression. */
  Expr ParseIf() {
    return WithSpan<Expr>([&]() {
      VLOG(9) << "Parser::ParseIf";
      Consume(TokenType::kIf);

      auto guard = WithSpan<Expr>([&] { return Parens<Expr>([&] { return ParseExpr(); }); });

      auto true_branch = Block<Expr>([&] {
        this->PushScope();
        auto expr = ParseExpr();
        this->PopScopes(1);
        return expr;
      });

      Match(TokenType::kElse);

      auto false_branch = Block<Expr>([&] {
        this->PushScope();
        auto expr = ParseExpr();
        this->PopScopes(1);
        return expr;
      });

      return relay::If(guard, true_branch, false_branch);
    });
  }

  /* This factors parsing a list of patterns for both tuples, and constructors. */
  Array<Pattern> ParsePatternList() {
    return ParseSequence<Pattern>(TokenType::kOpenParen, TokenType::kComma, TokenType::kCloseParen,
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
    VLOG(9) << "Parser::ParsePattern";
    auto next = Peek();
    switch (next->token_type) {
      case TokenType::kUnderscore: {
        Match(TokenType::kUnderscore);
        return PatternWildcard();
      }
      case TokenType::kLocal: {
        auto id = Match(TokenType::kLocal);
        Type type_annotation;
        if (WhenMatch(TokenType::kColon)) {
          type_annotation = ParseType();
        }
        auto var = BindVar(id.ToString(), type_annotation);
        return PatternVar(var);
      }
      case TokenType::kIdentifier: {
        auto id = Match(TokenType::kIdentifier);
        auto ctor = ctors.Get(id.ToString());
        if (!ctor) {
          diag_ctx.EmitFatal(
              // TODO(@jroesch): split into error and help
              // deal with multiple rendering
              Diagnostic::Error(id->span)
              << "undefined constructor name `" << id.ToString()
              << "`, perhaps you intended to write a"
              << "pattern variable, considering changing this to `%" << id.ToString() << "`");
        }
        if (Peek()->token_type == TokenType::kOpenParen) {
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
    Match(TokenType::kEqual);
    Consume(TokenType::kRAngle);
    auto expr = ParseExpr();
    PopScopes(1);
    return Clause(pattern, expr);
  }

  Expr ParseMatch(bool is_total) {
    return WithSpan<Expr>([&]() {
      Expr scrutinee = ParseAtomicExpr();

      Array<Clause> clauses =
          ParseSequence<Clause>(TokenType::kLCurly, TokenType::kComma, TokenType::kRCurly,
                                [&] { return ParseMatchArm(); });

      return relay::Match(scrutinee, clauses, is_total);
    });
  }

  Expr ParseExprBinOp() {
    VLOG(9) << "Parser::ParseExprBinOp";
    return WithSpan<Expr>([this] {
      // We must parse at least one expression, the default
      // case is that there is no operator and we will fall
      // through.
      std::vector<Expr> exprs;
      Expr expr = WithSpan<Expr>([this] { return ParseCallExpr(); });

      exprs.push_back(expr);

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

        Expr right = WithSpan<Expr>([this] { return ParseCallExpr(); });
        ICHECK(right->span.defined());

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
          ICHECK(new_op.op.defined()) << "a call op must be set " << new_op.op;
          exprs.push_back(
              relay::Call(new_op.op, {left, right}, Attrs(), {}, left->span.Merge(right->span)));
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
        ICHECK(new_op.op.defined()) << "a call op must be set " << new_op.op;
        exprs.push_back(
            relay::Call(new_op.op, {left, right}, Attrs(), {}, left->span.Merge(right->span)));
      }

      ICHECK_EQ(ops.size(), 0) << "No operations should be left on the operation stack.";

      ICHECK_EQ(exprs.size(), 1)
          << "Only a single expression should be left on the expression stack.";

      return exprs[0];
    });
  }

  ObjectRef ParseAttributeValue() {
    VLOG(9) << "Parser::ParseAttributeValue";
    auto next = Peek();
    switch (next->token_type) {
      case TokenType::kFloat:
      case TokenType::kInteger:
      case TokenType::kBoolean:
      case TokenType::kStringLiteral:
        return Match(next->token_type)->data;
      case TokenType::kMetaReference:
        return ParseMetaRef();
      case TokenType::kLSquare: {
        return ParseSequence<ObjectRef>(TokenType::kLSquare, TokenType::kComma, TokenType::kRSquare,
                                        [&]() { return ParseAttributeValue(); });
      }
      case TokenType::kOpenParen: {
        // TODO(@jroesch: need to figure out bracket vs. sequence)
        // return ParseSequence<ObjectRef>(TokenType::kOpenParen, TokenType::kComma,
        // TokenType::kCloseParen,
        //                                 [&]() { return ParseAttributeValue(); });
        return Bracket<ObjectRef>(TokenType::kOpenParen, TokenType::kCloseParen,
                                  [&]() { return ParseAttributeValue(); });
      }
      // TODO(@jroesch): not sure about this being the right way to handle nulls.
      case TokenType::kIdentifier: {
        if (auto text = next->data.as<tvm::String>()) {
          std::string id = text.value();
          if (id == "nullptr") {
            Match(TokenType::kIdentifier);
            return ObjectRef();
          }
          if (id == "None") {
            Match(TokenType::kIdentifier);
            return Optional<ObjectRef>();
          }
        }
      }
      default:
        return ParseAtomicExpr();
    }
  }

  Map<String, ObjectRef> ParseAttrs() {
    VLOG(9) << "Parser::ParseAttrs";
    Map<String, ObjectRef> kwargs;
    while (Peek()->token_type == TokenType::kIdentifier) {
      auto key = GetHierarchicalName(ParseHierarchicalName().data);
      Match(TokenType::kEqual);
      // TOOD(@jroesch): syntactically what do we allow to appear in attribute right hand side.
      auto value = ParseAttributeValue();
      // TODO(@jroesch): we need a robust way to handle this writing dtypes as strings in text
      // format is bad.
      kwargs.Set(key, value);
      WhenMatch(TokenType::kComma);
    }
    VLOG(9) << "Parser::ParseAttrs: kwargs=" << kwargs;
    return kwargs;
  }

  Expr ParseCallArgs(Expr op) {
    ICHECK(op.defined()) << "the operator must be defined";

    VLOG(9) << "Parser::ParseCallArgs";
    Attrs attrs;
    std::string op_key;
    bool is_op = false;

    if (auto op_node = op.as<OpNode>()) {
      is_op = true;
      op_key = op_node->attrs_type_key;
    }

    if (Peek()->token_type == TokenType::kOpenParen) {
      Array<Expr> args = ParseSequence<Expr>(
          TokenType::kOpenParen, TokenType::kComma, TokenType::kCloseParen,
          [&] { return ParseExpr(); },
          [&] {
            auto is_ident = Lookahead(1)->token_type == TokenType::kIdentifier;
            auto next_is_equal = Lookahead(2)->token_type == TokenType::kEqual;
            auto is_pretty_attrs = is_ident && next_is_equal;
            auto is_meta_next = Lookahead(1)->token_type == TokenType::kMetaReference;
            // TODO(@jroesch): might not handle trailing comma
            auto last_meta = Lookahead(2)->token_type == TokenType::kCloseParen;
            auto is_meta_attrs = is_meta_next && last_meta;

            if (is_pretty_attrs || is_meta_attrs) {
              if (is_meta_attrs) {
                auto meta_ref = ParseMetaRef();
                if (meta_ref.as<BaseAttrsNode>()) {
                  attrs = Downcast<Attrs>(meta_ref);
                } else {
                  // Not awesome parsing code here.
                  this->pos--;
                  return false;
                }
              } else {
                auto raw_attrs = ParseAttrs();
                if (is_op && op_key.size()) {
                  auto attr_obj = tvm::ReflectionVTable::Global()->CreateObject(op_key, raw_attrs);
                  ICHECK(attr_obj.defined());
                  attrs = Downcast<Attrs>(attr_obj);
                } else if (raw_attrs.count("attrs_type_key")) {
                  String attr_key = Downcast<String>(raw_attrs["attrs_type_key"]);
                  if (attr_key.size()) {
                    raw_attrs.erase("attrs_type_key");
                    auto attr_obj =
                        tvm::ReflectionVTable::Global()->CreateObject(attr_key, raw_attrs);
                    ICHECK(attr_obj.defined());
                    attrs = Downcast<Attrs>(attr_obj);
                  }
                } else {
                  this->diag_ctx.EmitFatal(Diagnostic::Error(op->span)
                                           << "unable to determine the 'attrs_type_key' with which "
                                              "to represent the call attributes for this operator");
                }
              }
              return true;
            }
            return false;
          });

      if (!attrs.defined()) {
        if (is_op && op_key.size()) {
          auto attr_obj = tvm::ReflectionVTable::Global()->CreateObject(op_key, {});
          ICHECK(attr_obj.defined());
          attrs = Downcast<Attrs>(attr_obj);
        }
      }

      // TODO(@jroesch): in a secondary pass adjust spans.
      return Expr(Call(op, args, attrs, {}));
    } else {
      return Expr();
    }

    return Expr();
  }

  Expr ParseCallExpr() {
    VLOG(9) << "Parser::ParseCallExpr";
    return WithSpan<Expr>([this] {
      Expr expr = ParseAtomicExpr();
      // Parse as many call args as possible, building up expression
      //
      // NB(@jroesch): this seems like a hack but in order to parse curried functions
      // and avoid complex grammar we will parse multiple call lists in a row.
      while (Peek()->token_type == TokenType::kOpenParen) {
        auto new_expr = ParseCallArgs(expr);

        if (new_expr.defined()) {
          expr = new_expr;
        } else {
          break;
        }
      }

      // We need a zero-arity case for constructors.
      if (auto ctor_node = expr.as<ConstructorNode>()) {
        if (ctor_node->inputs.size() == 0) {
          return Expr(Call(expr, {}));
        }
      }

      return expr;
    });
  }

  Expr GetOp(const std::string& op_name, const Span& span) {
    VLOG(9) << "op_name=" << op_name << " span=" << span;
    try {
      return Op::Get(op_name);
    } catch (const Error& e) {
      // we can relax this, but probably need to relax checks or return non-null here.
      this->diag_ctx.EmitFatal(Diagnostic::Error(span)
                               << "operator `" << op_name
                               << "` not found, perhaps you forgot to register it?");
      return Expr();
    }
  }

  Expr ParseAtomicExpr() {
    VLOG(9) << "Parser::ParseAtomicExpr";
    Expr expr = WithSpan<Expr>([this] {
      auto next = Peek();
      switch (next->token_type) {
        case TokenType::kInteger:
        case TokenType::kFloat: {
          Consume(next->token_type);
          auto number = NumberToNDArray(next);
          Expr e = Constant(number, next->span);
          ICHECK(e->span.defined()) << "constant spans must be defined";
          return e;
        }
        case TokenType::kBoolean: {
          Consume(TokenType::kBoolean);
          int64_t value = Downcast<tvm::Integer>(next->data).IntValue();
          Expr e = Constant(support::BoolToNDArray(value), next->span);
          ICHECK(e->span.defined()) << "constant spans must be defined";
          return e;
        }
        // Parse a local of the form `%x`.
        case TokenType::kLocal: {
          Consume(TokenType::kLocal);
          return Expr(LookupLocal(next));
        }
        // Parse a local of the form `@x`.
        case TokenType::kGlobal: {
          auto global_name = next.ToString();
          Consume(TokenType::kGlobal);
          auto global = AddOrGet(&global_names, global_name);
          return Expr(global);
        }
        // Parse a local of the form `x`.
        // Right now we fail to parse `x.y`.
        case TokenType::kIdentifier: {
          auto ctor = ctors.Get(next.ToString());
          if (ctor) {
            Consume(TokenType::kIdentifier);
            return Expr(ctor.value());
          } else {
            auto spanned_idents = ParseHierarchicalName();
            auto idents = spanned_idents.data;
            auto span = spanned_idents.span;
            return GetOp(GetHierarchicalName(idents), span);
          }
        }
        case TokenType::kGraph: {
          Consume(TokenType::kGraph);
          return LookupGraphBinding(next);
        }
        case TokenType::kMetaReference: {
          return Downcast<Expr>(ParseMetaRef());
        }
        case TokenType::kFn: {
          Consume(TokenType::kFn);
          Expr e = ParseFunctionDef();
          ICHECK(e->span.defined()) << "function spans must be defined.\n" << e;
          return e;
        }
        case TokenType::kIf: {
          Expr e = ParseIf();
          return e;
        }
        case TokenType::kRef: {
          Consume(TokenType::kRef);
          Match(TokenType::kOpenParen);
          auto ref_value = ParseExpr();
          Match(TokenType::kCloseParen);
          return static_cast<Expr>(RefCreate(ref_value));
        }
        case TokenType::kRefRead: {
          return WithSpan<Expr>([&]() {
            Consume(TokenType::kRefRead);
            Match(TokenType::kOpenParen);
            auto ref = ParseExpr();
            Match(TokenType::kCloseParen);
            return static_cast<Expr>(RefRead(ref));
          });
        }
        case TokenType::kRefWrite: {
          return WithSpan<Expr>([&]() {
            Consume(TokenType::kRefWrite);
            Match(TokenType::kOpenParen);
            auto ref = ParseExpr();
            Match(TokenType::kComma);
            auto value = ParseExpr();
            Match(TokenType::kCloseParen);
            return static_cast<Expr>(RefWrite(ref, value));
          });
        }
        case TokenType::kOpenParen: {
          Span sp = next->span;
          Consume(TokenType::kOpenParen);
          // parse '(' ')'
          if (WhenMatch(TokenType::kCloseParen)) {
            return Expr(Tuple(Array<Expr>()));
          } else {
            Expr subexpr = ParseExpr();
            // parse '(' expr ')'
            if (WhenMatch(TokenType::kCloseParen)) {
              return subexpr;
              // parse '( expr ',' * ')'
            } else if (WhenMatch(TokenType::kComma)) {
              Array<Expr> exprs = {subexpr};
              while (true) {
                if (WhenMatch(TokenType::kCloseParen)) {
                  break;
                } else {
                  auto element = ParseExpr();
                  auto comma = Peek();
                  if (WhenMatch(TokenType::kComma)) {
                    sp = sp.Merge(element->span.Merge(comma->span));
                  } else {
                    sp = sp.Merge(element->span);
                  }
                  exprs.push_back(element);
                }
              }
              Expr tuple = Tuple(exprs, sp);
              ICHECK(tuple->span.defined()) << "tuple span should be defined";
              return tuple;
            }
          }
        }
        default: {
          this->diag_ctx.EmitFatal(Diagnostic::Error(next->span)
                                   << "expected an expression found  " << Pretty(next->token_type));
          return Expr();
        }
      }
    });

    if (WhenMatch(TokenType::kPeriod)) {
      auto token = Match(TokenType::kInteger);
      auto index = token.ToNumber();
      auto span = token->span.Merge(expr->span);
      VLOG(9) << "Parser::ParseAtomicExpr: tuple get item";
      return relay::TupleGetItem(expr, index, span);
    } else {
      return expr;
    }
  }

  /*! \brief Parse a hierarchical name.
   *
   * The tokenizer produces a token stream of <id1> . <id2>
   * and so on for names of the form `nn.conv2d`.
   * Currently we only use string names everywhere instead
   * of a notion of a hierarchical name.
   *
   * The below utility reassembles a token stream into a
   * single stream inserting the required periods needed
   * to look up registered names.
   */
  Spanned<Array<String>> ParseHierarchicalName() {
    Array<String> idents;
    Span span;
    while (Peek()->token_type == TokenType::kIdentifier) {
      auto token = Peek();

      if (span.defined()) {
        span = span.Merge(token->span);
      } else {
        span = token->span;
      }

      auto name = token.ToString();
      idents.push_back(name);
      Consume(TokenType::kIdentifier);

      // Keep parsing while we see a trailing period.
      if (Peek()->token_type == TokenType::kPeriod) {
        Consume(TokenType::kPeriod);
        continue;
      } else {
        // No more periods means we are done!
        break;
      }
    }

    return Spanned<Array<String>>(idents, span);
  }

  std::string GetHierarchicalName(Array<String> idents) {
    ICHECK_NE(idents.size(), 0);
    std::stringstream hierarchical_name;
    int i = 0;
    int periods = idents.size() - 1;
    for (auto ident : idents) {
      hierarchical_name << ident;
      if (i < periods) {
        hierarchical_name << ".";
        i++;
      }
    }
    return hierarchical_name.str();
  }

  /*! \brief Parse a shape. */
  Array<tvm::PrimExpr> ParseShape() {
    auto dims = ParseSequence<tvm::PrimExpr>(
        TokenType::kOpenParen, TokenType::kComma, TokenType::kCloseParen, [&]() {
          tvm::PrimExpr dim;
          if (Peek()->token_type == TokenType::kMetaReference) {
            dim = Downcast<tvm::PrimExpr>(ParseMetaRef());
          } else if (WhenMatch(TokenType::kQuestion)) {
            dim = tvm::tir::Any();
          } else {
            dim = Downcast<tvm::PrimExpr>(Match(TokenType::kInteger)->data);
          }

          return dim;
        });
    return dims;
  }

  /*! \brief Parse a function type. */
  Type ParseFunctionType() {
    auto ty_params = ParseSequence<Type>(TokenType::kOpenParen, TokenType::kComma,
                                         TokenType::kCloseParen, [&]() { return ParseType(); });

    Match(TokenType::kMinus);
    Match(TokenType::kRAngle);
    auto ret_type = ParseType();

    return relay::FuncType(ty_params, ret_type, {}, {});
  }

  // Parses a user defined ADT or type variable.
  Type ParseNonPrimitiveType(const Token& tok) {
    return WithSpan<Type>([&]() {
      auto name = tok.ToString();
      Type head_type = LookupTypeVar(tok);

      if (!head_type.defined()) {
        // head_type = type_names.Get(name);
        head_type = AddOrGet(&type_names, name, TypeKind::kAdtHandle);
      }

      if (!head_type.defined()) {
        diag_ctx.EmitFatal(Diagnostic::Error(tok->span)
                           << "the type constructor `" << name << "` is undefined");
      }

      Array<Type> arg_types;
      if (Peek()->token_type == TokenType::kLSquare) {
        arg_types = ParseSequence<Type>(TokenType::kLSquare, TokenType::kComma, TokenType::kRSquare,
                                        [&]() { return ParseType(); });
      }

      if (arg_types.size()) {
        return static_cast<Type>(TypeCall(head_type, arg_types));
      } else {
        if (head_type.as<GlobalTypeVarNode>()) {
          return static_cast<Type>(TypeCall(head_type, {}));
        } else {
          return static_cast<Type>(head_type);
        }
      }
    });
  }

  /*! \brief Parses a TVM type.
   *
   * This matches either a `Tensor[shape, dtype]`, a user defined ADT, a tuple type,
   * a scalar type or an incomplete type `_`.
   */
  Type ParseType() {
    return WithSpan<Type>([&]() -> Type {
      auto tok = Peek();

      if (tok->token_type == TokenType::kOpenParen) {
        auto tys =
            ParseSequence<relay::Type>(TokenType::kOpenParen, TokenType::kComma,
                                       TokenType::kCloseParen, [&]() { return ParseType(); });
        return relay::TupleType(tys);
      } else if (WhenMatch(TokenType::kFn)) {
        return ParseFunctionType();
      } else if (WhenMatch(TokenType::kIdentifier)) {
        auto id = tok.ToString();
        if (id == "Tensor") {
          Match(TokenType::kLSquare);
          auto shape = ParseShape();
          Match(TokenType::kComma);
          auto dtype_tok = Match(TokenType::kIdentifier);
          auto dtype = DataType(String2DLDataType(dtype_tok.ToString()));
          Match(TokenType::kRSquare);
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
      } else if (WhenMatch(TokenType::kUnderscore)) {
        return IncompleteType();
      } else {
        this->diag_ctx.EmitFatal(Diagnostic::Error(tok->span)
                                 << "failed to parse type found " << tok);
        return Type();
      }
    });
  }

  template <typename R>
  R ConsumeWhitespace(std::function<R()> func) {
    auto old = this->ignore_whitespace;
    this->ignore_whitespace = true;
    while (tokens[pos]->token_type == TokenType::kWhitespace) {
      pos++;
    }
    auto res = func();
    this->ignore_whitespace = old;
    return res;
  }

  Map<String, Array<ObjectRef>> ParseMetadata() {
    if (Peek()->token_type == TokenType::kMetadata) {
      return Match(TokenType::kMetadata).ToMetadata();
    } else {
      return Map<String, Array<ObjectRef>>();
    }
  }

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

Parser InitParser(const std::string& file_name, const std::string& file_content,
                  const Optional<IRModule>& init_module, const MetaTable& init_meta_table) {
  VLOG(9) << "InitParser: file_name: " << file_name << "file_content_size: " << file_content.size();
  SourceName src_name = SourceName::Get(file_name);
  Source source(src_name, file_content);

  IRModule module;
  if (!init_module) {
    SourceMap source_map;
    module = IRModule({}, {}, {}, source_map);
  } else {
    module = init_module.value();
  }

  module->source_map.Add(source);

  auto diag_ctx = DiagnosticContext::Default(module);
  auto tokens_and_table = Tokenize(diag_ctx, source);

  auto tokens = tokens_and_table.first;
  MetaTable meta_data_table = tokens_and_table.second.ToMetadata();

  // Merge any entries in init_meta_table into anything captured in the #[metadata] section
  // of the file_content. Metadata references within file_content must use indexes which account
  // for this ordering.
  for (const auto& pair : init_meta_table) {
    Array<ObjectRef> items;
    if (meta_data_table.count(pair.first)) {
      items = meta_data_table[pair.first];
    }
    for (const auto& obj : pair.second) {
      items.push_back(obj);
    }
    meta_data_table.Set(pair.first, items);
  }

  return Parser(module, diag_ctx, source, tokens, DefaultOpTable(), std::move(meta_data_table));
}

IRModule ParseModule(const std::string& file_name, const std::string& file_content,
                     const Optional<IRModule>& init_module, const MetaTable& init_meta_table) {
  VLOG_CONTEXT << "ParseModule";
  VLOG(9) << "parsing and type-checking " << file_name;
  auto parser = InitParser(file_name, file_content, init_module, init_meta_table);
  auto mod = parser.ParseModule();
  ICHECK(mod.defined()) << "The parser must return a non-null module.";
  // NB(@jroesch): it is very important that we render any errors before we proceed
  // if there were any errors which allow the parser to proceed we must render them
  // here.
  parser.diag_ctx.Render();
  auto infer_type = tvm::relay::transform::InferType();
  ICHECK(infer_type.defined()) << "The type inferencer must be non-null.";
  return infer_type(mod);
}

Expr ParseExpr(const std::string& file_name, const std::string& file_content) {
  VLOG(9) << "ParseExpr";
  auto parser = InitParser(file_name, file_content, Optional<IRModule>(), MetaTable());
  parser.ParseSemVer(false);
  parser.PushScope();
  auto expr = parser.ParseExpr();
  parser.Match(TokenType::kEndOfFile);
  // NB(@jroesch): it is very important that we render any errors before we proceed
  // if there were any errors which allow the parser to proceed we must render them
  // here.
  parser.diag_ctx.Render();
  return expr;
}

/*!
 * \brief This pass pretty-prints mod then parses it back so as to establish spans and sources
 * for all Relay sub-expressions. This improves error and debugging diagnostics downstream for
 * modules constructed programaticaly rather than textually.
 */
Pass AnnotateSpans() {
  auto pass_func = [](const IRModule& mod, const PassContext& ctx) {
    String text = AsText(mod, /*show_meta_data=*/true);
    VLOG(1) << "AnnotateSpans intermediate text:" << std::endl << text;
    return ParseModule("GeneratedSource", text);
  };
  return CreateModulePass(pass_func, 0, "AnnotateSpans", {});
}

TVM_REGISTER_GLOBAL("relay.parser.ParseModuleInContext")
    .set_body_typed([](const std::string& file_name, const std::string& file_content,
                       const Optional<IRModule>& init_module, const MetaTable& init_meta_table) {
      return ParseModule(file_name, file_content, init_module, init_meta_table);
    });

TVM_REGISTER_GLOBAL("relay.parser.ParseModule").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK(args.size() >= 2 && args.size() <= 4) << "Expected 2-4 arguments, but got " << args.size();
  if (args.size() == 2) {
    *ret = ParseModule(args[0], args[1]);
  } else if (args.size() == 3) {
    *ret = ParseModule(args[0], args[1], args[2]);
  } else {
    *ret = ParseModule(args[0], args[1], args[2], args[3]);
  }
});

TVM_REGISTER_GLOBAL("relay.parser.ParseExpr")
    .set_body_typed([](tvm::String file_name, tvm::String file_content) {
      return ParseExpr(file_name, file_content);
    });

TVM_REGISTER_GLOBAL("relay._transform.AnnotateSpans").set_body_typed(AnnotateSpans);

}  // namespace relay
}  // namespace tvm
