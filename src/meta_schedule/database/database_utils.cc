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
#include <iomanip>
#include <sstream>
#include <vector>

#include "../../support/str_escape.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

void JSONDumps(ObjectRef json_obj, std::ostringstream& os) {
  if (!json_obj.defined()) {
    os << "null";
  } else if (const auto* int_imm = json_obj.as<IntImmNode>()) {
    if (int_imm->dtype == DataType::Bool()) {
      if (int_imm->value) {
        os << "true";
      } else {
        os << "false";
      }
    } else {
      os << int_imm->value;
    }
  } else if (const auto* runtime_bool = json_obj.as<runtime::Bool::ContainerType>()) {
    os << (runtime_bool->value ? "true" : "false");
  } else if (const auto* runtime_int = json_obj.as<runtime::Int::ContainerType>()) {
    os << runtime_int->value;
  } else if (const auto* float_imm = json_obj.as<FloatImmNode>()) {
    os << std::setprecision(20) << float_imm->value;
  } else if (const auto* runtime_float = json_obj.as<runtime::Float::ContainerType>()) {
    os << std::setprecision(20) << runtime_float->value;
  } else if (const auto* str = json_obj.as<runtime::StringObj>()) {
    os << '"' << support::StrEscape(str->data, str->size) << '"';
  } else if (const auto* array = json_obj.as<runtime::ArrayNode>()) {
    os << "[";
    int n = array->size();
    for (int i = 0; i < n; ++i) {
      if (i != 0) {
        os << ",";
      }
      JSONDumps(array->at(i), os);
    }
    os << "]";
  } else if (const auto* dict = json_obj.as<runtime::MapNode>()) {
    int n = dict->size();
    std::vector<std::pair<String, ObjectRef>> key_values;
    key_values.reserve(n);
    for (const auto& kv : *dict) {
      if (auto key = kv.first.as<String>()) {
        key_values.emplace_back(key.value(), kv.second);
      } else {
        LOG(FATAL) << "TypeError: Only string keys are supported in JSON dumps, but got: "
                   << kv.first->GetTypeKey();
      }
    }
    std::sort(key_values.begin(), key_values.end());
    os << "{";
    for (int i = 0; i < n; ++i) {
      const auto& kv = key_values[i];
      if (i != 0) {
        os << ",";
      }
      os << '"' << support::StrEscape(kv.first->data, kv.first->size) << '"';
      os << ":";
      JSONDumps(kv.second, os);
    }
    os << "}";
  } else if (json_obj->IsInstance<tir::IndexMapNode>()) {
    JSONDumps(String(SaveJSON(json_obj)), os);
  } else {
    LOG(FATAL) << "TypeError: Unsupported type in JSON object: " << json_obj->GetTypeKey();
  }
}

std::string JSONDumps(ObjectRef json_obj) {
  std::ostringstream os;
  JSONDumps(json_obj, os);
  return os.str();
}

class JSONTokenizer {
 public:
  enum class TokenType : int32_t {
    kEOF = 0,          // end of file
    kNull = 1,         // null
    kTrue = 2,         // true
    kFalse = 3,        // false
    kLeftSquare = 4,   // [
    kRightSquare = 5,  // ]
    kLeftCurly = 6,    // {
    kRightCurly = 7,   // }
    kComma = 8,        // ,
    kColon = 9,        // :
    kInteger = 10,     // integers
    kFloat = 11,       // floating point numbers
    kString = 12,      // string
  };

  struct Token {
    TokenType type;
    ObjectRef value{nullptr};
  };

  explicit JSONTokenizer(const char* st, const char* ed) : cur_(st), end_(ed) {}

  Token Next() {
    for (; cur_ != end_ && std::isspace(*cur_); ++cur_) {
    }
    if (cur_ == end_) return Token{TokenType::kEOF};
    if (NextLeftSquare()) return Token{TokenType::kLeftSquare};
    if (NextRightSquare()) return Token{TokenType::kRightSquare};
    if (NextLeftCurly()) return Token{TokenType::kLeftCurly};
    if (NextRightCurly()) return Token{TokenType::kRightCurly};
    if (NextComma()) return Token{TokenType::kComma};
    if (NextColon()) return Token{TokenType::kColon};
    if (NextNull()) return Token{TokenType::kNull};
    if (NextTrue()) return Token{TokenType::kTrue};
    if (NextFalse()) return Token{TokenType::kFalse};
    Token token;
    if (NextString(&token)) return token;
    if (NextNumber(&token)) return token;
    LOG(FATAL) << "ValueError: Cannot tokenize: " << std::string(cur_, end_);
    throw;
  }

 private:
  bool NextLeftSquare() { return NextLiteral('['); }
  bool NextRightSquare() { return NextLiteral(']'); }
  bool NextLeftCurly() { return NextLiteral('{'); }
  bool NextRightCurly() { return NextLiteral('}'); }
  bool NextComma() { return NextLiteral(','); }
  bool NextColon() { return NextLiteral(':'); }
  bool NextNull() { return NextLiteral("null", 4); }
  bool NextTrue() { return NextLiteral("true", 4); }
  bool NextFalse() { return NextLiteral("false", 5); }

  bool NextNumber(Token* token) {
    using runtime::DataType;
    bool is_float = false;
    const char* st = cur_;
    for (; cur_ != end_; ++cur_) {
      if (std::isdigit(*cur_) || *cur_ == '+' || *cur_ == '-') {
        continue;
      } else if (*cur_ == '.' || *cur_ == 'e' || *cur_ == 'E') {
        is_float = true;
      } else {
        break;
      }
    }
    if (st == cur_) {
      return false;
    }
    std::string to_parse(st, cur_);
    if (!is_float) {
      try {
        *token = Token{TokenType::kInteger, runtime::Int(std::stoll(to_parse))};
      } catch (const std::invalid_argument& e) {
        LOG(WARNING) << "ValueError: Invalid argument to std::stoll: " << to_parse
                     << ". Details: " << e.what() << ". Switching to std::stod now.";
        is_float = true;
      } catch (const std::out_of_range& e) {
        LOG(WARNING) << "ValueError: Out-of-range for std::stoll: " << to_parse
                     << ". Details: " << e.what() << ". Switching to std::stod now.";
        is_float = true;
      }
    }
    if (is_float) {
      try {
        *token = Token{TokenType::kFloat, runtime::Float(std::stod(to_parse))};
      } catch (const std::invalid_argument& e) {
        LOG(INFO) << "ValueError: Invalid argument to std::stod: " << to_parse
                  << ". Details: " << e.what();
      } catch (const std::out_of_range& e) {
        LOG(INFO) << "ValueError: Out-of-range for std::stod: " << to_parse
                  << ". Details: " << e.what();
      }
    }
    return true;
  }

  bool NextString(Token* token) {
    if (cur_ == end_ || *cur_ != '"') return false;
    ++cur_;
    std::string str;
    for (; cur_ != end_ && *cur_ != '\"'; ++cur_) {
      if (*cur_ != '\\') {
        str.push_back(*cur_);
        continue;
      }
      ++cur_;
      if (cur_ == end_) {
        LOG(FATAL) << "ValueError: Unexpected end of string: \\";
        throw;
      }
      switch (*cur_) {
        case '\"':
          str.push_back('\"');
          break;
        case '\\':
          str.push_back('\\');
          break;
        case '/':
          str.push_back('/');
          break;
        case 'b':
          str.push_back('\b');
          break;
        case 'f':
          str.push_back('\f');
          break;
        case 'n':
          str.push_back('\n');
          break;
        case 'r':
          str.push_back('\r');
          break;
        case 't':
          str.push_back('\t');
          break;
        default:
          LOG(FATAL) << "ValueError: Unsupported escape sequence: \\" << *cur_
                     << ". record:" << std::string(cur_, end_);
      }
    }
    if (cur_ == end_) {
      LOG(FATAL) << "ValueError: Unexpected end of string";
    }
    ++cur_;
    *token = Token{TokenType::kString, String(str)};
    return true;
  }

  bool NextLiteral(char c) {
    if (cur_ != end_ && *cur_ == c) {
      ++cur_;
      return true;
    }
    return false;
  }

  bool NextLiteral(const char* str, int len) {
    if (cur_ + len <= end_ && std::strncmp(cur_, str, len) == 0) {
      cur_ += len;
      return true;
    }
    return false;
  }
  /*! \brief The current pointer */
  const char* cur_;
  /*! \brief End of the string */
  const char* end_;

  friend class JSONParser;
};

class JSONParser {
 public:
  using TokenType = JSONTokenizer::TokenType;
  using Token = JSONTokenizer::Token;

  explicit JSONParser(const char* st, const char* ed) : tokenizer_(st, ed) {}

  ObjectRef Get() {
    Token token = tokenizer_.Next();
    if (token.type == TokenType::kEOF) {
      return ObjectRef(nullptr);
    }
    return ParseObject(std::move(token));
  }

 private:
  ObjectRef ParseObject(Token token) {
    switch (token.type) {
      case TokenType::kNull:
        return ObjectRef(nullptr);
      case TokenType::kTrue:
        return Bool(true);
      case TokenType::kFalse:
        return Bool(false);
      case TokenType::kLeftSquare:
        return ParseArray();
      case TokenType::kLeftCurly:
        return ParseDict();
      case TokenType::kString:
      case TokenType::kInteger:
      case TokenType::kFloat:
        return token.value;
      case TokenType::kRightSquare:
        LOG(FATAL) << "ValueError: Unexpected token: ]";
      case TokenType::kRightCurly:
        LOG(FATAL) << "ValueError: Unexpected token: }";
      case TokenType::kComma:
        LOG(FATAL) << "ValueError: Unexpected token: ,";
      case TokenType::kColon:
        LOG(FATAL) << "ValueError: Unexpected token: :";
      case TokenType::kEOF:
        LOG(FATAL) << "ValueError: Unexpected EOF";
      default:
        throw;
    }
  }

  Array<ObjectRef> ParseArray() {
    bool is_first = true;
    Array<ObjectRef> results;
    for (;;) {
      Token token;
      if (is_first) {
        is_first = false;
        token = Token{TokenType::kComma};
      } else {
        token = tokenizer_.Next();
      }
      // Three cases overall:
      // - Case 1. 1 token: "]"
      // - Case 2. 2 tokens: ",", "]"
      // - Case 3. 2 tokens: ",", "obj"
      if (token.type == TokenType::kRightSquare) {  // Case 1
        break;
      } else if (token.type == TokenType::kComma) {
        token = tokenizer_.Next();
        if (token.type == TokenType::kRightSquare) {  // Case 2
          break;
        }
        // Case 3
        results.push_back(ParseObject(std::move(token)));
        continue;
      } else {
        LOG(FATAL) << "ValueError: Unexpected token before: " << tokenizer_.cur_;
      }
    }
    return results;
  }

  Map<String, ObjectRef> ParseDict() {
    bool is_first = true;
    Map<String, ObjectRef> results;
    for (;;) {
      Token token;
      if (is_first) {
        is_first = false;
        token = Token{TokenType::kComma};
      } else {
        token = tokenizer_.Next();
      }
      // Three cases overall:
      // - Case 1. 1 token: "}"
      // - Case 2. 2 tokens: ",", "}"
      // - Case 3. 2 tokens: ",", "key", ":", "value"
      if (token.type == TokenType::kRightCurly) {  // Case 1
        break;
      } else if (token.type == TokenType::kComma) {
        token = tokenizer_.Next();
        if (token.type == TokenType::kRightCurly) {  // Case 2
          break;
        }
        // Case 3
        ObjectRef key = ParseObject(std::move(token));
        ICHECK(key->IsInstance<StringObj>())
            << "ValueError: key must be a string, but gets: " << key;
        token = tokenizer_.Next();
        CHECK(token.type == TokenType::kColon)
            << "ValueError: Unexpected token before: " << tokenizer_.cur_;
        ObjectRef value = ParseObject(tokenizer_.Next());
        results.Set(Downcast<String>(key), value);
        continue;
      } else {
        LOG(FATAL) << "ValueError: Unexpected token before: " << tokenizer_.cur_;
      }
    }
    return results;
  }

  JSONTokenizer tokenizer_;
};

ObjectRef JSONLoads(std::string str) {
  const char* st = str.c_str();
  const char* ed = st + str.length();
  return JSONParser(st, ed).Get();
}

}  // namespace meta_schedule
}  // namespace tvm
