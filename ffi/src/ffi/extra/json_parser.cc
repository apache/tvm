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
 * \file src/ffi/json/parser.cc
 *
 * \brief A minimalistic JSON parser based on ffi values.
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cinttypes>
#include <limits>

namespace tvm {
namespace ffi {
namespace json {

/*!
 * \brief Helper class to parse a JSON string.
 *
 * Keep leaf level string/number parse also in context.
 */
class JSONParserContext {
 public:
  JSONParserContext(const char* begin, const char* end) : begin_(begin), cur_(begin), end_(end) {
    last_line_begin_ = cur_;
  }

  /*!
   * \brief Peek the current character.
   * \return The current character, or -1 if the end of the string is reached.
   */
  int Peek() const {
    return (cur_ != end_ ? static_cast<int>(*reinterpret_cast<const uint8_t*>(cur_)) : -1);
  }

  /*!
   * \brief Skip the next char that we know is not a space
   *
   * \note Caller must explicitly call SkipSpaces first or use
   *       Peek already that confirms char is not any space char.
   */
  void SkipNextAssumeNoSpace() { ++cur_; }

  /*!
   * \brief Get the current position.
   * \return The current position.
   */
  const char* GetCurrentPos() const { return cur_; }

  /*!
   * \brief Set the current position for better error message
   * \param pos The new position.
   * \note implementation can do it as no-op if needed
   */
  void SetCurrentPosForBetterErrorMsg(const char* pos) { cur_ = pos; }

  /*!
   * \brief Skip the space characters.
   * \note This function does not check if the end of the string is reached.
   */
  void SkipSpaces() {
    while (cur_ != end_) {
      if (!(*cur_ == ' ' || *cur_ == '\t' || *cur_ == '\n' || *cur_ == '\r')) {
        break;
      }
      if (*cur_ == '\n') {
        ++line_counter_;
        last_line_begin_ = cur_ + 1;
      }
      ++cur_;
    }
  }

  /*!
   * \brief Check if the next characters match the given string.
   * \param str The string to match.
   * \param len The length of the string.
   * \return True if the next characters match the given string, false otherwise.
   */
  bool MatchLiteral(const char* pattern, int len) {
    const char* pend = pattern + len;
    const char* ptr = pattern;
    for (; ptr != pend && cur_ != end_; ++ptr, ++cur_) {
      if (*ptr != *cur_) {
        return false;
      }
    }
    // we get to the end of the pattern and match is successful
    return ptr == pend;
  }

  /*
   * \brief Parse the next strin starting with a double quote.
   * \param out The output string.
   * \return Whether the next string parsing is successful.
   */
  bool NextString(json::Value* out) {
    // NOTE: we keep string parsing logic here to allow some special
    // optimizations for simple string that do not e
    const char* start_pos = cur_;
    TVM_FFI_ICHECK(*cur_ == '\"');
    // skip first double quote
    ++cur_;
    // the loop focuses on simple string without escape characters
    for (; cur_ != end_; ++cur_) {
      if (*cur_ == '\"') {
        *out = String(start_pos + 1, cur_ - start_pos - 1);
        ++cur_;
        return true;
      }
      if (*cur_ < ' ' || *cur_ == '\\') {
        // fallback to full string handling
        return this->NextStringWithFullHandling(out, start_pos);
      }
    }
    this->SetCurrentPosForBetterErrorMsg(start_pos);
    this->SetErrorUnterminatedString();
    return false;
  }

  /*!
   * \brief Parse the next number.
   * \param out The output number.
   * \return Whether the next number parsing is successful.
   */
  bool NextNumber(json::Value* out) {
    const char* start_pos = cur_;
    if (cur_ == end_) {
      this->SetErrorExpectingValue();
      return false;
    }
    // JSON number grammar:
    //
    // number = [ minus ] int [ frac ] [ exp ]
    // decimal-point = %x2E       ; .
    // digit1-9 = %x31-39         ; 1-9
    // e = %x65 / %x45            ; e E
    // exp = e [ minus / plus ] 1*DIGIT
    // frac = decimal-point 1*DIGIT
    std::string temp_buffer;
    bool maybe_int = true;
    // parse [minus], cross check for Infinity/NaN/-Infinity
    if (*cur_ == '-') {
      temp_buffer.push_back('-');
      ++cur_;
      if (cur_ != end_ && *cur_ == 'I') {
        if (this->MatchLiteral("Infinity", 8)) {
          *out = FastMathSafeNegInf();
          return true;
        } else {
          this->SetCurrentPosForBetterErrorMsg(start_pos);
          this->SetErrorExpectingValue();
          return false;
        }
      }
    } else if (*cur_ == 'I') {
      if (this->MatchLiteral("Infinity", 8)) {
        *out = FastMathSafePosInf();
        return true;
      } else {
        this->SetCurrentPosForBetterErrorMsg(start_pos);
        this->SetErrorExpectingValue();
        return false;
      }
    } else if (*cur_ == 'N') {
      if (this->MatchLiteral("NaN", 3)) {
        *out = FastMathSafeNaN();
        return true;
      } else {
        this->SetCurrentPosForBetterErrorMsg(start_pos);
        this->SetErrorExpectingValue();
        return false;
      }
    }
    // read in all parts that are possibly part of a number
    while (cur_ != end_) {
      char next_char = *cur_;
      if ((next_char >= '0' && next_char <= '9') || next_char == 'e' || next_char == 'E' ||
          next_char == '+' || next_char == '-' || next_char == '.') {
        temp_buffer.push_back(next_char);
        if (next_char == '.' || next_char == 'e' || next_char == 'E') {
          maybe_int = false;
        }
        ++cur_;
      } else {
        break;
      }
    }
    if (temp_buffer.empty()) {
      this->SetErrorExpectingValue();
      return false;
    }
    // parse from temp_buffer_
    if (maybe_int) {
      // now try to parse the number as int64
      char* end_ptr;
      errno = 0;
      intmax_t int_val = strtoimax(temp_buffer.data(), &end_ptr, 10);
      if (errno == 0 && int_val >= std::numeric_limits<int64_t>::min() &&
          int_val <= std::numeric_limits<int64_t>::max() &&
          end_ptr == temp_buffer.data() + temp_buffer.size()) {
        *out = static_cast<int64_t>(int_val);
        return true;
      }
    }
    {
      // now try to parse number as double
      char* end_ptr;
      errno = 0;
      double double_val = strtod(temp_buffer.data(), &end_ptr);
      if (errno == 0 && end_ptr == temp_buffer.data() + temp_buffer.size()) {
        *out = double_val;
        return true;
      } else {
        this->SetCurrentPosForBetterErrorMsg(start_pos);
        this->SetErrorExpectingValue();
        return false;
      }
    }
  }

  /*!
   * \brief Get the current line context.
   * \return The current line context.
   */
  String GetSyntaxErrorContext(std::string err_prefix) const {
    int64_t column = static_cast<int64_t>(cur_ - last_line_begin_) + 1;
    int64_t char_pos = static_cast<int64_t>(cur_ - begin_);
    if (err_prefix.empty()) {
      err_prefix = "Syntax error";
    }
    err_prefix += ": line " + std::to_string(line_counter_) + " column " + std::to_string(column) +
                  " (char " + std::to_string(char_pos) + ")";
    return String(err_prefix);
  }

  std::string FinalizeErrorMsg() {
    if (error_msg_.empty()) {
      SetErrorDefault();
    }
    return std::string(error_msg_);
  }

  void SetErrorDefault() { error_msg_ = GetSyntaxErrorContext("Syntax error near"); }

  void SetErrorExpectingValue() { error_msg_ = GetSyntaxErrorContext("Expecting value"); }

  void SetErrorInvalidControlCharacter() {
    error_msg_ = GetSyntaxErrorContext("Invalid control character at");
  }

  void SetErrorUnterminatedString() {
    error_msg_ = GetSyntaxErrorContext("Unterminated string starting at");
  }

  void SetErrorInvalidUnicodeEscape() {
    error_msg_ = GetSyntaxErrorContext("Invalid \\uXXXX escape");
  }

  void SetErrorInvalidSurrogatePair() {
    error_msg_ = GetSyntaxErrorContext("Invalid surrogate pair of \\uXXXX escapes");
  }

  void SetErrorInvalidEscape() { error_msg_ = GetSyntaxErrorContext("Invalid \\escape"); }

  void SetErrorExtraData() { error_msg_ = GetSyntaxErrorContext("Extra data"); }

  void SetErrorExpectingPropertyName() {
    error_msg_ = GetSyntaxErrorContext("Expecting property name enclosed in double quotes");
  }

  void SetErrorExpectingColon() { error_msg_ = GetSyntaxErrorContext("Expecting \':\' delimiter"); }

  void SetErrorExpectingComma() { error_msg_ = GetSyntaxErrorContext("Expecting \',\' delimiter"); }

 private:
  static double FastMathSafePosInf() {
#ifdef __FAST_MATH__
    union {
      uint64_t from;
      double to;
    } u;
    u.from = 0x7FF0000000000000ULL;  // write "from", read "to"
    return u.to;
#else
    return std::numeric_limits<double>::infinity();
#endif
  }

  static double FastMathSafeNegInf() {
#ifdef __FAST_MATH__
    union {
      uint64_t from;
      double to;
    } u;
    u.from = 0xFFF0000000000000ULL;  // write "from", read "to"
    return u.to;
#else
    return -std::numeric_limits<double>::infinity();
#endif
  }

  static double FastMathSafeNaN() {
#ifdef __FAST_MATH__
    union {
      uint64_t from;
      double to;
    } u;
    u.from = 0x7FF8000000000000ULL;  // write "from", read "to"
    return u.to;
#else
    return std::numeric_limits<double>::quiet_NaN();
#endif
  }

  // Full string parsing with escape and unicode handling
  bool NextStringWithFullHandling(Any* out, const char* start_pos) {
    // copy over the prefix that was already parsed
    std::string out_str(start_pos + 1, cur_ - start_pos - 1);
    while (cur_ != end_) {
      if (*cur_ < ' ') {
        this->SetErrorInvalidControlCharacter();
        return false;
      }
      if (*cur_ == '\"') {
        *out = String(std::move(out_str));
        ++cur_;
        return true;
      }
      if (*cur_ == '\\') {
        ++cur_;
        switch (*cur_) {
          // handle escape characters per JSON spec(RFC 8259)
#define HANDLE_ESCAPE_CHAR(pattern, val) \
  case pattern:                          \
    ++cur_;                              \
    out_str.push_back(val);              \
    break
          HANDLE_ESCAPE_CHAR('\"', '\"');
          HANDLE_ESCAPE_CHAR('\\', '\\');
          HANDLE_ESCAPE_CHAR('/', '/');
          HANDLE_ESCAPE_CHAR('b', '\b');
          HANDLE_ESCAPE_CHAR('f', '\f');
          HANDLE_ESCAPE_CHAR('n', '\n');
          HANDLE_ESCAPE_CHAR('r', '\r');
          HANDLE_ESCAPE_CHAR('t', '\t');
#undef HANDLE_ESCAPE_CHAR
          case 'u': {
            const char* escape_pos = cur_;
            // handle unicode code point
            ++cur_;
            int32_t first_i16, code_point = 0;
            if (!Parse4Hex(&first_i16)) {
              this->SetCurrentPosForBetterErrorMsg(escape_pos);
              this->SetErrorInvalidUnicodeEscape();
              return false;
            }
            // Check if the first i16 is a UTF-16 surrogate pair
            //
            // Surrogate pair encoding rule:
            // U' = yyyyyyyyyyxxxxxxxxxx  // U - 0x10000
            // W1 = 110110yyyyyyyyyy      // 0xD800 + yyyyyyyyyy
            // W2 = 110111xxxxxxxxxx      // 0xDC00 + xxxxxxxxxx
            //
            // Range of W1 and W2:
            // 0xD800 - 0xDBFF for W1
            // 0xDC00 - 0xDFFF for W2
            // both W1 and W2 fit into 0xD800 - 0xDFFF
            // Detect if the first i16 fit into range of W1/W2
            if (first_i16 >= 0xD800 && first_i16 <= 0xDFFF) {
              // we are in the surrogate pair range
              if (first_i16 >= 0xDC00) {
                this->SetCurrentPosForBetterErrorMsg(escape_pos);
                this->SetErrorInvalidSurrogatePair();
                // we need to return false instead because this range is for W2
                return false;
              }
              if (!this->MatchLiteral("\\u", 2)) {
                this->SetCurrentPosForBetterErrorMsg(escape_pos);
                this->SetErrorInvalidSurrogatePair();
                return false;
              }
              escape_pos = cur_;
              // get the value of the W2 (second i16)
              int32_t second_i16;
              if (!Parse4Hex(&second_i16)) {
                this->SetCurrentPosForBetterErrorMsg(escape_pos);
                this->SetErrorInvalidUnicodeEscape();
                return false;
              }
              if (!(second_i16 >= 0xDC00 && second_i16 <= 0xDFFF)) {
                this->SetCurrentPosForBetterErrorMsg(escape_pos);
                this->SetErrorInvalidSurrogatePair();
                return false;
              }
              // recover the code point
              code_point = ((first_i16 - 0xD800) << 10) + (second_i16 - 0xDC00) + 0x10000;
            } else {
              // not a surrogate case, just assign as code point
              code_point = first_i16;
            }
            // now need to push back the string based on UTF-8 encoding
            // UTF-8 encoding rule: four cases
            // ------------------------------------------------------------
            // Pattern                                | code point range
            // ------------------------------------------------------------
            // 0xxxxxxx                               | 0x0 - 0x7F
            // 110xxxxx 10xxxxxx                      | 0x80 - 0x7FF
            // 1110xxxx 10xxxxxx 10xxxxxx             | 0x800 - 0xFFFF
            // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx    | 0x10000 - end
            // ------------------------------------------------------------
            if (code_point < 0x80) {
              out_str.push_back(code_point);
            } else if (code_point < 0x800) {
              // first byte: 110xxxxx (5 effective bits)
              // second byte: 10xxxxxx (6 effecive bits)
              // shift by 6 bits to get the first bytes
              out_str.push_back(0xC0 | (code_point >> 6));
              // mask by 6 effective bits
              out_str.push_back(0x80 | (code_point & 0x3F));
            } else if (code_point < 0x10000) {
              // first byte: 1110xxxx (4 effective bits)
              // second byte: 10xxxxxx (6 effecive bits)
              // third byte: 10xxxxxx (6 effecive bits)
              // shift by 12 bits to get the first bytes
              out_str.push_back(0xE0 | (code_point >> 12));
              // shift by 6 bits to get the second bytes, mask by 6 effective bits
              out_str.push_back(0x80 | ((code_point >> 6) & 0x3F));
              // mask by 6 effective bits
              out_str.push_back(0x80 | (code_point & 0x3F));
            } else {
              // first byte: 11110xxx (3 effective bits)
              // second byte: 10xxxxxx (6 effecive bits)
              // third byte: 10xxxxxx (6 effecive bits)
              // fourth byte: 10xxxxxx (6 effecive bits)
              // shift by 18 bits to get the first bytes
              out_str.push_back(0xF0 | (code_point >> 18));
              // shift by 12 bits to get the second bytes, mask by 6 effective bits
              out_str.push_back(0x80 | ((code_point >> 12) & 0x3F));
              // shift by 6 bits to get the third bytes, mask by 6 effective bits
              out_str.push_back(0x80 | ((code_point >> 6) & 0x3F));
              // mask by 6 effective bits
              out_str.push_back(0x80 | (code_point & 0x3F));
            }
            break;
          }
          default: {
            this->SetErrorInvalidEscape();
            return false;
          }
        }
      } else {
        out_str.push_back(*cur_);
        ++cur_;
      }
    }
    this->SetCurrentPosForBetterErrorMsg(start_pos);
    this->SetErrorUnterminatedString();
    return false;
  }
  /*!
   * \brief Parse the four hex digits of a unicode code point per json spec.
   * \param out_i16 The output i16 number
   * \return True if four hex digits are parsed successfully, false otherwise.
   */
  bool Parse4Hex(int32_t* out_i16) {
    int32_t result = 0;
    for (int i = 0; i < 4; ++i, ++cur_) {
      int hex_val = *reinterpret_cast<const uint8_t*>(cur_);
      if (hex_val >= '0' && hex_val <= '9') {
        hex_val -= '0';
      } else if (hex_val >= 'a' && hex_val <= 'f') {
        hex_val -= 'a' - 0xa;
      } else if (hex_val >= 'A' && hex_val <= 'F') {
        hex_val -= 'A' - 0xa;
      } else {
        return false;
      }
      result = result * 16 + hex_val;
    }
    *out_i16 = result;
    return true;
  }

  /*! \brief The beginning of the string */
  const char* begin_;
  /*! \brief The current pointer */
  const char* cur_;
  /*! \brief End of the string */
  const char* end_;
  /*! \brief The beginning of the last line */
  const char* last_line_begin_;
  /*! \brief The error message */
  std::string error_msg_;
  /*! \brief The line counter */
  int64_t line_counter_{1};
};

class JSONParser {
 public:
  static json::Value Parse(const String& json_str, String* error_msg) {
    JSONParser parser(json_str);
    json::Value result;
    if (parser.ParseValue(&result) && parser.ParseTail()) {
      if (error_msg != nullptr) {
        *error_msg = String("");
      }
      return result;
    }
    if (error_msg != nullptr) {
      *error_msg = parser.ctx_.FinalizeErrorMsg();
      TVM_FFI_ICHECK(!error_msg->empty());
    } else {
      TVM_FFI_THROW(ValueError) << parser.ctx_.FinalizeErrorMsg();
    }
    // note that when we don't throw, error msg is set to indicate
    // an error happens
    return nullptr;
  }

 private:
  explicit JSONParser(String json_str) : ctx_(json_str.data(), json_str.data() + json_str.size()) {}

  bool ParseTail() {
    ctx_.SkipSpaces();
    // there are extra data in the tail
    if (ctx_.Peek() != -1) {
      ctx_.SetErrorExtraData();
      return false;
    }
    return true;
  }

  bool ParseValue(json::Value* out) {
    ctx_.SkipSpaces();
    // record start pos for cases where we might need to reset
    // current position for better error message
    auto start_pos = ctx_.GetCurrentPos();
    // check if the end of the string is reached
    switch (ctx_.Peek()) {
      case -1: {
        ctx_.SetErrorExpectingValue();
        return false;
      }
      case '{': {
        return ParseObject(out);
      }
      case '[': {
        return ParseArray(out);
      }
      case '\"': {
        return ctx_.NextString(out);
      }
      case 't': {
        ctx_.SkipNextAssumeNoSpace();
        if (ctx_.MatchLiteral("rue", 3)) {
          *out = true;
          return true;
        } else {
          ctx_.SetCurrentPosForBetterErrorMsg(start_pos);
          ctx_.SetErrorExpectingValue();
          return false;
        }
      }
      case 'f': {
        ctx_.SkipNextAssumeNoSpace();
        if (ctx_.MatchLiteral("alse", 4)) {
          *out = false;
          return true;
        } else {
          ctx_.SetCurrentPosForBetterErrorMsg(start_pos);
          ctx_.SetErrorExpectingValue();
          return false;
        }
      }
      case 'n': {
        ctx_.SkipNextAssumeNoSpace();
        if (ctx_.MatchLiteral("ull", 3)) {
          *out = nullptr;
          return true;
        } else {
          ctx_.SetCurrentPosForBetterErrorMsg(start_pos);
          ctx_.SetErrorExpectingValue();
          return false;
        }
      }
      default: {
        return ctx_.NextNumber(out);
      }
    }
    return false;
  }

  bool ParseObject(json::Value* out) {
    size_t stack_top = object_temp_stack_.size();
    json::Object result;
    ctx_.SkipNextAssumeNoSpace();
    ctx_.SkipSpaces();
    int next_char = ctx_.Peek();
    if (next_char == -1) {
      ctx_.SetErrorExpectingPropertyName();
      return false;
    }
    // empty object
    if (next_char == '}') {
      ctx_.SkipNextAssumeNoSpace();
      *out = json::Object();
      return true;
    }
    // non-empty object
    while ((next_char = ctx_.Peek()) != -1) {
      if (next_char != '\"') {
        ctx_.SetErrorExpectingPropertyName();
        return false;
      }
      json::Value key;
      if (!ctx_.NextString(&key)) return false;
      ctx_.SkipSpaces();
      if (ctx_.Peek() != ':') {
        ctx_.SetErrorExpectingColon();
        return false;
      }
      ctx_.SkipNextAssumeNoSpace();
      json::Value value;
      if (!ParseValue(&value)) return false;
      object_temp_stack_.emplace_back(key, value);
      // result.Set(key, value);
      ctx_.SkipSpaces();
      if (ctx_.Peek() == '}') {
        ctx_.SkipNextAssumeNoSpace();
        *out = json::Object(object_temp_stack_.begin() + stack_top, object_temp_stack_.end());
        // recover the stack to original state
        object_temp_stack_.resize(stack_top);
        return true;
      } else if (ctx_.Peek() == ',') {
        ctx_.SkipNextAssumeNoSpace();
        // must skip space so next iteration do not have to do so
        ctx_.SkipSpaces();
      } else {
        ctx_.SetErrorExpectingComma();
        return false;
      }
    }
    return false;
  }

  bool ParseArray(json::Value* out) {
    size_t stack_top = array_temp_stack_.size();
    ctx_.SkipNextAssumeNoSpace();
    ctx_.SkipSpaces();
    int next_char = ctx_.Peek();
    if (next_char == -1) {
      ctx_.SetErrorExpectingValue();
      return false;
    }
    // empty array
    if (next_char == ']') {
      ctx_.SkipNextAssumeNoSpace();
      *out = json::Array();
      return true;
    }
    // non-empty array
    while ((next_char = ctx_.Peek()) != -1) {
      json::Value value;
      // no need to skip space here because we already skipped space
      // at the beginning or in previous iteration
      if (!ParseValue(&value)) return false;
      array_temp_stack_.emplace_back(std::move(value));
      ctx_.SkipSpaces();
      next_char = ctx_.Peek();
      if (next_char == ',') {
        ctx_.SkipNextAssumeNoSpace();
        // must skip space so next iteration do not have to do so
        ctx_.SkipSpaces();
      } else if (next_char == ']') {
        ctx_.SkipNextAssumeNoSpace();
        *out = json::Array(array_temp_stack_.begin() + stack_top, array_temp_stack_.end());
        // recover the stack
        array_temp_stack_.resize(stack_top);
        return true;
      } else {
        ctx_.SetErrorExpectingComma();
        return false;
      }
    }
    return false;
  }

  JSONParserContext ctx_;
  // Temp stack for intermediate values
  // we first create a persistent stack to store the parsed values
  // then create the final array/object object with the precise size
  std::vector<Any> array_temp_stack_;
  std::vector<std::pair<Any, Any>> object_temp_stack_;
};

json::Value Parse(const String& json_str, String* error_msg) {
  return JSONParser::Parse(json_str, error_msg);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.json.Parse",
                        [](const String& json_str) { return json::Parse(json_str); });
});

}  // namespace json
}  // namespace ffi
}  // namespace tvm
