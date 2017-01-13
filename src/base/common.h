/*!
 *  Copyright (c) 2016 by Contributors
 * \file common.h
 * \brief Common utilities
 */
#ifndef TVM_BASE_COMMON_H_
#define TVM_BASE_COMMON_H_

#include <tvm/base.h>
#include <string>

namespace tvm {

inline std::string Type2String(const Type& t) {
  if (t.code()  ==Type::Handle) return "handle";
  std::ostringstream os;
  os << t;
  return os.str();
}

inline Type String2Type(std::string s) {
  std::istringstream is(s);
  halide_type_code_t code = Type::Int;
  if (s.substr(0, 3) == "int") {
    code = Type::Int; s = s.substr(3);
  } else if (s.substr(0, 4) == "uint") {
    code = Type::UInt; s = s.substr(4);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s == "handle") {
    return Type(Type::Handle, 0, 0);
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  int bits = 32, lanes = 1;
  if (sscanf(s.c_str(), "%dx%d", &bits, &lanes) == 0) {
    LOG(FATAL) << "unknown type " << s;
  }
  return Type(code, bits, lanes);
}

}  // namespace tvm
#endif  // TVM_BASE_COMMON_H_
