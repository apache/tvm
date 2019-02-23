/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/arithmetic/analyzer.cc
 */
#include <tvm/arithmetic.h>

namespace tvm {
namespace arith {

Analyzer::Analyzer()
    : const_int_bound(this) {
}

}  // namespace arith
}  // namespace tvm
