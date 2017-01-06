/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to schedule pass.
 * \file c_api_lang.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/schedule.h>
#include "../schedule/bound.h"
#include "./c_api_registry.h"

namespace tvm {
namespace schedule {
using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

#define REGISTER_SCHEDULE_PASS1(PassName)                         \
  TVM_REGISTER_API(_schedule_## PassName)                         \
  .set_body([](const ArgStack& args,  RetValue *ret) {            \
      *ret = PassName(args.at(0));                                \
    })                                                            \


REGISTER_SCHEDULE_PASS1(InferBound);

}  // namespace schedule
}  // namespace tvm
