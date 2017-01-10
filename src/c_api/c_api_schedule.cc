/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to schedule pass.
 * \file c_api_lang.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/schedule.h>
#include <tvm/schedule_pass.h>
#include "./c_api_registry.h"
#include "../schedule/graph.h"

namespace tvm {
namespace schedule {
using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

#define REGISTER_SCHEDULE_PASS1(PassName)                         \
  TVM_REGISTER_API(_schedule_## PassName)                         \
  .set_body([](const ArgStack& args,  RetValue *ret) {            \
      *ret = PassName(args.at(0));                                \
    })                                                            \

#define REGISTER_SCHEDULE_PASS2(PassName)                         \
  TVM_REGISTER_API(_schedule_## PassName)                         \
  .set_body([](const ArgStack& args,  RetValue *ret) {            \
      *ret = PassName(args.at(0), args.at(1));                    \
    })                                                            \


REGISTER_SCHEDULE_PASS1(InferBound);
REGISTER_SCHEDULE_PASS1(CreateReadGraph);
REGISTER_SCHEDULE_PASS2(PostDFSOrder);

}  // namespace schedule
}  // namespace tvm
