#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "gtest/gtest.h"

namespace tvm {
namespace runtime {
namespace hexagon {


TVM_REGISTER_GLOBAL("hexagon.run_all_tests").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RUN_ALL_TESTS();
});

}}}