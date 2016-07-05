#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <nngraph/op.h>
#include <utility>

NNGRAPH_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2)
.attr("inplace_pair", std::make_pair(0, 0));

NNGRAPH_REGISTER_OP(add)
.attr<std::string>("nick_name", "plus");


TEST(Op, GetAttr) {
  using namespace nngraph;
  auto add = Op::Get("add");
  auto nick = Op::GetAttr<std::string>("nick_name");

  CHECK_EQ(nick[add], "plus");
}
