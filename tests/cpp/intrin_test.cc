#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/intrin.h>
#include <utility>

TVM_REGISTER_INTRIN(add)
.describe("add two data together")
.set_inplace(true);

TVM_REGISTER_INTRIN(add)
.set_attr<std::string>("nick_name", "plus");


TEST(Intrin, GetAttr) {
  using namespace tvm;
  auto add = Intrin::Get("add");
  auto nick = Intrin::GetAttr<std::string>("nick_name");
  bool inplace = add->detect_inplace();

  CHECK_EQ(nick[add], "plus");
  CHECK_EQ(inplace, true);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
