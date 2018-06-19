#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Array, Expr) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Array<Expr> list{x, z, z};
  LOG(INFO) << list.size();
  LOG(INFO) << list[0];
  LOG(INFO) << list[1];
}

TEST(Array, Mutate) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Array<Expr> list{x, z, z};
  auto list2 = list;
  list.Set(1, x);
  CHECK(list[1].same_as(x));
  CHECK(list2[1].same_as(z));
}

TEST(Map, Expr) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  auto zz = z + 1;
  Map<Expr, Expr> dict{{x, z}, {z, 2}};
  CHECK(dict.size() == 2);
  CHECK(dict[x].same_as(z));
  CHECK(dict.count(z));
  CHECK(!dict.count(zz));
}

TEST(StrMap, Expr) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<std::string, Expr> dict{{"x", z}, {"z", 2}};
  CHECK(dict.size() == 2);
  CHECK(dict["x"].same_as(z));
}

TEST(Map, Mutate) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<Expr, Expr> dict{{x, z}, {z, 2}};
  auto zz = z + 1;
  CHECK(dict[x].same_as(z));
  dict.Set(x, zz);
  auto dict2 = dict;
  CHECK(dict2.count(z) == 1);
  dict.Set(zz, x);
  CHECK(dict2.count(zz) == 0);
  CHECK(dict.count(zz) == 1);

  auto it = dict.find(zz);
  CHECK(it != dict.end() && (*it).second.same_as(x));

  it = dict2.find(zz);
  CHECK(it == dict.end());

  LOG(INFO) << dict;
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
