#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <nngraph/tuple.h>

TEST(Tuple, Basic) {
  using nngraph::Tuple;
  using nngraph::TShape;
  Tuple<int> x{1, 2, 3};
  Tuple<int> y{1, 2, 3, 5, 6};
  x = std::move(y);

  CHECK_EQ(x.ndim(), 5);
  Tuple<int> z{1, 2, 3, 5, 6};
  std::ostringstream os;
  os << z;
  CHECK_EQ(os.str(), "(1,2,3,5,6)");
  std::istringstream is(os.str());
  is >> y;
  CHECK_EQ(x, y);
  Tuple<nngraph::index_t> ss{1, 2, 3};
  TShape s = ss;
  s = std::move(ss);
  CHECK((s == TShape{1, 2, 3}));
}
