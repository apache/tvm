// Copyright (c) 2016 by Contributors
#include <nngraph/op.h>
#include <nngraph/graph.h>
#include <nngraph/tuple.h>
#include <nngraph/graph_attr_types.h>
#include <string>

void test_op() {
  using namespace nngraph;
  auto add = Op::Get("add");
  auto nick = Op::GetAttr<std::string>("nick_name");
  LOG(INFO) << "nick=" << nick[add];
}

void test_tuple() {
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


void test_graph() {
  nngraph::Graph g;
  g.DFSVisit([](const std::shared_ptr<const nngraph::Node>& n){
    });
}
int main() {
  test_tuple();
  return 0;
}
