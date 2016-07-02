// Copyright (c) 2016 by Contributors
#include <nngraph/op.h>
#include <nngraph/graph.h>

int main() {
  using namespace nngraph;
  auto add = Op::Get("add");
  auto nick = Op::GetAttr<std::string>("nick_name");
  LOG(INFO) << "nick=" << nick[add];
  return 0;
}
