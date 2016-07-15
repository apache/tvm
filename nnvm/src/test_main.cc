// Copyright (c) 2016 by Contributors
#include <nnvm/op.h>
#include <nnvm/graph.h>
#include <nnvm/tuple.h>
#include <nnvm/c_api.h>
#include <nnvm/graph_attr_types.h>
#include <dmlc/timer.h>
#include <string>

void test_speed() {
  auto add = nnvm::Op::Get("add");
  double tstart = dmlc::GetTime();
  size_t rep = 1000;
  size_t n = 1000;
  std::unordered_map<std::string, const nnvm::Symbol*> tmp;
  std::vector<const nnvm::Symbol*> vec{2};
  std::string name = "xx";
  for (size_t t = 0; t < rep; ++t) {
    nnvm::Symbol s = nnvm::Symbol::CreateVariable("x");
    for (size_t i = 0; i < n; ++i) {
      nnvm::Symbol nw = nnvm::Symbol::CreateFunctor(add, {});
      vec[0] = &s;
      vec[1] =&s;
      tmp.clear();
      nw.Compose(vec, tmp, name);
      s = nw;
    }
  }
  double tend = dmlc::GetTime();
  LOG(INFO) << "compose speed = " << n * rep / (tend - tstart) << " ops/sec";
}

void test_node_speed() {
  using namespace nnvm;
  auto add = nnvm::Op::Get("add");
  double tstart = dmlc::GetTime();
  size_t rep = 1000;
  size_t n = 1000;
  for (size_t t = 0; t < rep; ++t) {
    nnvm::Symbol s = nnvm::Symbol::CreateVariable("x");
    for (size_t i = 0; i < n; ++i) {
      auto xx = NodeEntry{Node::Create(), 0, 0};
      NodeEntry x = s.outputs[0];
      xx.node->op = add;
      xx.node->inputs.emplace_back(x);
      xx.node->inputs.emplace_back(x);
      Symbol ss;
      ss.outputs.push_back(xx);
      s = ss;
    }
  }
  double tend = dmlc::GetTime();
  LOG(INFO) << "test_node_speed speed = " << n * rep / (tend - tstart) << " ops/sec";
}

void test_api_speed() {
  auto add = (void*)nnvm::Op::Get("add");  // NOLINT(*)
  double tstart = dmlc::GetTime();
  size_t rep = 1000;
  size_t n = 1000;
  std::unordered_map<std::string, const nnvm::Symbol*> tmp;
  std::vector<const nnvm::Symbol*> vec{2};
  std::string name = "xx";
  for (size_t t = 0; t < rep; ++t) {
    SymbolHandle s;
    NNSymbolCreateVariable("xx", &s);
    for (size_t i = 0; i < n; ++i) {
      SymbolHandle arg[2];
      SymbolHandle ss;
      NNSymbolCreateAtomicSymbol(add, 0, nullptr, nullptr, &ss);
      arg[0] = s;
      arg[1] = s;
      NNSymbolCompose(ss, "nn", 2, nullptr, arg);
      s = ss;
    }
  }
  double tend = dmlc::GetTime();
  LOG(INFO) << "API compose speed = " << n * rep / (tend - tstart) << " ops/sec";
}

int main() {
  test_speed();
  test_node_speed();
  test_api_speed();
  return 0;
}
