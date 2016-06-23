#include <nngraph/graph.h>

int main() {
  nngraph::any a = 1;
  LOG(INFO) << nngraph::get<int>(a);
  return 0;
}
