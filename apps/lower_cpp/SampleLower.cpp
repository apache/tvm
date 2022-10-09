//
// Created by sunhh on 22-9-2.
//
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
#include <tvm/driver/driver_api.h>

//using source from tutor/language/reduction.py

using namespace tvm;
using namespace tvm::te;

int main(int argc, char** argv){
    printf("enter main\n");
  auto n = var("n");
  Array<PrimExpr> shape;
  shape.push_back(n);

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");

  auto C = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "C");

  auto s = create_schedule({C->op});

  auto cAxis = C->op.as<ComputeOpNode>()->axis;

  IterVar bx, tx;
  s[C].split(cAxis[0], 64, &bx, &tx);

  auto args = Array<Tensor>({A, B, C});
  std::unordered_map<Tensor, Buffer> binds;

  auto target = Target("llvm");

  auto lowered = LowerSchedule(s, args, "func", binds,GlobalVarSupply(NameSupply("")));
  LOG(INFO) <<"\n"
      <<
      lowered;
  return 0;
}