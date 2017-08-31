#include <iostream>
#include <string>
#include <vector>

#include "../tvm/src/codegen/codegen_c.h"

#include "tvm/tvm.h"
#include "tvm/ir_pass.h"
#include "tvm/schedule_pass.h"

using namespace std;

int main(void) {
  tvm::Var M("M");
  tvm::Var N("N");
  tvm::Var K("K");
  tvm::Tensor I = tvm::placeholder({M, K}, tvm::Float(32), "I");
  tvm::Tensor W = tvm::placeholder({K, N}, tvm::Float(32), "W");
  tvm::IterVar rv = tvm::reduce_axis(tvm::Range{0, K}, "kk");

  auto O = tvm::compute(
    {M, N},
    [&](tvm::Var i, tvm::Var j) {
      return tvm::sum(I[i][rv] * W[rv][j], {rv});
    },
    "O");

  tvm::Array<tvm::Operation> ops({O->op});
  auto schedule = tvm::create_schedule(ops).normalize();
  auto bounds = tvm::schedule::InferBound(schedule);
  auto stmt = tvm::schedule::ScheduleOps(schedule, bounds);
  stmt = tvm::ir::Simplify(stmt);
  std::cout << stmt << std::endl;

  auto bufferI = tvm::BufferNode::make(
    tvm::Var("pI", tvm::Handle()),
    tvm::Float(32),
    tvm::Array<tvm::Expr>({M, K}),
    tvm::Array<tvm::Expr>({M, K}),
    tvm::Expr(0),
    std::string("A"),
    std::string("Input"),
    0,
    0
  );

  auto bufferW = tvm::BufferNode::make(
    tvm::Var("pW", tvm::Handle()),
    tvm::Float(32),
    tvm::Array<tvm::Expr>({K, N}),
    tvm::Array<tvm::Expr>({K, N}),
    tvm::Expr(0),
    std::string("B"),
    std::string("Input"),
    0,
    0
  );

  auto bufferO = tvm::BufferNode::make(
    tvm::Var("pO", tvm::Handle()),
    tvm::Float(32),
    tvm::Array<tvm::Expr>({M, N}),
    tvm::Array<tvm::Expr>({M, N}),
    tvm::Expr(0),
    std::string("C"),
    std::string("Output"),
    0,
    0
  );

  tvm::Array<tvm::NodeRef> tvmArgs({bufferI, bufferW, bufferO});
  auto api = tvm::ir::MakeAPI(stmt, "matmul", tvmArgs, 3,true);
  std::cout << api << std::endl;

  tvm::codegen::CodeGenC cg;
  cg.Init(false);
  cg.AddFunction(api);
  std::cout << cg.Finish() << std::endl;
}

