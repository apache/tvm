//
// Created by ruxiliang on 2022/3/23.
//
#include <gtest/gtest.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>


using namespace tvm::runtime;

TEST(NDArray,MemoryManagement){
  auto handle = NDArray::Empty({200,20,20,20},DLDataType{kDLFloat,32,4},DLDevice{kDLCPU,0});
  EXPECT_EQ(handle.use_count(),1);
  auto handle_view = handle.CreateView({20,20,20,20},DLDataType{kDLFloat,32,4});
  EXPECT_EQ(handle.use_count(),2);
  handle_view.reset();
  EXPECT_EQ(handle.use_count(),1);
  handle.reset();
  EXPECT_EQ(handle.use_count(),0);
}