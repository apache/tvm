// RUN: mlir-mrelay-opt %s -mhlo-legalize-to-mrelay -mlir-print-stacktrace-on-diagnostic | FileCheck %s

// CHECK-LABEL: func @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xi32>)
// CHECK: return %[[ARG0]]

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 27 : i32}} {
  func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> attributes {tf.entry_function = {control_outputs = "", inputs = "input0,input1", outputs = "Add"}} {
    %0 = "mhlo.add"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    return %0 : tensor<10xi32>
  }
}
