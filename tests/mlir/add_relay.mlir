module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 27 : i32}} {
  func @main(%arg0: tensor<10x5xi32>, %arg1: tensor<10x5xi32>) -> tensor<10x5xi32> attributes {tf.entry_function = {control_outputs = "", inputs = "input0,input1", outputs = "Add"}} {
    %0 = "mrelay.add"(%arg0, %arg1) : (tensor<10x5xi32>, tensor<10x5xi32>) -> tensor<10x5xi32>
    return %0 : tensor<10x5xi32>
  }
}
