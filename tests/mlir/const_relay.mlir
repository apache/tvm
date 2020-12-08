module {
  func @const_func() -> tensor<2x3xf32> {
    %0 = "mrelay.const"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
