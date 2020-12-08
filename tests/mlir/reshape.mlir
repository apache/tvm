// RUN: mlir-mrelay-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @const_fold_collapse_to_scalar
func @const_fold_collapse_to_scalar() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i32>
  %cst = mhlo.constant dense<42> : tensor<1x1xi32>
  %0 = "mhlo.reshape"(%cst) : (tensor<1x1xi32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_fold_collapse_to_tensor
func @const_fold_collapse_to_tensor() -> tensor<2xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<2xi32>
  %cst = mhlo.constant dense<42> : tensor<1x2xi32>
  %0 = "mhlo.reshape"(%cst) : (tensor<1x2xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @const_fold_expand
func @const_fold_expand() -> tensor<1xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<1xi32>
  %cst = mhlo.constant dense<42> : tensor<i32>
  %0 = "mhlo.reshape"(%cst) : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<1xi32>
}

