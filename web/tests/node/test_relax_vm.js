/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/* eslint-disable no-undef */
// Load Emscripten Module, need to change path to root/lib
const path = require("path");
const fs = require("fs");
const assert = require("assert");
const tvmjs = require("../../dist");

const wasmPath = tvmjs.wasmPath();
const wasmSource = fs.readFileSync(path.join(wasmPath, "test_relax.wasm"));

const tvm = new tvmjs.Instance(
  new WebAssembly.Module(wasmSource),
  tvmjs.createPolyfillWASI()
);

function randomArray(length, max) {
  return Array.apply(null, Array(length)).map(function () {
    return Math.random() * max;
  });
}

test("add one", () => {
  tvm.beginScope();
  // Load system library
  const vm = tvm.createVirtualMachine(tvm.cpu());
  // grab pre-loaded function
  const fadd = vm.getFunction("main");

  assert(tvm.isPackedFunc(fadd));
  const n = 124;
  const A = tvm.empty(n).copyFrom(randomArray(n, 1));
  const B = tvm.empty(n).copyFrom(randomArray(n, 1));

  // call the function.
  const C = fadd(A, B);
  const AA = A.toArray(); // retrieve values in js array
  const BB = B.toArray(); // retrieve values in js array
  const CC = C.toArray(); // retrieve values in js array
  // verify
  for (var i = 0; i < BB.length; ++i) {
    assert(Math.abs(CC[i] - (AA[i] + BB[i])) < 1e-5);
  }
  tvm.endScope();
  // assert auto release scope behavior
  assert(vm.mod.getHandle(false) == 0);
  assert(fadd._tvmPackedCell.getHandle(false) == 0);
});
