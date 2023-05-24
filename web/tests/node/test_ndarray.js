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
const path = require("path");
const fs = require("fs");
const assert = require("assert");
const tvmjs = require("../../dist/tvmjs.bundle")

const wasmPath = tvmjs.wasmPath();
const wasmSource = fs.readFileSync(path.join(wasmPath, "tvmjs_runtime.wasm"));

let tvm = new tvmjs.Instance(
  new WebAssembly.Module(wasmSource),
  tvmjs.createPolyfillWASI()
);

// Basic fields.
assert(tvm.listGlobalFuncNames() !== undefined);

// Test ndarray
function testArrayCopy(dtype, arrayType) {
  let data = [1, 2, 3, 4, 5, 6];
  let a = tvm.empty([2, 3], dtype).copyFrom(data);

  assert(a.device.toString() == "cpu(0)");
  assert(a.shape[0] == 2 && a.shape[1] == 3);

  let ret = a.toArray();
  assert(ret instanceof arrayType);
  assert(ret.toString() == arrayType.from(data).toString());
}

test("array copy", () => {
  tvm.withNewScope(() => {
    testArrayCopy("float32", Float32Array);
    testArrayCopy("int", Int32Array);
    testArrayCopy("int8", Int8Array);
    testArrayCopy("uint8", Uint8Array);
    testArrayCopy("float64", Float64Array);
  });
});
