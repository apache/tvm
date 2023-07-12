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
  tvmjs.createPolyfillWASI());

test("object", () => {
  tvm.withNewScope(() => {
    let data = [1, 2, 3, 4, 5, 6];
    let a = tvm.empty([2, 3], "float32").copyFrom(data);

    let t = tvm.makeTVMArray([]);
    let b = tvm.makeTVMArray([a, t]);
    // assert b instanceof tvmjs.TVMArray
    assert(b instanceof tvmjs.TVMArray);
    assert(b.size() == 2);

    let t1 = b.get(1);
    assert(t1.getHandle() == t.getHandle());

    let s0 = tvm.makeString("hello world");
    assert(s0.toString() == "hello world");
    s0.dispose();

    let ret_string = tvm.getGlobalFunc("testing.ret_string");
    let s1 = ret_string("hello");
    assert(s1.toString() == "hello");
    ret_string.dispose();
    s1.dispose();
  });
});
