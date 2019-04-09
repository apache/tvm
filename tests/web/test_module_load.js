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

// Load Emscripten Module, need to change path to root/lib
const path = require("path");
process.chdir(path.join(__dirname, "../../build"));
var Module = require("../../build/test_module.js");
// Bootstrap TVMruntime with emscripten module.
const tvm_runtime = require("../../web/tvm_runtime.js");
const tvm = tvm_runtime.create(Module);

// Load system library
var sysLib = tvm.systemLib();

function randomArray(length, max) {
  return Array.apply(null, Array(length)).map(function() {
    return Math.random() * max;
  });
}

function testAddOne() {
  // grab pre-loaded function
  var faddOne = sysLib.getFunction("add_one");
  tvm.assert(tvm.isPackedFunc(faddOne));
  var n = 124;
  var A = tvm.empty(n).copyFrom(randomArray(n, 1));
  var B = tvm.empty(n);
  // call the function.
  faddOne(A, B);
  // verify
  for (var i = 0; i < B.length; ++i) {
    tvm.assert(B[i] == A[i] + 1);
  }
  faddOne.release();
}

testAddOne();
sysLib.release();
console.log("Finish verifying test_module_load");
