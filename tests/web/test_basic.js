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

// Load Emscripten Module, need to change path to root/build
const path = require("path");
process.chdir(path.join(__dirname, "../../build"));
var Module = require("../../build/libtvm_web_runtime.js");
// Bootstrap TVMruntime with emscripten module.
const tvm_runtime = require("../../web/tvm_runtime.js");
const tvm = tvm_runtime.create(Module);

// Basic fields.
tvm.assert(tvm.float32 == "float32");
tvm.assert(tvm.listGlobalFuncNames() !== "undefined");
var sysLib = tvm.systemLib();
tvm.assert(typeof sysLib.getFunction !== "undefined");
sysLib.release();

// Test ndarray
function testArrayCopy(dtype, arr) {
  var data = [1, 2, 3, 4, 5, 6];
  var a = tvm.empty([2, 3], dtype);
  a.copyFrom(data);
  var ret = a.asArray();
  tvm.assert(ret instanceof arr);
  tvm.assert(ret.toString() == arr.from(data));
  a.release();
}

testArrayCopy("float32", Float32Array);
testArrayCopy("int", Int32Array);
testArrayCopy("int8", Int8Array);
testArrayCopy("uint8", Uint8Array);
testArrayCopy("float64", Float64Array);

// Function registration
tvm.registerFunc("xyz", function(x, y) {
  return x + y;
});
