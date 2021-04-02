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
const tvmjs = require("../../dist");

const wasmPath = tvmjs.wasmPath();
const EmccWASI = require(path.join(wasmPath, "tvmjs_runtime.wasi.js"));
const wasmSource = fs.readFileSync(path.join(wasmPath, "tvmjs_runtime.wasm"));

let tvm = new tvmjs.Instance(
  new WebAssembly.Module(wasmSource),
  new EmccWASI()
);

test("GetGlobal", () => {
  let flist = tvm.listGlobalFuncNames();
  let faddOne = tvm.getGlobalFunc("testing.add_one");
  let fecho = tvm.getGlobalFunc("testing.echo");

  assert(faddOne(tvm.scalar(1, "int")) == 2);
  // check function argument with different types.
  assert(fecho(1123) == 1123);
  assert(fecho("xyz") == "xyz");

  let bytes = new Uint8Array([1, 2, 3]);
  let rbytes = fecho(bytes);
  assert(rbytes.length == bytes.length);

  for (let i = 0; i < bytes.length; ++i) {
    assert(rbytes[i] == bytes[i]);
  }

  assert(fecho(undefined) == undefined);

  let arr = tvm.empty([2, 2]).copyFrom([1, 2, 3, 4]);
  let arr2 = fecho(arr);
  assert(arr.handle == arr2.handle);
  assert(arr2.toArray().toString() == arr.toArray().toString());

  let mod = tvm.systemLib();
  let ret = fecho(mod);
  assert(ret.handle == mod.handle);
  assert(flist.length != 0);

  mod.dispose();
  ret.dispose();
  arr.dispose();
  arr2.dispose();
  fecho.dispose();
  faddOne.dispose();
});

test("ReturnFunc", () => {
  function addy(y) {
    function add(x, z) {
      return x + y + z;
    }
    return add;
  }

  let fecho = tvm.getGlobalFunc("testing.echo");
  let myf = tvm.toPackedFunc(addy);
  assert(tvm.isPackedFunc(myf));
  let myf2 = tvm.toPackedFunc(myf);
  assert(myf2._tvmPackedCell.handle === myf._tvmPackedCell.handle);
  let f = myf(10);

  assert(tvm.isPackedFunc(f));
  assert(f(11, 0) == 21);
  assert(f("x", 1) == "x101");
  assert(f("x", "yz") == "x10yz");

  fecho.dispose();
  myf.dispose();
  myf2.dispose();
  // test multiple dispose.
  f.dispose();
  f.dispose();
});

test("RegisterGlobal", () => {
  tvm.registerFunc("xyz", function (x, y) {
    return x + y;
  });

  let f = tvm.getGlobalFunc("xyz");
  assert(f(1, 2) == 3);
  f.dispose();

  let syslib = tvm.systemLib();
  syslib.dispose();
});

test("NDArrayCbArg", () => {
  let use_count = tvm.getGlobalFunc("testing.object_use_count");

  let fcheck = tvm.toPackedFunc(function (x) {
    assert(use_count(x) == 2);
    x.dispose();
  });
  let x = tvm.empty([2], "float32").copyFrom([1, 2]);
  assert(use_count(x) == 1);
  fcheck(x);
  assert(use_count(x) == 1);
});

test("Logging", () => {
  const log_info = tvm.getGlobalFunc("testing.log_info_str");
  log_info("helow world")
  log_info.dispose();
});
