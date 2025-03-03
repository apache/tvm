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

// for now skip exception testing
// as it may not be compatible with asyncify
const exceptionEnabled = false;
const wasmPath = tvmjs.wasmPath();
const wasmSource = fs.readFileSync(path.join(wasmPath, "tvmjs_runtime.wasm"));

let tvm = new tvmjs.Instance(
  new WebAssembly.Module(wasmSource),
  tvmjs.createPolyfillWASI()
);


test("GetGlobal", () => {
  tvm.beginScope();
  let flist = tvm.listGlobalFuncNames();
  let faddOne = tvm.getGlobalFunc("testing.add_one");
  let fecho = tvm.getGlobalFunc("testing.echo");

  assert(faddOne(tvm.scalar(1, "int")) == 2);
  assert(faddOne(tvm.scalar(-1, "int")) == 0);

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

  tvm.beginScope();

  let arr = tvm.empty([2, 2]).copyFrom([1, 2, 3, 4]);
  let arr2 = fecho(arr);
  assert(arr.getHandle() == arr2.getHandle());
  assert(arr2.toArray().toString() == arr.toArray().toString());

  tvm.moveToParentScope(arr2);
  tvm.endScope();
  // test move to parent scope and tracking
  assert(arr.getHandle(false) == 0);
  assert(arr2.handle != 0);

  let mod = tvm.systemLib();
  let ret = fecho(mod);
  assert(ret.getHandle() == mod.getHandle());
  assert(flist.length != 0);
  tvm.endScope();

  // assert auto release scope behavior
  assert(mod.getHandle(false) == 0);
  assert(ret.getHandle(false) == 0);
  assert(arr2.getHandle(false) == 0);
  assert(fecho._tvmPackedCell.getHandle(false) == 0);
  assert(faddOne._tvmPackedCell.getHandle(false) == 0);
});

test("ReturnFunc", () => {
  tvm.beginScope();
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
  tvm.endScope();
});

test("RegisterGlobal", () => {
  tvm.beginScope();
  tvm.registerFunc("xyz", function (x, y) {
    return x + y;
  });

  let f = tvm.getGlobalFunc("xyz");
  assert(f(1, 2) == 3);
  f.dispose();

  let syslib = tvm.systemLib();
  syslib.dispose();
  tvm.endScope();
});

test("ExceptionPassing", () => {
  if (!exceptionEnabled) return;

  tvm.beginScope();
  tvm.registerFunc("throw_error", function (msg) {
    throw Error(msg);
  });
  let f = tvm.getGlobalFunc("throw_error");
  try {
    f("error-xyz");
    throw Error("error not caught");
  } catch (error) {
    assert(error.message.indexOf("error-xyz") != -1);
  }
  tvm.endScope();
});


test("AsyncifyFunc", async () => {
  if (!tvm.asyncifyEnabled()) {
    console.log("Skip asyncify tests as it is not enabled..");
    return;
  }
  tvm.beginScope();
  tvm.registerAsyncifyFunc("async_sleep_echo", async function (x) {
    await new Promise(resolve => setTimeout(resolve, 10));
    return x;
  });
  let fecho = tvm.wrapAsyncifyPackedFunc(
    tvm.getGlobalFunc("async_sleep_echo")
  );
  let fcall = tvm.wrapAsyncifyPackedFunc(
    tvm.getGlobalFunc("testing.call")
  );
  assert((await fecho(1)) == 1);
  assert((await fecho(2)) == 2);
  assert((await fcall(fecho, 2) == 2));
  tvm.endScope();
  assert(fecho._tvmPackedCell.getHandle(false) == 0);
  assert(fcall._tvmPackedCell.getHandle(false) == 0);
});

test("NDArrayCbArg", () => {
  tvm.beginScope();
  let use_count = tvm.getGlobalFunc("testing.object_use_count");
  let record = [];

  let fcheck = tvm.toPackedFunc(function (x, retain) {
    assert(use_count(x) == 2);
    assert(x.handle != 0);
    record.push(x);
    if (retain) {
      tvm.detachFromCurrentScope(x);
    }
  });

  let x = tvm.empty([2], "float32").copyFrom([1, 2]);
  assert(use_count(x) == 1);

  fcheck(x, 0);
  // auto-released when it is out of scope.
  assert(record[0].getHandle(false) == 0);

  assert(use_count(x) == 1);

  fcheck(x, 1);
  assert(use_count(x) == 2);
  assert(record[1].handle != 0);
  tvm.attachToCurrentScope(record[1]);
  tvm.endScope();
  assert(record[1].getHandle(false) == 0);
});

test("Logging", () => {
  tvm.beginScope();
  const log_info = tvm.getGlobalFunc("testing.log_info_str");
  log_info("helow world")
  log_info.dispose();
  tvm.endScope();
});
