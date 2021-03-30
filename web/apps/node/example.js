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
/**
 * Example code to start the runtime.
 */
const path = require("path");
const fs = require("fs");
const tvmjs = require("../../dist");

const wasmPath = tvmjs.wasmPath();
const EmccWASI = require(path.join(wasmPath, "tvmjs_runtime.wasi.js"));
const wasmSource = fs.readFileSync(path.join(wasmPath, "tvmjs_runtime.wasm"));
// Here we pass the javascript module generated by emscripten as the
// LibraryProvider to provide WASI related libraries.
// the async version of the API.
tvmjs.instantiate(wasmSource, new EmccWASI())
.then((tvm) => {
    const log_info = tvm.getGlobalFunc("testing.log_info_str");
    log_info("hello world");
    // List all the global functions from the runtime.
    console.log("Runtime functions using EmccWASI\n", tvm.listGlobalFuncNames());
});
