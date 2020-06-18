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

// Javascript RPC server example
// Start and connect to websocket proxy.

// Load Emscripten Module, need to change path to root/lib
const path = require("path");
process.chdir(path.join(__dirname, "../lib"));
var Module = require("../lib/libtvm_web_runtime.js");
// Bootstrap TVMruntime with emscripten module.
const tvm_runtime = require("../web/tvm_runtime.js");
const tvm = tvm_runtime.create(Module);

var websock_proxy = "ws://localhost:9190/ws";
var num_sess = 100;
tvm.startRPCServer(websock_proxy, "js", num_sess)
