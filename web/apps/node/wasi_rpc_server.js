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
 * Example code to start the RPC server on nodejs using WASI
 */
const { WASI } = require("wasi");
const tvmjs = require("../../dist");

// Get import returns a fresh library in each call.
const getImports = () => {
  return new WASI({
    args: process.argv,
    env: process.env
  });
};

const proxyUrl = "ws://localhost:8888/ws";

new tvmjs.RPCServer(proxyUrl, "wasm", getImports, console.log);
