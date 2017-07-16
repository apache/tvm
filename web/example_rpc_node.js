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
