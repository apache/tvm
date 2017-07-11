// Load Emscripten Module, need to change path to root/lib
const path = require("path");
process.chdir(path.join(__dirname, "../../lib"));
var Module = require("../../lib/libtvm_web_runtime.js");
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
