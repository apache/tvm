// Load Emscripten Module, need to change path to root/lib
const path = require("path");
process.chdir(path.join(__dirname, "../../lib"));
var Module = require("../../lib/test_module.js");
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
