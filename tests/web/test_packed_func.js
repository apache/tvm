// Load Emscripten Module, need to change path to root/lib
const path = require("path");
process.chdir(path.join(__dirname, "../../lib"));
var Module = require("../../lib/libtvm_web_runtime.js");
// Bootstrap TVMruntime with emscripten module.
const tvm_runtime = require("../../web/tvm_runtime.js");
const tvm = tvm_runtime.create(Module);

function testGetGlobal() {
  var targs = [10, 10.0, "hello"]
  tvm.registerFunc("my_packed_func", function () {
    tvm.assert(Array.from(arguments).toString() == targs, "assert fail");
    return 10
  });
  var f = tvm.getGlobalFunc("my_packed_func")
  tvm.assert(tvm.isPackedFunc(f));
  y = f.apply(null, targs);
  tvm.assert(y == 10);
  f.release();
}


function testReturnFunc() {
  function addy(y) {
    function add(x) {
      return x + y;
    }
    return add;
  }
  var myf = tvm.convertFunc(addy);
  var f = myf(10);
  tvm.assert(tvm.isPackedFunc(f));
  tvm.assert(f(11) == 21);
  myf.release();
  f.release();
}

function testByteArray() {
  var a = new Uint8Array(3);
  a[0] = 1;
  a[1] = 2;
  function myfunc(ss){
    tvm.assert(ss instanceof Uint8Array);
    tvm.assert(ss.toString() == a);
  }
  f = tvm.convertFunc(myfunc);
  f(a);
  f.release();
}

testGetGlobal();
testReturnFunc();
testByteArray();
