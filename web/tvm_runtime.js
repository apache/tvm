/**
 * TVM Javascript web runtime library.
 *
 * @projectname tvm
 * @version 0.1
 */
/* eslint no-unused-vars: "off" */
/* eslint no-unexpected-multiline: "off" */
/* eslint indent: "off" */
/**
 * TVM Runtime namespace.
 * Provide tvm_runtime.create to create a {@link tvm.TVMRuntime}.
 *
 * @namespace tvm_runtime
 */
var tvm_runtime = tvm_runtime || {};

/**
 * TVM root namespace.
 * The classes inside this namespace need to be constructed by factory functions.
 * Use {@link tvm_runtime}.create to get started.
 *
 * @namespace tvm
 */
(function() {
  /**
   * TVMRuntime object for interacting with TVM runtime.
   * This object can be constructed using {@link tvm_runtime}.create
   *
   * @class
   * @memberof tvm
   */
  function TVMRuntime() {
    // Utility function to throw error
    function throwError(message) {
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message;
    }
    var Module = this.Module;
    var Runtime = this.Runtime;
    if (typeof Module === "undefined") {
      throwError("Emscripten Module is not available");
    }
    // constants
    var SIZEOF_POINTER = 4;
    var SIZEOF_SIZE_T = 4;
    var SIZEOF_FLOAT = 4;
    var SIZEOF_INT = 4;
    var SIZEOF_INT8 = 1;
    var SIZEOF_INT64 = 8;
    var SIZEOF_DOUBLE = 8;
    var SIZEOF_TYPE = 4;
    var SIZEOF_CTX = SIZEOF_INT + SIZEOF_INT;
    var SIZEOF_TVMVALUE = SIZEOF_DOUBLE;
    var ARRAY_OFFSET_DATA = 0;
    var ARRAY_OFFSET_CTX = ARRAY_OFFSET_DATA + SIZEOF_POINTER;
    var ARRAY_OFFSET_DEV_TYPE = ARRAY_OFFSET_CTX;
    var ARRAY_OFFSET_DEV_ID = ARRAY_OFFSET_CTX + SIZEOF_INT;
    var ARRAY_OFFSET_NDIM = ARRAY_OFFSET_CTX + SIZEOF_CTX;
    var ARRAY_OFFSET_DTYPE = ARRAY_OFFSET_NDIM + SIZEOF_INT;
    var ARRAY_OFFSET_DTYPE_CODE = ARRAY_OFFSET_DTYPE;
    var ARRAY_OFFSET_DTYPE_BITS = ARRAY_OFFSET_DTYPE_CODE + SIZEOF_INT8;
    var ARRAY_OFFSET_DTYPE_LANES = ARRAY_OFFSET_DTYPE_BITS + SIZEOF_INT8;
    var ARRAY_OFFSET_SHAPE = ARRAY_OFFSET_DTYPE + SIZEOF_TYPE;
    var ARRAY_OFFSET_STRIDES = ARRAY_OFFSET_STRIDES + SIZEOF_POINTER;
    var ARRAY_OFFSET_BYTE_OFFSET = ARRAY_OFFSET_STRIDES + SIZEOF_POINTER;
    // Type codes
    var kInt = 0;
    var kUInt = 1;
    var kFloat = 2;
    var kHandle = 3;
    var kNull = 4;
    var kTVMType = 5;
    var kTVMContext = 6;
    var kArrayHandle = 7;
    var kNodeHandle = 8;
    var kModuleHandle = 9;
    var kFuncHandle = 10;
    var kStr = 11;
    var kBytes = 12;
    //-----------------------------------------
    // TVM CWrap library
    // ----------------------------------------
    var TVMGetLastError = Module.cwrap(
      "TVMGetLastError",
      "string", // const char*
      []);

    var TVMAPISetLastError = Module.cwrap
    ("TVMAPISetLastError",
     null,
     ["string" // const char*
     ]);

    var TVMModImport = Module.cwrap
    ("TVMModImport",
     "number",
     ["number", // TVMModuleHandle mod
      "number"  // TVMModuleHandle dep
     ]);

    var TVMModGetFunction = Module.cwrap
    ("TVMModGetFunction",
     "number",
     ["number", // TVMModuleHandle mod
      "string", // const char* func_name
      "number", // int query_imports
      "number"  // TVMFunctionHandle *out
     ]);

    var TVMModFree = Module.cwrap
    ("TVMModFree",
     "number",
     ["number"  // TVMModeHandle mod
     ]);

    var TVMFuncFree = Module.cwrap
    ("TVMFuncFree",
     "number",
     ["number"  // TVMFunctionHandle func
     ]);

    var TVMFuncCall = Module.cwrap
    ("TVMFuncCall",
     "number",
     ["number", // TVMFunctionHandle func
      "number", // TVMValue* arg_values
      "number", // int* arg_tcodes
      "number", // int num_args
      "number", // int ret_val
      "number"  // int ret_type_code
     ]);

    var TVMCFuncSetReturn = Module.cwrap
    ("TVMCFuncSetReturn",
     "number",
     ["number", // TVMRetValueHandle ret
      "number", // TVMValue* value
      "number", // int* type_code
      "number" // int num_ret
     ]);

    var TVMCbArgToReturn = Module.cwrap
    ("TVMCbArgToReturn",
     "number",
     ["number", // TVMValue* value
      "number"  // int code
     ]);

    var TVMFuncCreateFromCFunc = Module.cwrap
    ("TVMFuncCreateFromCFunc",
     "number",
     ["number", // TVMPackedCFunc func,
      "number", // void* resource_handle
      "number", // TVMPackedCFuncFinalizer fin
      "number"  // TVMFunctionHandle *out
     ]);

    var TVMFuncRegisterGlobal = Module.cwrap
    ("TVMFuncRegisterGlobal",
     "number",
     ["string", // name
      "number", // TVMFunctionHandle f
      "number"  // int override
     ]);

    var TVMFuncGetGlobal = Module.cwrap
    ("TVMFuncGetGlobal",
     "number",
     ["string", // const char* name
      "number"  // TVMFunctionHandle* out
     ]);

    var TVMFuncListGlobalNames = Module.cwrap
    ("TVMFuncListGlobalNames",
     "number",
     ["number", // int* out_size
      "number"  // const char*** out_array
     ]);


    var TVMArrayAlloc = Module.cwrap
    ("TVMArrayAlloc",
     "number",
     ["number", // const tvm_index_t* shape
      "number", // int ndim
      "number", // int dtype_code
      "number", // int dtype_bits
      "number", // int dtype_lanes
      "number", // int device_type
      "number", // int device_id
      "number"  // int TVMArrayHandle* out
     ]);

    var TVMArrayFree = Module.cwrap
    ("TVMArrayFree",
     "number",
     ["number"  // TVMArrayHandle handle
     ]);

    var TVMArrayCopyFromTo = Module.cwrap
    ("TVMArrayCopyFromTo",
     "number",
     ["number", // TVMArrayHandle from
      "number" // TVMArrayHandle to
     ]);

    var TVMArrayCopyFromBytes = Module.cwrap
    ("TVMArrayCopyFromBytes",
     "number",
     ["number", // TVMArrayHandle handle
      "number", // int data
      "number"  // size_t nbytes
     ]);

    var TVMArrayCopyToBytes = Module.cwrap
    ("TVMArrayCopyToBytes",
     "number",
     ["number", // TVMArrayHandle handle
      "number", // int data
      "number"  // size_t nbytes
     ]);

    //-----------------------------------------
    // Static utility functions
    // ----------------------------------------
    this.assert = function(condition, message) {
      if (!condition) {
        message = message || "assert failed";
        throwError(message);
      }
    };
    var CHECK = this.assert;

    function TVM_CALL(ret) {
      if (ret != 0) {
        throwError(TVMGetLastError());
      }
    }

    function CInt64ArrayToJS(ptr, size) {
      var ret = [];
      for (var i = 0; i < size; ++i) {
        ret.push(Module.getValue(ptr + i * SIZEOF_INT64, "i64"));
      }
      return ret;
    }

    function CStringToJS(ptr) {
      var ret = [];
      var ch = 1;
      while (ch != 0) {
        ch = Module.getValue(ptr, "i8");
        if (ch != 0) {
          ret.push(String.fromCharCode(ch));
        }
        ++ptr;
      }
      return ret.join("");
    }

    function CBytesToJS(ptr) {
      var data = Module.getValue(ptr, "*");
      var size = Module.getValue(ptr + SIZEOF_POINTER, "i32");
      var ret = new Uint8Array(new ArrayBuffer(size));
      ret.set(new Uint8Array(Module.HEAPU8.buffer, data, size));
      return ret;
    }

    function StringToUint8Array(str) {
      var arr = new Uint8Array(str.length);
      for(var i = 0; i < str.length; ++i) {
        arr[i] = str.charCodeAt(i);
      }
      return arr;
    }
    //-----------------------------------------
    // Class declarations
    // ----------------------------------------
    function CBuffer(nbytes) {
      this.data = Module._malloc(nbytes);
    }

    function RefTVMValue() {
      this.data = Module._malloc(SIZEOF_TVMVALUE);
    }

    function TVMArgs(nargs) {
      this.nargs = nargs;
      this.value = Module._malloc(SIZEOF_TVMVALUE * nargs);
      this.tcode = Module._malloc(SIZEOF_INT * nargs);
      this.temp = [];
    }

    function TVMType(code, bits, lanes) {
      this.code = code;
      this.bits = bits;
      this.lanes = lanes;
    }
    /**
     * TVM device context.
     * @class
     * @memberof tvm
     */
    function TVMContext(device_type, device_id) {
      this.device_type = device_type;
      this.device_id = device_id;
    }
    /**
     * TVM  n-dimensional array.
     *
     * Use {@link tvm.TVMRuntime}.empty to create an instance.
     * @class
     * @memberof tvm
     */
    function NDArray(handle) {
      this.handle = handle;
      this.ndim = Module.getValue(this.handle + ARRAY_OFFSET_NDIM, "i32");
      // shape
      var cshape = Module.getValue(this.handle + ARRAY_OFFSET_SHAPE, "*");
      this.shape = CInt64ArrayToJS(cshape, this.ndim);
      // dtype
      var code = Module.getValue(this.handle + ARRAY_OFFSET_DTYPE_CODE, "i8");
      var bits = Module.getValue(this.handle + ARRAY_OFFSET_DTYPE_BITS, "i8");
      var lanes = Module.getValue(this.handle + ARRAY_OFFSET_DTYPE_LANES, "i16");
      var dtype =  new TVMType(code, bits, lanes);
      this.dtype = dtype;
      this.BYTES_PER_ELEMENT = (dtype.bits * dtype.lanes / 8);
      // ctx
      var device_type = Module.getValue(this.handle + ARRAY_OFFSET_DEV_TYPE, "i32");
      var device_id = Module.getValue(this.handle + ARRAY_OFFSET_DEV_ID, "i32");
      this.context = new TVMContext(device_type, device_id);
      // byte_offset
      this.byteOffset = Module.getValue(this.handle + ARRAY_OFFSET_BYTE_OFFSET, "i64");
    }

    function TVMFunction(handle) {
      this.handle = handle;
    }
    /**
     * Module container of TVM generated functions.
     *
     * @class
     * @memberof tvm
     */
    function TVMModule(handle) {
      this.handle = handle;
    }
    /**
     * A typed scalar constant.
     * This can be used to pass number as integer types to tvm function.
     * Use {@link tvm.TVMRuntime}.constant to create an instance.
     * @class
     * @memberof tvm
     */
    function TVMConstant(value, dtype) {
      this.value = value;
      this.dtype = dtype;
    }
    //-----------------------------------------
    // Private Functions
    // ----------------------------------------
    function getTVMType(dtype) {
      if (dtype instanceof TVMType) return dtype;
      if (typeof dtype == "string") {
      var pattern = dtype;
        var code, bits = 32, lanes = 1;
        if (pattern.substring(0, 5) == "float") {
          pattern = pattern.substring(5, pattern.length);
          code = kFloat;
        } else if (pattern.substring(0, 3) == "int") {
          pattern = pattern.substring(3, pattern.length);
          code = kInt;
        } else if (pattern.substring(0, 4) == "uint") {
          pattern = pattern.substring(4, pattern.length);
          code = kUInt;
        } else if (pattern.substring(0, 6) == "handle") {
          pattern = pattern.substring(5, pattern.length);
          code = kHandle;
          bits = 64;
        } else {
          throw throwError("Unknown dtype " + dtype);
        }
        var arr = pattern.split("x");
        if (arr.length >= 1) {
          var parsed = parseInt(arr[0]);
          if (parsed == arr[0]) {
            bits = parsed;
          }
        }
        if (arr.length >= 2) {
          lanes = parseInt(arr[1]);
        }
        return new TVMType(code, bits, lanes);
      } else {
        throw throwError("Unknown dtype " + dtype);
      }
    }

    function TVMRetValueToJS(vptr, tcode) {
      switch (tcode) {
      case kInt:
      case kUInt: return Module.getValue(vptr, "i64");
      case kFloat: return Module.getValue(vptr, "double");
      case kFuncHandle: return makeTVMFunction(Module.getValue(vptr, "*"));
      case kModuleHandle: return new TVMModule(Module.getValue(vptr, "*"));
      case kNull: return null;
      case kStr: return CStringToJS(Module.getValue(vptr, "*"));
      case kBytes: return CBytesToJS(Module.getValue(vptr, "*"));
      default: throwError("Unsupported return type code=" + tcode);
      }
    }

    function makeTVMFunction(handle) {
      var func = new TVMFunction(handle);
      var ret = function () {
        // alloc
        var args = new TVMArgs(arguments.length);
        var rvalue = new RefTVMValue();
        var rtcode = new RefTVMValue();
        args.setArguments(arguments);
        TVM_CALL(TVMFuncCall(handle, args.value, args.tcode,
                             args.nargs, rvalue.data, rtcode.data));
        var rv = TVMRetValueToJS(rvalue.data, rtcode.asInt());
        // release
        args.release();
        rvalue.release();
        rtcode.release();
        return rv;
      };
      var release = function() {
        func.release();
      };
      ret._tvm_function = func;
      ret.release = release;
      return ret;
    }
    //-----------------------------------------
    // Javascript PackedCallback System
    // ----------------------------------------
    var funcTable = [0];
    var freeFuncId = [];

    function invokeCallback(arg_value, arg_tcode, nargs, ret, handle) {
      var args = [];
      for (var i = 0; i < nargs; ++i) {
        var vptr = arg_value + i * SIZEOF_TVMVALUE;
        var tcode = Module.getValue(arg_tcode + i * SIZEOF_INT, "i32");
        if (tcode == kNodeHandle ||
            tcode == kFuncHandle ||
            tcode == kModuleHandle) {
          TVM_CALL(TVMCbArgToReturn(vptr, tcode));
        }
        args.push(TVMRetValueToJS(vptr, tcode));
      }
      var rv = funcTable[handle].apply(null, args);
      if (typeof rv !== "undefined") {
        // alloc
        var rarg = new TVMArgs(1);
        rarg.setArguments([rv]);
        TVM_CALL(TVMCFuncSetReturn(ret, rarg.value, rarg.tcode, 1));
        // release
        rarg.release();
      }
      return 0;
    }
    function freeCallback(handle) {
      funcTable[handle] = 0;
      freeFuncId.push(handle);
    }
    var fptrInvokeCallback = null;
    var fptrFreeCallback = null;
    if (typeof Runtime !== "undefined") {
      fptrInvokeCallback = Runtime.addFunction(invokeCallback);
      fptrFreeCallback = Runtime.addFunction(freeCallback);
    }
    /**
     * Check if a function is TVM PackedFunc
     * @param {Function} f function to be checked.
     * @return {boolean} Whether f is PackedFunc
     */
    this.isPackedFunc = function(f) {
      return (typeof f._tvm_function !== "undefined");
    };
    var isPackedFunc = this.isPackedFunc;
    /**
     * Convert a javascript function to TVM function.
     * @param {Function} f javascript function.
     * @return {Function} The created TVMFunction.
     */
    this.convertFunc = function(f) {
      if (isPackedFunc(f)) return f;
      CHECK(fptrInvokeCallback !== null, "Emscripten Runtime is not available");
      var fid;
      if (freeFuncId.length != 0) {
        fid = freeFuncId.pop();
      } else {
        fid = funcTable.length;
        funcTable.push(0);
      }
      funcTable[fid] = f;
      // alloc
      var out = new RefTVMValue();
      TVM_CALL(TVMFuncCreateFromCFunc(
        fptrInvokeCallback, fid, fptrFreeCallback, out.data));
      var out_handle = out.asHandle();
      // release
      out.release();
      return makeTVMFunction(out_handle);
    };
    var convertFunc = this.convertFunc;
    //-----------------------------------------
    // Private Class declarations
    // ----------------------------------------
    CBuffer.prototype = {
      /**
       * Finalizer: resources from the object.
       */
      release : function() {
        if (this.data != 0) {
          Module._free(this.data);
          this.data = 0;
        }
      },
    };
    // RefTVMValue
    RefTVMValue.prototype = {
      /**
       * Finalizer: resources from the object.
       */
      release : function() {
        if (this.data != 0) {
          Module._free(this.data);
          this.data = 0;
        }
      },
      asInt : function() {
        return Module.getValue(this.data, "i32");
      },
      asInt64 : function() {
        return Module.getValue(this.data, "i64");
      },
      asDouble : function() {
        return Module.getValue(this.data, "double");
      },
      asHandle : function() {
        return Module.getValue(this.data, "*");
      }
    };
    // TVMArgs
    TVMArgs.prototype = {
      release : function() {
        if (this.value != 0) {
          Module._free(this.value);
          Module._free(this.tcode);
          this.value = 0;
          for (var i = 0; i< this.temp.length; ++i) {
            if (this.temp[i].release instanceof Function) {
              this.temp[i].release();
            }
          }
        }
      },
      setInt : function(index, value) {
        Module.setValue(this.tcode + index * SIZEOF_INT, kInt, "i32");
        Module.setValue(this.value + index * SIZEOF_TVMVALUE, value, "i64");
      },
      setDouble : function(index, value) {
        Module.setValue(this.tcode + index * SIZEOF_INT, kFloat, "i32");
        Module.setValue(this.value + index * SIZEOF_TVMVALUE, value, "double");
      },
      setHandle : function(index, value, tcode) {
        Module.setValue(this.tcode + index * SIZEOF_INT, tcode, "i32");
        Module.setValue(this.value + index * SIZEOF_TVMVALUE, value, "*");
      },
      setString : function(index, value) {
        var sdata = new CBuffer(value.length);
        Module.HEAPU8.set(StringToUint8Array(value), sdata.data);
        this.temp.push(sdata);
        Module.setValue(this.tcode + index * SIZEOF_INT, kStr, "i32");
        Module.setValue(this.value + index * SIZEOF_TVMVALUE, sdata.data, "*");
      },
      setBytes : function(index, value) {
        CHECK(value instanceof Uint8Array);
        var sdata = new CBuffer(value.length);
        var sheader = new CBuffer(SIZEOF_POINTER + SIZEOF_SIZE_T);
        Module.HEAPU8.set(new Uint8Array(value), sdata.data);
        Module.setValue(sheader.data, sdata.data, "*");
        Module.setValue(sheader.data + SIZEOF_POINTER, value.length, "i32");
        this.temp.push(sdata);
        this.temp.push(sheader);
        Module.setValue(this.tcode + index * SIZEOF_INT, kBytes, "i32");
        Module.setValue(this.value + index * SIZEOF_TVMVALUE, sheader.data, "*");
      },
      setArguments : function(args) {
        for (var i = 0; i < args.length; ++i) {
          var v = args[i];
          var tp = typeof v;
          if (v instanceof NDArray) {
            this.setHandle(i, v.handle, kArrayHandle);
          } else if (v instanceof TVMConstant) {
            var code = getTVMType(v.dtype).code;
            if (code == kInt || code == kUInt) {
              this.setInt(i, v.value);
            } else if (code == kFloat) {
              this.setDouble(i, v.value);
            } else {
              CHECK(code == kHandle);
              this.setHandle(i, v.value, kHandle);
            }
          } else if (tp == "number") {
            this.setDouble(i, v);
          } else if (typeof v._tvm_function !== "undefined") {
            this.setString(i, v._tvm_function.handle, kFuncHandle);
          } else if (v === null) {
            this.setHandle(i, 0, kNull);
          } else if (tp == "string") {
            this.setString(i, v);
          } else if (v instanceof Uint8Array) {
            this.setBytes(i, v);
          } else if (v instanceof Function) {
            v = convertFunc(v);
            this.temp.push(v);
            this.setHandle(i, v._tvm_function.handle, kFuncHandle);
          } else {
            throwError("Unsupported argument type " + tp);
          }
        }
      }
    };
    // TVMType
    var TYPE_CODE2STR = {
      0 : "int",
      1 : "uint",
      2 : "float",
      4 : "handle"
    };

    TVMType.prototype = {
      toString : function() {
        var ret = TYPE_CODE2STR[this.code] + this.bits.toString();
        if (this.lanes != 1) {
          return ret + "x" + this.lanes.toString();
        } else {
          return ret;
        }
      }
    };
    // TVMFunction
    TVMFunction.prototype = {
      release : function() {
        if (this.handle != 0) {
          TVM_CALL(TVMFuncFree(this.handle));
          this.handle = 0;
        }
      }
    };
    // TVMContext
    var CTX_MASK2STR = {
      1 : "cpu",
      2 : "gpu",
      4 : "opencl",
      8 : "metal",
      9 : "vpi"
    };
    var CTX_STR2MASK = {
      "cpu": 1,
      "gpu": 2,
      "cuda": 2,
      "cl": 4,
      "opencl": 4,
      "metal": 8,
      "vpi": 9
    };
    TVMContext.prototype = {
      toString : function() {
        return CTX_MASK2STR[this.device_type] + "(" + this.device_id.toString() + ")";
      }
    };
    //-----------------------------------------
    // Public Functions
    // ----------------------------------------
    /**
     * Construct a TVMContext given device type and id.
     *
     * @param {number} device_type, string or int, The device type.
     * @param {number} device_id, the device id.
     * @return {tvm.TVMContext} The created TVMContext
     */
    this.context = function(device_type, device_id) {
      if (typeof device_type == "string") {
        device_type = CTX_STR2MASK[device_type];
      }
      return new TVMContext(device_type, device_id);
    };
    var context = this.context;
    /**
     * Create empty ndarray with given shape.
     *
     * @param {Array.<number>} shape The shape of the array.
     * @param {string} dtype The data type of the array, optional, default="float32"
     * @param {tvm.TVMContext} ctx The context of the array, optional, default=cpu(0).
     * @return {tvm.NDArray} The created ndarray.
     */
    this.empty = function(shape, dtype, ctx) {
      dtype = (typeof dtype !== "undefined") ?  dtype: "float32";
      ctx = (typeof ctx !== "undefined") ?  ctx : context("cpu", 0);
      shape = (typeof shape == "number") ? [shape] : shape;
      // alloc
      var cshape = Module._malloc(SIZEOF_INT64 * shape.length);
      var out = new RefTVMValue();
      for (var i = 0; i < shape.length; ++i) {
        Module.setValue(cshape + i * SIZEOF_INT64, shape[i], "i64");
      }
      dtype = getTVMType(dtype);
      TVM_CALL(TVMArrayAlloc(cshape, shape.length,
                             dtype.code, dtype.bits, dtype.lanes,
                             ctx.device_type, ctx.device_id,
                             out.data));
      var out_handle = out.asHandle();
      // release
      Module._free(cshape);
      out.release();
      return new NDArray(out_handle);
    };
    /**
     * List all global function names in the TVM runtime.
     * @return {Array.<string>} List of global function names.
     */
    this.listGlobalFuncNames = function() {
      // alloc
      var out_size = new RefTVMValue();
      var out_array = new RefTVMValue();
      TVM_CALL(TVMFuncListGlobalNames(out_size.data, out_array.data));
      var length = out_size.asInt();
      var base = out_array.asHandle();
      var names = [];
      for (var i = 0 ; i < length; ++i) {
        names.push(
          CStringToJS(Module.getValue(base + i * SIZEOF_POINTER, "*")));
      }
      // release
      out_size.release();
      out_array.release();
      return names;
    };
    /**
     * Get a global function from TVM runtime.
     *
     * @param {string} The name of the function.
     * @return {Function} The corresponding function.
     */
    this.getGlobalFunc = function (name) {
      // alloc
      var out = new RefTVMValue();
      TVM_CALL(TVMFuncGetGlobal(name, out.data));
      var out_handle = out.asHandle();
      // release
      out.release();
      return makeTVMFunction(out_handle);
    };
    var getGlobalFunc = this.getGlobalFunc;
    /**
     * Register function to be global function in tvm runtime.
     * @param {string} name The name of the function.
     * @param {Function} f function to be registered.
     * @param {boolean} override Whether overwrite function in existing registry.
     */
    this.registerFunc = function(name, f, override) {
      f = convertFunc(f);
      override = (typeof override !== "undefined") ?  override: false;
      var ioverride = override ? 1 : 0;
      TVM_CALL(TVMFuncRegisterGlobal(name, f._tvm_function.handle, ioverride));
    };
    /**
     * Create a typed scalar constant.
     * This can be used to pass number as integer types to tvm function.
     *
     * @param {number} value The value of the data.
     * @param {string} dtype The data type.
     * @param {tvm.TVMConstant} The created typed scalar.
     */
    this.constant = function(value, dtype) {
      return new TVMConstant(value, dtype);
    };
    //-----------------------------------------
    // Wrap of TVM Functions.
    // ----------------------------------------
    var fGetSystemLib = getGlobalFunc("module._GetSystemLib");
    /**
     * Get system-wide library module singleton.
     * System lib is a global module that contains self register functions in startup.
     * @return {tvm.TVMModule} The system module singleton.
     */
    this.systemLib = function() {
      return fGetSystemLib();
    };
    //-----------------------------------------
    // Class defintions
    // ----------------------------------------
    // NDArray.
    NDArray.prototype = {
      /**
       * Finalizer: resources from the object.
       */
      release : function() {
        if (this.handle != 0) {
          TVM_CALL(TVMArrayFree(this.handle));
          this.handle = 0;
        }
      },
      /**
       * Copy data from another NDArray or javascript array.
       * The number of elements must match.
       *
       * @param {Array} data The source data array.
       */
      copyFrom : function(data) {
        if (data instanceof NDArray) {
          TVM_CALL(TVMArrayCopyFromTo(data.handle, this.handle));
        } else {
          var size = this.shape.reduce(function(a, b) { return a * b; }, 1);
          if (data.length != size) {
            throwError("data size and shape mismatch data.length" + data.length + " vs " + size);
          }
          if (this.dtype == "float32") {
            data = Float32Array.from(data);
          } else if (this.dtype == "float64") {
            data = Float64Array.from(data);
          } else if (this.dtype == "int32") {
            data = Int32Array.from(data);
          } else if (this.dtype == "int8") {
            data = Int8Array.from(data);
          } else if (this.dtype == "uint8") {
            data = Uint8Array.from(data);
          } else {
            throwError("Unsupported data type " + this.dtype);
          }
          return this.copyFromRawBytes(new Uint8Array(data.buffer));
        }
      },
      /**
       * Copy data from raw bytes.
       * @param {Uint8Array} data Uint8Array of bytes.
       */
      copyFromRawBytes : function(data) {
        var size = this.shape.reduce(function(a, b) { return a * b; }, 1);
        var dtype = getTVMType(this.dtype);
        var nbytes = this.BYTES_PER_ELEMENT * size;
        CHECK(data instanceof Uint8Array);
        CHECK(data.length == nbytes,
              "Data length and bytes do not match " + data.length +
              " vs " + nbytes);
        var temp = Module._malloc(nbytes);
        Module.HEAPU8.set(data, temp);
        TVM_CALL(TVMArrayCopyFromBytes(this.handle, temp, nbytes));
        Module._free(temp);
        return this;
      },
      /**
       * Return a copied Uint8Array of the raw bytes in the NDArray.
       * @return {Uint8Array} The created array.
       */
      asRawBytes : function() {
        var size = this.shape.reduce(function(a, b) { return a * b; }, 1);
        var nbytes = this.BYTES_PER_ELEMENT * size;
        var temp = Module._malloc(nbytes);
        TVM_CALL(TVMArrayCopyToBytes(this.handle, temp, nbytes));
        var ret = new Uint8Array(new ArrayBuffer(nbytes));
        ret.set(new Uint8Array(Module.HEAPU8.buffer, temp, nbytes));
        Module._free(temp);
        return ret;
      },
      /**
       * Return Array data content as javascript typed array.
       * @return {TypedArray} The created array.
       */
      asArray : function() {
        if (this.dtype == "float32") {
          return new Float32Array(this.asRawBytes().buffer);
        } else if (this.dtype == "float64") {
          return new Float64Array(this.asRawBytes().buffer);
        } else if (this.dtype == "int32") {
          return new Int32Array(this.asRawBytes().buffer);
        } else if (this.dtype == "int8") {
          return new Int8Array(this.asRawBytes().buffer);
        } else if (this.dtype == "uint8") {
          return new Uint8Array(this.asRawBytes().buffer);
        } else {
          throwError("Unsupported data type " + this.dtype);
        }
      }
    };

    TVMModule.prototype = {
      /**
       * Finalizer: resources from the object.
       */
      release : function() {
        if (this.handle != 0) {
          TVM_CALL(TVMModFree(this.handle));
          this.handle = 0;
        }
      },
      /**
       * Get function from the module.
       * @param {string} name The name of the function.
       * @return {Function} The correspondin function.
       */
      getFunction : function(name) {
        // alloc
        var out = new RefTVMValue();
        TVM_CALL(TVMModGetFunction(this.handle, name, 0, out.data));
        var out_handle = out.asHandle();
        // release
        out.release();
        if (out_handle == 0) {
          throwError("Module has no function " + name);
        }
        return makeTVMFunction(out_handle);
      },
      /**
       * Add module to the import list of current one.
       * @param {tvm.TVMModule} mod The other module to be imported.
       */
      import_module : function(mod) {
        CHECK(mod instanceof TVMModule, "mod must be instance of TVMModule");
        TVM_CALL(TVMModImport(this.handle, mod.handle));
      }
    };
    //-----------------------------------------
    // Static variables.
    // ----------------------------------------
    /** Float32 type */
    this.float32 = "float32";
    /** Int32 type */
    this.int32 = "int32";
  }
  /**
   * Create a TVM runtime given emscripten module.
   * @property {string} create
   * @memberof tvm_runtime
   * @param Module The emscripten module.
   * @param Runtime The emscripten runtime, optional
   * @return {tvm.TVMRuntime} The created TVM runtime.
   */
  this.create = function(Module, Runtime) {
    var tvm = {};
    tvm.Module = Module;
    if (typeof Runtime == "undefined") {
      Runtime = Module.Runtime;
    }
    tvm.Runtime = Runtime;
    TVMRuntime.apply(tvm);
    return tvm;
  };
}).apply(tvm_runtime);

// export things in node
if (typeof module !== "undefined" && module.exports) {
  module.exports = tvm_runtime;
}
