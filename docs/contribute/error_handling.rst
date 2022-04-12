..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _error-handling-guide:

Error Handling Guide
====================

.. contents::
  :depth: 2
  :local:

TVM contains structured error classes to indicate specific types of error.
Please raise a specific error type when possible, so that users can
write code to handle a specific error category if necessary.
You can directly raise the specific error object in python.
In other languages like c++, you simply add ``<ErrorType>:`` prefix to
the error message(see below).

.. note::

   Please refer to :py:mod:`tvm.error` for the list of errors.

Raise a Specific Error in C++
-----------------------------
You can add ``<ErrorType>:`` prefix to your error message to
raise an error of the corresponding type.
Note that you do not have to add a new type
:py:class:`tvm.error.TVMError` will be raised by default when
there is no error type prefix in the message.
This mechanism works for both ``LOG(FATAL)`` and ``ICHECK`` macros.
The following code gives an example on how to do so.

.. code:: c

  // src/api_test.cc
  void ErrorTest(int x, int y) {
    ICHECK_EQ(x, y) << "ValueError: expect x and y to be equal."
    if (x == 1) {
      LOG(FATAL) << "InternalError: cannot reach here";
    }
  }

The above function is registered as PackedFunc into the python frontend,
under the name ``tvm._api_internal._ErrorTest``.
Here is what will happen if we call the registered function:

.. code::

  >>> import tvm
  >>> tvm.testing.ErrorTest(0, 1)
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/path/to/tvm/python/tvm/_ffi/_ctypes/function.py", line 190, in __call__
      raise get_last_ffi_error()
  ValueError: Traceback (most recent call last):
    [bt] (3) /path/to/tvm/build/libtvm.so(TVMFuncCall+0x48) [0x7fab500b8ca8]
    [bt] (2) /path/to/tvm/build/libtvm.so(+0x1c4126) [0x7fab4f7f5126]
    [bt] (1) /path/to/tvm/build/libtvm.so(+0x1ba2f8) [0x7fab4f7eb2f8]
    [bt] (0) /path/to/tvm/build/libtvm.so(+0x177d12) [0x7fab4f7a8d12]
    File "/path/to/tvm/src/api/api_test.cc", line 80
  ValueError: Check failed: x == y (0 vs. 1) : expect x and y to be equal.
  >>>
  >>> tvm.testing.ErrorTest(1, 1)
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/path/to/tvm/python/tvm/_ffi/_ctypes/function.py", line 190, in __call__
      raise get_last_ffi_error()
  tvm.error.InternalError: Traceback (most recent call last):
    [bt] (3) /path/to/tvm/build/libtvm.so(TVMFuncCall+0x48) [0x7fab500b8ca8]
    [bt] (2) /path/to/tvm/build/libtvm.so(+0x1c4126) [0x7fab4f7f5126]
    [bt] (1) /path/to/tvm/build/libtvm.so(+0x1ba35c) [0x7fab4f7eb35c]
    [bt] (0) /path/to/tvm/build/libtvm.so(+0x177d12) [0x7fab4f7a8d12]
    File "/path/to/tvm/src/api/api_test.cc", line 83
  InternalError: cannot reach here
  TVM hint: You hit an internal error. Please open a thread on https://discuss.tvm.ai/ to report it.

As you can see in the above example, TVM's ffi system combines
both the python and c++'s stacktrace into a single message, and generate the
corresponding error class automatically.


How to choose an Error Type
---------------------------
You can go through the error types are listed below, try to use common
sense and also refer to the choices in the existing code.
We try to keep a reasonable amount of error types.
If you feel there is a need to add a new error type, do the following steps:

- Send a RFC proposal with a description and usage examples in the current codebase.
- Add the new error type to :py:mod:`tvm.error` with clear documents.
- Update the list in this file to include the new error type.
- Change the code to use the new error type.

We also recommend to use less abstraction when creating the short error messages.
The code is more readable in this way, and also opens path to craft specific
error messages when necessary.

.. code:: python

   def preferred():
       # Very clear about what is being raised and what is the error message.
       raise OpNotImplemented("Operator relu is not implemented in the MXNet frontend")

   def _op_not_implemented(op_name):
       return OpNotImplemented("Operator {} is not implemented.").format(op_name)

   def not_preferred():
       # Introduces another level of indirection.
       raise _op_not_implemented("relu")

If we need to introduce a wrapper function that constructs multi-line error messages,
please put wrapper in the same file so other developers can look up the implementation easily.
