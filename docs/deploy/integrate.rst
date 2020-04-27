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

Integrate TVM into Your Project
===============================

TVM's runtime is designed to be lightweight and portable.
There are several ways you can integrate TVM into your project.

This article introduces possible ways to integrate TVM
as a JIT compiler to generate functions on your system.


DLPack Support
--------------

TVM's generated function follows the PackedFunc convention.
It is a function that can take positional arguments including
standard types such as float, integer, string.
The PackedFunc takes DLTensor pointer in `DLPack <https://github.com/dmlc/dlpack>`_ convention.
So the only thing you need to solve is to create a corresponding DLTensor object.



Integrate User Defined C++ Array
--------------------------------

The only thing we have to do in C++ is to convert your array to DLTensor and pass in its address as
``DLTensor*`` to the generated function.


## Integrate User Defined Python Array

Assume you have a python object ``MyArray``. There are three things that you need to do

- Add ``_tvm_tcode`` field to your array which returns ``tvm.TypeCode.ARRAY_HANDLE``
- Support ``_tvm_handle`` property in your object, which returns the address of DLTensor in python integer
- Register this class by ``tvm.register_extension``

.. code:: python

   # Example code
   import tvm

   class MyArray(object):
       _tvm_tcode = tvm.TypeCode.ARRAY_HANDLE

       @property
       def _tvm_handle(self):
           dltensor_addr = self.get_dltensor_addr()
           return dltensor_addr

       # You can put registration step in a separate file mypkg.tvm.py
       # and only optionally import that if you only want optional dependency.
  tvm.register_extension(MyArray)
