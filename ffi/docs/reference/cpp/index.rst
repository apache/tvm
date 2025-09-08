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

C++ API
=======

This page contains the API reference for the C++ API. The full API index below
can be a bit dense, so we recommend the following tips first:

- Please read the :ref:`C++ Guide<cpp-guide>` for a high-level overview of the C++ API.

  - The C++ Guide and examples will likely be sufficient to get started with most use cases.

- The :ref:`cpp-key-classes` lists the key classes that are most commonly used.
- You can go to the Full API Index at the bottom of this page to access the full list of APIs.

  - We usually group the APIs by files. You can look at the file hierarchy in the
    full API index and navigate to the specific file to find the APIs in that file.

Header Organization
-------------------

The C++ APIs are organized into the following folders:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Folder
     - Description
   * - ``tvm/ffi/``
     - Core functionalities that support Function, Any, Object, etc.
   * - ``tvm/ffi/container/``
     - Additional container types such as Array, Map, Shape, Tensor, Variant ...
   * - ``tvm/ffi/reflection/``
     - Reflection support for function and type information registration.
   * - ``tvm/ffi/extra/``
     - Extra APIs that are built on top.


.. _cpp-key-classes:

Key Classes
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :cpp:class:`tvm::ffi::Function`
     - Type-erased function that implements the ABI.
   * - :cpp:class:`tvm::ffi::Any`
     - Type-erased container for any supported value.
   * - :cpp:class:`tvm::ffi::AnyView`
     - Lightweight view of Any without ownership.
   * - :cpp:class:`tvm::ffi::Object`
     - Base class for all heap-allocated FFI objects.
   * - :cpp:class:`tvm::ffi::ObjectRef`
     - Reference class for objects.
   * - :cpp:class:`tvm::ffi::Tensor`
     - Multi-dimensional tensor with DLPack support.
   * - :cpp:class:`tvm::ffi::Shape`
     - Tensor shape container.
   * - :cpp:class:`tvm::ffi::Module`
     - Dynamic library module that can load exported functions.
   * - :cpp:class:`tvm::ffi::String`
     - String type for FFI.
   * - :cpp:class:`tvm::ffi::Bytes`
     - Byte array type.
   * - :cpp:class:`tvm::ffi::Array`
     - Dynamic array container.
   * - :cpp:class:`tvm::ffi::Tuple`
     - Heterogeneous tuple container.
   * - :cpp:class:`tvm::ffi::Map`
     - Key-value map container.
   * - :cpp:class:`tvm::ffi::Optional`
     - Optional value wrapper.
   * - :cpp:class:`tvm::ffi::Variant`
     - Type-safe union container.



.. _cpp-full-api-index:

Full API Index
--------------

.. toctree::
   :maxdepth: 2

   generated/index.rst
