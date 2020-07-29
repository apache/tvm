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

Introduction to Module Serialization
====================================

When to deploy TVM runtime module, no matter whether it is CPU or GPU, TVM only needs one single dynamic
shared library. The key is our unified module serialization mechanism. This document will introduce TVM module
serialization format standard and implementation details.

*********************
Module Export Example
*********************

Let us build one ResNet-18 workload for GPU as an example first.

.. code:: python

   from tvm import relay
   from tvm.relay import testing
   from tvm.contrib import util
   import tvm

   # Resnet18 workload
   resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)

   # build
   with relay.build_config(opt_level=3):
       _, resnet18_lib, _ = relay.build_module.build(resnet18_mod, "cuda", params=resnet18_params)

   # create one tempory directory
   temp = util.tempdir()

   # path lib
   file_name = "deploy.so"
   path_lib = temp.relpath(file_name)

   # export library
   resnet18_lib.export_library(path_lib)

   # load it back
   loaded_lib = tvm.runtime.load_module(path_lib)
   assert loaded_lib.type_key == "library"
   assert loaded_lib.imported_modules[0].type_key == "cuda"

*************
Serialization
*************

The entrance API is ``export_library`` of ``tvm.module.Module``.
Inside this function, we will do the following steps:

1. Collect all DSO modules (LLVM modules and C modules)

2. Once we have DSO modules, we will call ``save`` function to save them into files.

3. Next, we will check whether we have imported modules, such as CUDA,
   OpenCL or anything else. We don't restrict the module type here.
   Once we have imported modules, we will create one file named ``devc.o`` / ``dev.cc``
   (so that we could embed the binary blob data of import modules into one dynamic shared library),
   then call function ``_PackImportsToLLVM`` or ``_PackImportsToC`` to do module serialization.

4. Finally, we call ``fcompile`` which invokes ``_cc.create_shared`` to get
   dynamic shared library.

.. note::
    1. For C source modules, we will compile them and link them together with the DSO module.

    2. Use ``_PackImportsToLLVM`` or ``_PackImportsToC`` depends on whether we enable LLVM in TVM.
       They achieve the same goal in fact.

***************************************************
Under the Hood of Serialization and Format Standard
***************************************************

As said before, we will do the serialization work in the ``_PackImportsToLLVM`` or ``_PackImportsToC``.
They both call ``SerializeModule`` to serialize the runtime module. In ``SerializeModule``
function, we firstly construct one helper class ``ModuleSerializer``. It will take ``module`` to do some
initialization work, like marking module index. Then we could use its ``SerializeModule`` to serialize module.

For better understanding, let us dig the implementation of this class a little deeper.

The following code is used to construct ``ModuleSerializer``:

.. code:: c++

   explicit ModuleSerializer(runtime::Module mod) : mod_(mod) {
     Init();
   }
   private:
   void Init() {
     CreateModuleIndex();
     CreateImportTree();
   }

In ``CreateModuleIndex()``, We will inspect module import relationship
using DFS and create index for them. Note the root module is fixed at
location 0. In our example, we have module relationship like this:

.. code:: c++

  llvm_mod:imported_modules
    - cuda_mod

So LLVM module will have index 0, CUDA module will have index 1.

After constructing module index, we will try to construct import tree (``CreateImportTree()``),
which will be used to restore module import relationship when we load
the exported library back. In our design, we use CSR format to store
import tree, each row is parent index, the child indices correspond to its children
index. In code, we use ``import_tree_row_ptr_`` and
``import_tree_child_indices_`` to represent them.

After initialization, we could serialize module using ``SerializeModule`` function.
In its function logic, we will assume the serialization format like this:

.. code:: c++

   binary_blob_size
   binary_blob_type_key
   binary_blob_logic
   binary_blob_type_key
   binary_blob_logic
   ...
   _import_tree
   _import_tree_logic

``binary_blob_size`` is the number of blobs we will have in this
serialization step. There will be three blobs in our example which
are created for LLVM module, CUDA module, and ``_import_tree``, respectively.

``binary_blob_type_key`` is the blob type key of module. For LLVM / C module, whose
blob type key is ``_lib``. For CUDA module, it is ``cuda``, which could be got by ``module->type_key()``.

``binary_blob_logic`` is the logic handling of blob. For most of blob (like CUDA, OpenCL), we will call
``SaveToBinary`` function to serialize blob into binary. However, like LLVM / C module, we will only write
``_lib`` to indicate this is a DSO module.

.. note::
   Whether or not it is required to implement the SaveToBinary virtual function depends on
   how the module is used. For example, If the module has information we need when we load
   the dynamic shared library back, we should do. Like CUDA module, we need its binary data
   passing to GPU driver when we load the dynamic shared library, so we should implement
   ``SaveToBinary`` to serialize its binary data. But for host module (like DSO), we don't
   need other information when we load the dynamic shared library, so we don't need to implement
   ``SaveToBinary``. However, if in the future, we want to record some meta information of DSO module,
   we could implement ``SaveToBinary`` for DSO module too.

Finally, we will write one key ``_import_tree`` unless our module only
has one DSO module and it is in the root. It is used to reconstruct the
module import relationship when we load the exported library back as said
before. The ``import_tree_logic`` is just to write ``import_tree_row_ptr_``
and ``import_tree_child_indices_`` into stream.

After this step, we will pack it into a symbol
``runtime::symbol::tvm_dev_mblob`` that can be recovered in the dynamic
libary.

Now, we complete the serialization part. As you have seen, we could
support arbitrary modules to import ideally.

****************
Deserialization
****************

The entrance API is ``tvm.runtime.load``. This function
is to call ``_LoadFromFile`` in fact. If we dig it a little deeper, this is
``Module::LoadFromFile``. In our example, the file is ``deploy.so``,
according to the function logic, we will call ``module.loadfile_so`` in
``dso_library.cc``. The key is here:

.. code:: c++

   // Load the imported modules
   const char* dev_mblob = reinterpret_cast<const char*>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));
   Module root_mod;
   if (dev_mblob != nullptr) {
   root_mod = ProcessModuleBlob(dev_mblob, lib);
   } else {
   // Only have one single DSO Module
   root_mod = Module(n);
   }

As said before, we will pack the blob into the symbol
``runtime::symbol::tvm_dev_mblob``. During deserialization part, we will
inspect it. If we have ``runtime::symbol::tvm_dev_mblob``, we will call ``ProcessModuleBlob``,
whose logic like this:

.. code:: c++

   READ(blob_size)
   READ(blob_type_key)
   for (size_t i = 0; i < blob_size; i++) {
       if (blob_type_key == "_lib") {
         // construct dso module using lib
       } else if (blob_type_key == "_import_tree") {
         // READ(_import_tree_row_ptr)
         // READ(_import_tree_child_indices)
       } else {
         // call module.loadbinary_blob_type_key, such as module.loadbinary_cuda
         // to restore.
       }
   }
   // Using _import_tree_row_ptr and _import_tree_child_indices to
   // restore module import relationship. The first module is the
   // root module according to our invariance as said before.
   return root_module;

After this, we will set the ``ctx_address`` to be the ``root_module`` so
that allow lookup of symbol from root (so all symbols are visible).

Finally, we complete the deserialization part.
