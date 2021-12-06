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

.. _model_library_format:

Model Library Format
====================

About Model Library Format
--------------------------

TVM traditionally exports generated libraries as Dynamic Shared Objects (e.g. DLLs (Windows) or .so
(linux)). Inferences can be performed using those libraries by loading them into an executable using
``libtvm_runtime.so``. This process is very dependent on services provided by traditional OS.

For deployment to unconventional platforms (e.g. those lacking traditional OS), TVM provides another
output format, Model Library Format. Initially, the microTVM project is the primary use case for this
format. Should it become useful in other use cases (and in particular, should it become possible to
export BYOC artifacts in Model Library Format), it could be used as a general-purpose TVM export
format.  Model Library Format is a tarball containing a file for each piece of the TVM compiler
output.

What can be Exported?
---------------------

At the time of writing, export is limited to full models built with ``tvm.relay.build``.

Directory Layout
----------------

Model Library Format is contained within a tarball. All paths are relative to the root of the
tarball:

- ``/`` - Root of the tarball

  - ``codegen`` - Root directory for all generated device code

    - (see `codegen`_ section)

  - ``executor-config/`` - Configuration for the executor which drives model inference

    - ``graph/`` - Root directory containing configuration for the GraphExecutor

      - ``graph.json`` - GraphExecutor JSON configuration

  -  ``metadata.json`` - Machine-parseable metadata for this model

  - ``parameters/`` - Root directory where simplified parameters are placed

    - ``<model_name>.params`` - Parameters for the model tvm.relay._save_params format

  - ``src/`` - Root directory for all source code consumed by TVM

    - ``relay.txt`` - Relay source code for the generated model

Description of Sub-directories
------------------------------

.. _subdir_codegen:

``codegen``
^^^^^^^^^^^

All TVM-generated code is placed in this directory. At the time of writing, there is 1 file per
Module in the generated Module tree, though this restriction may change in the future. Files in
this directory should have filenames of the form ``<target>/(lib|src)/<unique_name>.<format>``.

These components are described below:

 * ``<target>`` - Identifies the TVM target on which the code should run. Currently, only ``host``
   is supported.
 * ``<unique_name>`` - A unique slug identifying this file. Currently ``lib<n>``, with ``<n>>`` an
   auto-incrementing integer.
 * ``<format>`` - Suffix identifying the filename format. Currently ``c`` or ``o``.

An example directory tree for a CPU-only model is shown below:

- ``codegen/`` - Codegen directory

  - ``host/`` - Generated code for ``target_host``

    -  ``lib/`` - Generated binary object files

      - ``lib0.o`` - LLVM module (if ``llvm`` target is used)
      - ``lib1.o`` - LLVM CRT Metadata Module (if ``llvm`` target is used)

    - ``src/`` - Generated C source

      - ``lib0.c`` - C module (if ``c`` target is used)
      - ``lib1.c`` - C CRT Metadata module (if ``c`` target is used)

``executor-config``
^^^^^^^^^^^^^^^^^^^

Contains machine-parsable configuration for executors which can drive model inference. Currently,
only the GraphExecutor produces configuration for this directory, in ``graph/graph.json``. This
file should be read in and the resulting string supplied to the ``GraphExecutor()`` constructor for
parsing.

``parameters``
^^^^^^^^^^^^^^

Contains machine-parseable parameters. A variety of formats may be provided, but at present, only
the format produced by ``tvm.relay._save_params`` is supplied. When building with
``tvm.relay.build``,  the ``name`` parameter is considered to be the model name. A single file is
created in this directory ``<model_name>.json``.

``src``
^^^^^^^

Contains source code parsed by TVM. Currently, just the Relay source code is created in
``src/relay.txt``.

Metadata
--------

Machine-parseable metadata is placed in a file ``metadata.json`` at the root of the tarball.
Metadata is a dictionary with these keys:

- ``export_datetime``: Timestamp when this Model Library Format was generated, in
  `strftime <https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>`_
  format ``"%Y-%M-%d %H:%M:%SZ",``.
- ``memory``: A summary of the memory usage of each generated function. Documented in
  `Memory Usage Summary`_.
- ``model_name``: The name of this model (e.g. the ``name`` parameter supplied to
  ``tvm.relay.build``).
- ``executors``: A list of executors supported by this model. Currently, this list is always
  ``["graph"]``.
- ``target``: A dictionary mapping ``device_type`` (the underlying integer, as a string) to the
  sub-target which describes that relay backend used for that ``device_type``.
- ``version``: A numeric version number that identifies the format used in this Model Library
  Format. This number is incremented when the metadata structure or on-disk structure changes.
  This document reflects version ``5``.

Memory Usage Summary
^^^^^^^^^^^^^^^^^^^^

A dictionary with these sub-keys:

 - ``"main"``: ``list[MainFunctionWorkspaceUsage]``. A list summarizing memory usage for each
   workspace used by the main function and all sub-functions invoked.
 - ``"operator_functions"``: ``map[string, list[FunctionWorkspaceUsage]]``. Maps operator function
   name to a list summarizing memory usage for each workpace used by the function.

A ``MainFunctionWorkspaceUsage`` is a dict with these keys:

- ``"device"``: ``int``. The ``device_type`` associated with this workspace.
- ``"workspace_size_bytes"``: ``int``. Number of bytes needed in this workspace by this function
  and all sub-functions invoked.
- ``"constants_size_bytes"``: ``int``. Size of the constants used by the main function.
- ``"io_size_bytes"``: ``int``. Sum of the sizes of the buffers used from this workspace by this
  function and sub-functions.

A ``FunctionWorkspaceUsage`` is a dict with these keys:

- ``"device"``: ``int``. The ``device_type`` associated with this workspace.
- ``"workspace_size_bytes"``: ``int``. Number of bytes needed in this workspace by this function.
