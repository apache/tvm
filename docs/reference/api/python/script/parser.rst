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

tvm.script.parser
-----------------

tvm.script.parser
*****************
.. automodule:: tvm.script.parser
   :members:
   :imported-members:

tvm.script.parser.protocol_registry
***********************************
Language variants register source syntax policies independently of their
construction hooks. Namespace objects and lazy initialization callbacks are
registered through :func:`tvm.script.parser.register_namespace` and
:func:`tvm.script.parser.register_namespace_initializer`.

.. automodule:: tvm.script.parser.protocol_registry
   :members: constexpr, args_policy, register_scalar_annotation, mutable_cell_decl, result_span, module_decorator, declaration_kind

The language variant aliases below share the public construction namespaces documented
in :doc:`script`. Parser entry points above use the canonical frontend.

tvm.script.parser.ir
********************
.. automodule:: tvm.script.parser.ir

tvm.script.parser.relax
***********************
.. automodule:: tvm.script.parser.relax

tvm.script.parser.tirx
**********************
.. automodule:: tvm.script.parser.tirx
