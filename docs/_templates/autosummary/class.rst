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

{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% for member in ["expr_ty", "equal", "astype", "script", "show"] if member in inherited_members %}
.. automethod:: {{ fullname }}.{{ member }}
{% endfor %}


{% if fullname == "tvm_tirx.ir.For" %}
ForKind
-------

.. autoclass:: tvm_tirx.ir.ForKind
{% endif %}

{% if fullname == "tvm_tirx.ir.Call" %}
CallEffectKind
--------------

.. autoclass:: tvm_tirx.ir.CallEffectKind
{% endif %}

{% if fullname == "tvm_tirx.ir.BaseFunc" %}
CallingConv
-----------

.. autoclass:: tvm_tirx.ir.CallingConv
{% endif %}

{% if fullname == "tvm_tirx.ir.Location" %}
Unknown location
----------------

.. autodata:: tvm_tirx.ir.UNKNOWN_LOC
   :no-value:

   Shared :class:`~tvm_tirx.ir.UnknownLoc` sentinel when no source location is
   available. Runtime-only imports expose ``None`` instead.
{% endif %}
