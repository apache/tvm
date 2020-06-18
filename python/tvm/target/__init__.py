# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Target description and codgen module.

TVM's target string is in fomat ``<target_name> [-option=value]...``.

Note
----
The list of options include:

- **-device=<device name>**

   The device name.

- **-mtriple=<target triple>** or **-target**

   Specify the target triple, which is useful for cross
   compilation.

- **-mcpu=<cpuname>**

   Specify a specific chip in the current architecture to
   generate code for. By default this is infered from the
   target triple and autodetected to the current architecture.

- **-mattr=a1,+a2,-a3,...**

   Override or control specific attributes of the target,
   such as whether SIMD operations are enabled or not. The
   default set of attributes is set by the current CPU.

- **-system-lib**

   Build TVM system library module. System lib is a global module that contains
   self registered functions in program startup. User can get the module using
   :any:`tvm.runtime.system_lib`.
   It is useful in environments where dynamic loading api like dlopen is banned.
   The system lib will be available as long as the result code is linked by the program.

We can use :py:func:`tvm.target.create` to create a tvm.target.Target from the target string.
We can also use other specific function in this module to create specific targets.
"""
from .target import Target, create
from .target import cuda, rocm, mali, intel_graphics, opengl, arm_cpu, rasp, vta, bifrost, hexagon
from .generic_func import GenericFunc
from .generic_func import generic_func, get_native_generic_func, override_native_generic_func
from . import datatype
from . import codegen
from .intrin import register_intrin_rule
from .build_config import BuildConfig, build_config
