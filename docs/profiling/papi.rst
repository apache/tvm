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


Getting Started With PAPI
=========================

The Performance Application Programming Interface (PAPI) is a library that
provides performance counters on a variety of platforms. Performance counters
provide accurate low-level information about processors behavior during a given
execution run. This information can contain simple metrics like total cycle
count, cache misses, and instructions executed as well as more high level
information like total FLOPS and warp occupancy. PAPI makes these metrics
available while profiling.

Installing PAPI
---------------

PAPI can either be installed using your package manager (``apt-get install
libpapi-dev`` on Ubuntu), or from source here:
https://bitbucket.org/icl/papi/src/master/.


Building TVM With PAPI
----------------------

To include PAPI in your build of TVM, set the following line in you ``config.cmake``:

.. code::

   set(USE_PAPI ON)

If PAPI is installed in a non-standard place, you can specify where it is like so:

.. code::

   set(USE_PAPI path/to/papi.pc)


Using PAPI While Profiling
--------------------------

If TVM has been built with PAPI (see above), then calling the
:py:meth:`tvm.runtime.GraphModule.profile` will automatically include results
from a default set of performance counters. To change which performance
counters are reported, set the ``TVM_PAPI_${DEVICE}_METRICS`` environment
variable (where ``${DEVICE}`` is the device you are running on; ``GPU`` if
using a gpu, ``CPU`` for the cpu) to a semicolon separated list of metrics. For
example, ``TVM_PAPI_CPU_METRICS=perf::INSTRUCTIONS;perf::BRANCH-INSTRUCTIONS``
would report the number of instructions executed and the number of branch
instructions executed. You can find a list of available metrics by running the
``papi_avail`` and ``papi_native_avail`` commands.
