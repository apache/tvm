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

tvm.autotvm
-----------
.. automodule:: tvm.autotvm
.. autofunction:: tvm.autotvm.apply_history_best

tvm.autotvm.measure
~~~~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.measure.measure

.. autoclass:: tvm.autotvm.measure.MeasureInput
    :members:

.. autoclass:: tvm.autotvm.measure.MeasureResult
    :members:

.. autofunction:: tvm.autotvm.measure.measure_option

.. autofunction:: tvm.autotvm.measure.create_measure_batch

.. autoclass:: tvm.autotvm.measure.measure_methods.LocalBuilder

.. autoclass:: tvm.autotvm.measure.measure_methods.RPCRunner

.. autoclass:: tvm.autotvm.measure.measure_methods.LocalRunner

tvm.autotvm.tuner
~~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.tuner
    :members:

.. autoclass:: tvm.autotvm.tuner.Tuner
    :members:

.. autoclass:: tvm.autotvm.tuner.RandomTuner
    :members:
    :inherited-members:

.. autoclass:: tvm.autotvm.tuner.GridSearchTuner
    :members:
    :inherited-members:

.. autoclass:: tvm.autotvm.tuner.GATuner
    :members:
    :inherited-members:

.. autoclass:: tvm.autotvm.tuner.XGBTuner
    :members:
    :inherited-members:

.. automodule:: tvm.autotvm.tuner.callback
    :members:

tvm.autotvm.task
~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.task
    :members:

.. automodule:: tvm.autotvm.task.task
    :members:

.. automodule:: tvm.autotvm.task.space
    :members:

.. automodule:: tvm.autotvm.task.dispatcher
    :members:

.. automodule:: tvm.autotvm.task.topi_integration
    :members:

tvm.autotvm.record
~~~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.record
    :members:
