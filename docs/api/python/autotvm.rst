Auto-tuning API
---------------
.. automodule:: tvm.autotvm

tvm.autotvm.measure
~~~~~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.measure.measure

.. class:: tvm.autotvm.MeasureInput(target, task, config)

      stores all the necessary inputs for a measurement.

      :param Target target: The target device
      :param Task task: Task function
      :param ConfigEntity config: Specific configuration

.. class:: tvm.autotvm.MeasureResult(costs, error_no, all_cost, timestamp)

      stores all the results of a measurement

     :param tuple costs:
        If no error occurs for this measure, it is an array of measured running times.
        If some error occurs during the measure, it is an array of the exception objections.
     :param int error_no:
        denote error type, defined by MeasureErrorNo
     :param float all_cost:
        all cost of this measure, including rpc, compilation, test runs
     :param float timestamp:
        The absolute time stamp when we finish measurement.

.. autofunction:: tvm.autotvm.measure_option

.. autofunction:: tvm.autotvm.measure.create_measure_batch


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


tvm.autotvm.template
~~~~~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.template
    :members:

.. automodule:: tvm.autotvm.template.space
    :members:

tvm.autotvm.task
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.task
    :members:

.. automodule:: tvm.autotvm.task.task
    :members:

tvm.autotvm.record
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: tvm.autotvm.record
    :members:
