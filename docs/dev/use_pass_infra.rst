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

.. _use-pass-infra:

How to customize optimization pass pipeline
===========================================
In the :ref:`pass-infra` tutorial, we have introduced the pass infra that
provides a systematic and consistent way for TVM to manage the optimization
passes at different layers (such as Relay and tir). This doc illustrates how we
can use some of the more advanced features in the pass infra, e.g. customizing your
own optimization pass pipeline and debugging. Here, we will mainly focus on
using sequential passes. For more details about other type of passes and how
they interact with ``Sequential``, please refer to :ref:`pass-infra`.

The following example not only illustrates how users can directly create a sequential
pass using Python APIs (this could be applied to module- and function-level
passes as well), but also explains how we can build an optimization pipeline
using ``Sequential`` associated with other types of passes.

.. code:: python

    # Create a simple Relay program.
    shape = (1, 2, 3)
    c_data = np.array(shape).astype("float32")
    tp = relay.TensorType(shape, "float32")
    c = relay.const(c_data)
    x = relay.var("x", tp)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(x, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    func = relay.Function([x], z2)

    # Customize the optimization pipeline.
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.AlterOpLayout()
    ])

    # Create a module to perform optimizations.
    mod = relay.Module({"main": func})

    # Users can disable any passes that they don't want to execute by providing
    # a list, e.g. disabled_pass=["EliminateCommonSubexpr"].
    with relay.build_config(opt_level=3):
        with tvm.target.create("llvm"):
            # Perform the optimizations.
            mod = seq(mod)

Debugging
=========

The pass infra provides a special pass (``PrintIR``) to dump the IR of the
whole module after applying a certain pass. A slightly modified version of the
sequential pass example could be like the following to enable IR dumping for
``FoldConstant`` optimization.

.. code:: python

    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        transform.PrintIR(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.AlterOpLayout()
    ])

By inserting the ``PrintIR`` pass after ``FoldConstant``, the pass infra will
dump out the module IR when ``FoldConstant`` is done. Users can plug in this
pass after any pass they want to debug for viewing the optimization effect.

There is a more flexible debugging mechanism also exposed by the build configuration
object. One can pass a tracing function which can be used to execute arbitrary code
before and/or after each pass. A tracing function will receive a ``IRModule``, ``PassInfo``,
and a boolean indicating whether you are executing before, or after a pass.
An example is below.

.. code:: python

    def print_ir(mod, info, is_before):
        """Print the name of the pass, the IR, only before passes execute."""
        if is_before:
            print(f"Running pass: {}", info)
            print(mod)

    with relay.build_config(opt_level=3, trace=print_ir):
            with tvm.target.create("llvm"):
                # Perform the optimizations.
                mod = seq(mod)


For more pass infra related examples in Python and C++, please refer to
`tests/python/relay/test_pass_manager.py`_ and
`tests/cpp/relay_transform_sequential.cc`_, respectively.

.. _tests/python/relay/test_pass_manager.py: https://github.com/apache/incubator-tvm/blob/master/tests/python/relay/test_pass_manager.py

.. _tests/cpp/relay_transform_sequential.cc: https://github.com/apache/incubator-tvm/blob/master/tests/cpp/relay_transform_sequential.cc
