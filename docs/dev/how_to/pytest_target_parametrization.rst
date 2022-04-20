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

Python Target Parametrization
=============================

Summary
-------

For any supported runtime, TVM should produce numerically
correct results.  Therefore, when writing unit tests that validate
the numeric output, these unit tests should be run on all supported
runtimes.  Since this is a very common use case, TVM has helper
functions to parametrize unit tests such that they will run on all
targets that are enabled and have a compatible device.

A single python function in the test suite can expand to several
parameterized unit tests, each of which tests a single target device.
In order for a test to be run, all of the following must be true.

- The test exists in a file or directory that has been passed to
  `pytest`.

- The pytest marks applied to the function, either explicitly or
  through target parametrization, must be compatible with the
  expression passed to pytest's `-m` argument.

- For parametrized tests using the `target` fixture, the target must
  appear in the environment variable `TVM_TEST_TARGETS`.

- For parametrized tests using the `target` fixture, the build
  configuration in `config.cmake` must enable the corresponding
  runtime.

Unit-Test File Contents
-----------------------

.. _pytest-marks: https://docs.pytest.org/en/6.2.x/mark.html

The recommended method to run a test on multiple targets is by
parametrizing the test.  This can be done explicitly for a fixed list
of targets by decorating with
``@tvm.testing.parametrize_targets('target_1', 'target_2', ...)``, and
accepting ``target`` or ``dev`` as function arguments.  The function
will be run once for each target listed, and the success/failure of
each target is reported separately.  If a target cannot be run because
it is disabled in the `config.cmake`, or because no appropriate
hardware is present, then that target will be reported as skipped.

.. code-block:: python

    # Explicit listing of targets to use.
    @tvm.testing.parametrize_target('llvm', 'cuda')
    def test_function(target, dev):
        # Test code goes here

For tests that should run correctly on all targets, the decorator can
be omitted.  Any test that accepts a ``target`` or ``dev`` argument
will automatically be parametrized over all targets specified in
``TVM_TEST_TARGETS``.  The parametrization provides the same
pass/fail/skipped report for each target, while allowing the test
suite to be easily extended to cover additional targets.

.. code-block:: python

    # Implicitly parametrized to run on all targets
    # in environment variable TVM_TEST_TARGETS
    def test_function(target, dev):
        # Test code goes here

The ``@tvm.testing.parametrize_targets`` can also be used as a bare
decorator to explicitly draw attention to the parametrization, but has
no additional effect.

.. code-block:: python

    # Explicitly parametrized to run on all targets
    # in environment variable TVM_TEST_TARGETS
    @tvm.testing.parametrize_targets
    def test_function(target, dev):
        # Test code goes here


Specific targets can be excluded or marked as expected to fail using
the ``@tvm.testing.exclude_targets`` or
``@tvm.testing.known_failing_targets`` decorators.  For more
information on their intended use cases, please see their docstrings.

In some cases it may be necessary to parametrize across multiple
parameters.  For instance, there may be target-specific
implementations that should be tested, where some targets have more
than one implementation.  These can be done by explicitly
parametrizing over tuples of arguments, such as shown below.  In these
cases, only the explicitly listed targets will run, but they will
still have the appropriate ``@tvm.testing.requires_RUNTIME`` mark
applied to them.

.. code-block:: python

   @pytest.mark.parametrize('target,impl', [
        ('llvm', cpu_implementation),
        ('cuda', gpu_implementation_small_batch),
        ('cuda', gpu_implementation_large_batch),
    ])
    def test_function(target, dev, impl):
        # Test code goes here


The parametrization functionality is implemented
on top of pytest marks.  Each test function can
be decorated with `pytest marks <pytest-marks>`_
to include metadata.  The most frequently applied
marks are as follows.

- ``@pytest.mark.gpu`` - Tags a function as using GPU
  capabilities. This has no effect on its own, but can be paired with
  command-line arguments ``-m gpu`` or ``-m 'not gpu'`` to restrict
  which tests pytest will execute.  This should not be called on its
  own, but is part of other marks used in unit-tests.

- ``@tvm.testing.uses_gpu`` - Applies ``@pytest.mark.gpu``.  This
  should be used to mark unit tests that may use the GPU, if one is
  present.  This decorator is only needed for tests that explicitly
  loop over ``tvm.testing.enabled_targets()``, but that is no longer
  the preferred style of writing unit tests (see below).  When using
  ``tvm.testing.parametrize_targets()``, this decorator is implicit
  for GPU targets, and does not need to be explicitly applied.

- ``@tvm.testing.requires_gpu`` - Applies ``@tvm.testing.uses_gpu``,
  and additionally marks that the test should be skipped
  (``@pytest.mark.skipif``) entirely if no GPU is present.

- ``@tvfm.testing.requires_RUNTIME`` - Several decorators
  (e.g. ``@tvm.testing.requires_cuda``), each of which skips a test if
  the specified runtime cannot be used. A runtime cannot be used if it
  is disabled in the ``config.cmake``, or if a compatible device is
  not present. For runtimes that use the GPU, this includes
  ``@tvm.testing.requires_gpu``.

When using parametrized targets, each test run is decorated with the
``@tvm.testing.requires_RUNTIME`` that corresponds to the target
being used.  As a result, if a target is disabled in ``config.cmake``
or does not have appropriate hardware to run, it will be explicitly
listed as skipped.

There also exists a ``tvm.testing.enabled_targets()`` that returns
all targets that are enabled and runnable on the current machine,
based on the environment variable ``TVM_TEST_TARGETS``, the build
configuration, and the physical hardware present.  Most current tests
explicitly loop over the targets returned from ``enabled_targets()``,
but it should not be used for new tests.  The pytest output for this
style silently skips runtimes that are disabled in ``config.cmake``,
or do not have a device on which they can run.  In addition, the test
halts on the first target to fail, which is ambiguous as to whether
the error occurs on a particular target, or on every target.

.. code-block:: python

    # Old style, do not use.
    def test_function():
        for target,dev in tvm.testing.enabled_targets():
            # Test code goes here



Running locally
---------------

To run the python unit-tests locally, use the command ``pytest`` in
the ``${TVM_HOME}`` directory.

- Environment variables
    - ``TVM_TEST_TARGETS`` should be a semicolon-separated list of
      targets to run. If unset, will default to the targets defined in
      ``tvm.testing.DEFAULT_TEST_TARGETS``.

      Note: If ``TVM_TEST_TARGETS`` does not contain any targets that
      are both enabled, and have an accessible device of that type,
      then the tests will fall back to running on the ``llvm`` target
      only.

    - ``TVM_LIBRARY_PATH`` should be a path to the ``libtvm.so``
      library. This can be used, for example, to run tests using a
      debug build. If unset, will search for ``libtvm.so`` relative to
      the TVM source directory.

- Command-line arguments

    - Passing a path to a folder or file will run only the unit tests
      in that folder or file. This can be useful, for example, to
      avoid running tests located in ``tests/python/frontend`` on a
      system without a specific frontend installed.

    - The ``-m`` argument only runs unit tests that are tagged with a
      specific pytest marker. The most frequent usage is to use ``m
      gpu`` to run only tests that are marked with
      ``@pytest.mark.gpu`` and use a GPU to run. It can also be used
      to run only tests that do not use a GPU, by passing ``m 'not
      gpu'``.

      Note: This filtering takes place after the selection of targets
      based on the ``TVM_TEST_TARGETS`` environment variable.  Even if
      ``-m gpu`` is specified, if ``TVM_TEST_TARGETS`` does not
      contain GPU targets, no GPU tests will be run.

Running in local docker container
---------------------------------

.. _tlcpack: https://hub.docker.com/u/tlcpack

The ``docker/bash.sh`` script can be used to run unit tests inside the
same docker image as is used by the CI.  The first argument should
specify which docker image to run (e.g. ``docker/bash.sh ci_gpu``).
Allowed image names are defined at the top of the Jenkinsfile located
in the TVM source directory, and map to images at `tlcpack`_.

If no additional arguments are given, the docker image will be loaded
with an interactive bash session.  If a script is passed as an
optional argument (e.g. ``docker/bash.sh ci_gpu tests/scripts/task_python_unittest.sh``), then that script will be
executed inside the docker image.

Note: The docker images contain all system dependencies, but do not
include the ``build/config.cmake`` configuration file for those
systems.  The TVM source directory is used as the home directory of
the docker image, and so this will default to using the same
config/build directories as the local config.  One solution is to
maintain separate ``build_local`` and ``build_docker`` directories,
and make a symlink from ``build`` to the appropriate folder when
entering/exiting docker.

Running in CI
-------------

Everything in the CI starts from the task definitions present in the
Jenkinsfile.  This includes defining which docker image gets used,
what the compile-time configuration is, and which tests are included
in which stages.

- Docker images

  Each task of the Jenkinsfile (e.g. 'BUILD: CPU') makes calls to
  ``docker/bash.sh``.  The argument following the call to
  docker/bash.sh defines the docker image in CI, just as it does
  locally.

- Compile-time configuration

  The docker image does not have the ``config.cmake`` file built into
  it, so this is the first step in each of the ``BUILD`` tasks.  This
  is done using the ``tests/scripts/task_config_build_*.sh`` scripts.
  Which script is used depends on the build being tested, and is
  specified in the Jenkinsfile.

  Each ``BUILD`` task concludes by packing a library for use in later
  tests.

- Which tests run

  The ``Unit Test`` and ``Integration Test`` stages of the Jenkinsfile
  determine how ``pytest`` is called.  Each task starts by unpacking a
  compiled library that was previous compiled in the ``BUILD`` stage,
  then runs a test script
  (e.g. ``tests/script/task_python_unittest.sh``).  These scripts set
  the files/folders and command-line options that are passed to
  ``pytest``.

  Several of these scripts include the ``-m gpu`` option, which
  restricts the tests to only run tests that include the
  ``@pytest.mark.gpu`` mark.
