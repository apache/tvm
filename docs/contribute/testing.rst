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

Testing TVM
===========

This page describes how to write and run Python tests for TVM,
including the target parametrization utilities used by CI.

Python Target Parametrization
-----------------------------

Summary
~~~~~~~

For any supported runtime, TVM should produce numerically
correct results.  Therefore, when writing unit tests that validate
the numeric output, these unit tests should be run on all supported
runtimes.  Since this is a very common use case, TVM has helper
functions to parametrize unit tests such that they will run on all
targets that are enabled and have a compatible device.

A single Python function in the test suite can expand to several
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
~~~~~~~~~~~~~~~~~~~~~~~

.. _pytest-marks: https://docs.pytest.org/en/stable/how-to/mark.html

The recommended way to run a test on multiple targets is to parametrize
over ``target`` with ``@pytest.mark.parametrize``.  Tag each GPU target
with ``pytest.mark.gpu`` so the CI routes it to a GPU node, skip a target
that cannot run on the current machine with
:py:func:`tvm.testing.device_enabled`, and obtain its device with
``tvm.device_from_target(target)``.  The function is run once per target, the
success/failure of each is reported separately, and a target whose device
is disabled in ``config.cmake`` or absent from the machine is reported as
skipped.

.. code-block:: python

    @pytest.mark.parametrize(
        "target",
        ["llvm", pytest.param("cuda", marks=pytest.mark.gpu)],
    )
    def test_function(target):
        if not tvm.testing.device_enabled(target):
            pytest.skip(f"{target} not enabled")
        dev = tvm.device_from_target(target)
        # Test code goes here

For a test that only applies to a single target, omit the parametrization
and gate the test with ``@pytest.mark.skipif`` (plus ``@pytest.mark.gpu``
for a GPU target):

.. code-block:: python

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not tvm.testing.device_enabled("cuda"), reason="cuda not enabled"
    )
    def test_function():
        target = "cuda"
        dev = tvm.cuda()
        # Test code goes here

To exclude a target, leave it out of the parametrize list.  To mark a
target as expected to fail, wrap it with
``pytest.param("target", marks=pytest.mark.xfail(reason=...))``.

Additional parameters can be combined with the target parametrization by
stacking ``@pytest.mark.parametrize`` decorators, or by listing tuples of
arguments.  Tag the GPU rows with ``pytest.mark.gpu`` and skip in the body
as above:

.. code-block:: python

   @pytest.mark.parametrize("target,impl", [
        ("llvm", cpu_implementation),
        pytest.param("cuda", gpu_implementation_small_batch, marks=pytest.mark.gpu),
        pytest.param("cuda", gpu_implementation_large_batch, marks=pytest.mark.gpu),
    ])
    def test_function(target, impl):
        if not tvm.testing.device_enabled(target):
            pytest.skip(f"{target} not enabled")
        dev = tvm.device_from_target(target)
        # Test code goes here


Tests gate on hardware and carry metadata using
`pytest marks <pytest-marks>`_.  The most frequently applied
marks are as follows.

- ``@pytest.mark.gpu`` - Tags a function as using GPU
  capabilities. This has no effect on its own, but can be paired with
  the command-line arguments ``-m gpu`` or ``-m 'not gpu'`` to restrict
  which tests pytest will execute.  Apply it to any test that needs a
  GPU so that the CI runs it only on GPU nodes.

- ``@pytest.mark.skipif(not tvm.testing.env.has_X(), reason=...)`` -
  Skips a test when a required runtime or hardware feature is not
  available.  The :py:mod:`tvm.testing.env` module exposes one memoized
  probe per capability (e.g. ``has_cuda()``, ``has_rocm()``,
  ``has_vulkan()``, ``has_gpu()``, ``has_llvm()``), each of which
  returns ``False`` when the runtime is disabled in ``config.cmake`` or
  no compatible device is present.  Pair it with ``@pytest.mark.gpu``
  for tests that use the GPU::

      @pytest.mark.gpu
      @pytest.mark.skipif(not tvm.testing.env.has_cuda(), reason="need cuda")
      def test_cuda_vectorize_add():
          # Test code goes here

- ``pytest.importorskip("package_name")`` - Skips a test (or the whole
  module, when called at import time) if an optional Python package is
  not installed.  Use this instead of a ``skipif`` for package
  dependencies.

Tests that execute on a local GPU must put the complete live-device
lifetime in a small callback passed to
:py:func:`tvm.testing.run_with_gpu_lock`.  Target construction and
compilation remain outside so that pytest-xdist workers can compile in
parallel.  Device creation, allocation, execution, synchronization,
host conversion, result checks, and child-process teardown remain inside
the callback so no device-backed object outlives the lock.

.. code-block:: python

    @pytest.mark.gpu
    @pytest.mark.skipif(not tvm.testing.env.has_cuda(), reason="need cuda")
    def test_cuda_add_one():
        target = tvm.target.Target("cuda -arch=sm_90")
        executable = tvm.compile(make_add_one_module(), target)
        host_input = np.arange(16, dtype="float32")

        def run_and_check():
            dev = tvm.cuda(0)
            device_input = tvm.runtime.tensor(host_input, dev)
            device_output = tvm.runtime.empty(host_input.shape, "float32", dev)
            executable(device_input, device_output)
            dev.sync()
            tvm.testing.assert_allclose(device_output.numpy(), host_input + 1)

        tvm.testing.run_with_gpu_lock(run_and_check)

The wrapper uses the existing :py:class:`tvm_ffi.utils.FileLock` with a
persistent machine-local path.  A process exit releases the kernel lock;
the remaining file is not stale ownership.  Test startup must never
delete or rotate it, because another process could then lock a different
inode.  Set ``TVM_TEST_LOCK_DIR`` only when all cooperating processes need
an explicitly configured shared machine-local directory.  The default
temporary path coordinates processes running as the same user.  Multi-user
runners sharing a GPU must use one administrator-provisioned directory and
persistent lock file that every contender can write, or enforce exclusivity
through the runner.  A per-user lock path cannot protect a GPU shared across
users because each user would lock a different file.

Running Locally
~~~~~~~~~~~~~~~

To run the Python unit tests locally, use the command ``pytest`` in
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
      avoid running tests located in ``tests/python/contrib`` on a
      system without a specific backend installed.

    - The ``-m`` argument only runs unit tests that are tagged with a
      specific pytest marker. The most frequent usage is to use
      ``-m gpu`` to run only tests that are marked with
      ``@pytest.mark.gpu`` and use a GPU to run. It can also be used
      to run only tests that do not use a GPU, by passing ``not gpu``
      as the marker expression to ``-m``.

      Note: This filtering takes place after the selection of targets
      based on the ``TVM_TEST_TARGETS`` environment variable.  Even if
      ``-m gpu`` is specified, if ``TVM_TEST_TARGETS`` does not
      contain GPU targets, no GPU tests will be run.

Running in a Local Docker Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _tlcpack: https://hub.docker.com/u/tlcpack

The ``docker/bash.sh`` script can be used to run unit tests inside the
same docker image as is used by the CI.  The first argument should
specify which docker image to run (e.g. ``docker/bash.sh ci_gpu``).
Allowed image names are defined in ``ci/jenkins/docker-images.ini`` in the TVM source directory,
and map to images at `tlcpack`_.

If no additional arguments are given, the docker image will be loaded
with an interactive bash session. Additional arguments run a command inside
the container. First build the platform wheels as described in
:ref:`pr-testing`, then source the installation helper before running tests:

.. code-block:: bash

    docker/bash.sh ci_gpu bash -euo pipefail -c '
        source tests/scripts/test_task_install_dist_wheels.sh
        tests/scripts/task_python_unittest_gpuonly.sh
    '

The helper installs the wheels from ``dist/`` into ``.local/python`` and sets
``PYTHONPATH`` for subsequent commands in the same shell. In an interactive
container, source the helper before invoking pytest or a test script.

The checkout is shared with the container. Native builds use ``build/`` and
wheel artifacts use ``dist/``; clean these directories and rebuild when
switching platform configurations.

Running in CI
~~~~~~~~~~~~~

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

  The Jenkinsfile sources ``tests/scripts/task_install_build_dependencies.sh``
  once to clean ``dist/`` and install build prerequisites. It then invokes
  ``tests/scripts/task_build_<platform>.sh``, which sets the full platform CMake
  options and builds FFI, TIRx, and TVM wheels in that order. All three wheel
  builds use the root CMake source and the same ``build/`` directory, with
  package-specific install components. Enabled C++ tests run from that same
  native build.

  The three wheels in ``dist/`` are transferred to the test nodes.

- Which tests run

  The ``Unit Test`` stage of the Jenkinsfile determines how ``pytest``
  is called. Each consumer downloads the wheels from the ``BUILD`` stage
  and sources ``tests/scripts/test_task_install_dist_wheels.sh`` before
  running a test script (e.g. ``tests/scripts/task_python_unittest.sh``).
  These scripts set
  the files/folders and command-line options that are passed to
  ``pytest``.

  Several of these scripts include the ``-m gpu`` option, which
  restricts the tests to only run tests that include the
  ``@pytest.mark.gpu`` mark.
