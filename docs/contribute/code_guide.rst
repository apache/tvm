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

.. _code_guide:

Code Guide and Tips
===================

.. contents::
  :depth: 2
  :local:

This is a document used to record tips in TVM codebase for reviewers and contributors.
Most of them are summarized through lessons during the contributing and process.


C++ Code Styles
---------------
- Use the Google C/C++ style.
- The public facing functions are documented in doxygen format.
- Favor concrete type declaration over ``auto`` as long as it is short.
- Favor passing by const reference (e.g. ``const Expr&``) over passing by value.
  Except when the function consumes the value by copy constructor or move,
  pass by value is better than pass by const reference in such cases.
- Favor ``const`` member function when possible.

We use ``clang-format`` to enforce the code style. Because different version
of clang-format might change by its version, it is recommended to use the same
version of the clang-format as the main one.
You can also use the following command via docker.

.. code:: bash

    # Run a specific file through clang-format
    docker/bash.sh ci_lint clang-format-10 [path-to-file]

    # Run all linters, including clang-format
    python tests/scripts/ci.py lint


clang-format is also not perfect, when necessary, you can use disble clang-format on certain code regions.

.. code :: c

   // clang-format off
   void Test() {
      // clang-format will be disabled in this region.
   }
   // clang-format on


Because clang-format may not recognize macros, it is recommended to use macro like normal function styles.


.. code :: c

   #define MACRO_IMPL { custom impl; }
   #define MACRO_FUNC(x)

   // not preferred, because clang-format might recognize it as types.
   virtual void Func1() MACRO_IMPL

   // preferred
   virtual void Func2() MACRO_IMPL;

   void Func3() {
     // preferred
     MACRO_FUNC(xyz);
   }


Python Code Styles
------------------
- The functions and classes are documented in `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_ format.
- Check your code style using ``python tests/scripts/ci.py lint``
- Stick to language features in ``python 3.7``

- For functions with early returns, prefer ``if``/``elif``/``else``
  chains for functions with parallel and short bodies to the
  conditions, such as functions that apply a simple mapping to the
  arguments.  For more procedural functions, especially where the
  final ``else`` block would be much longer than the ``if`` and
  ``elif`` blocks, prefer having the final ``else`` case unindented.

  The pylint check ``no-else-return`` is disabled to allow for this
  distinction.  See further discussion `here
  <https://github.com/apache/tvm/pull/11327>`.

  .. code:: python

    # All cases have bodies with similar flow control.  While this could
    # be expressed as a sequence of if conditions, a reader would need to
    # inspect the body of each condition to know that only one conditional
    # body may be reached.
    def sign(x):
        if x > 0:
            return "+"
        elif x < 0:
            return "-"
        else:
            return ""

    # The initial special case is an early return for a special case,
    # followed by a more general method.  Using an else block for the
    # condition would add unnecessary indentation for the remainder of the
    # function.
    def num_unique_subsets(values):
        if len(values)==0:
            return 1

        # Longer, more general solution here
        ...

Writing Python Tests
--------------------
We use `pytest <https://docs.pytest.org/en/stable/>`_ for all python testing. ``tests/python`` contains all the tests.

If you want your test to run over a variety of targets, use the :py:func:`tvm.testing.parametrize_targets` decorator. For example:

.. code:: python

  @tvm.testing.parametrize_targets
  def test_mytest(target, dev):
    ...

will run ``test_mytest`` with ``target="llvm"``, ``target="cuda"``, and few others. This also ensures that your test is run on the correct hardware by the CI. If you only want to test against a couple targets use ``@tvm.testing.parametrize_targets("target_1", "target_2")``. If you want to test on a single target, use the associated decorator from :py:func:`tvm.testing`. For example, CUDA tests use the ``@tvm.testing.requires_cuda`` decorator.


Network Resources
-----------------

In CI, downloading files from the Internet is a big source of flaky test failures (e.g. remote
server can go down or be slow), so try to avoid using the network at all during tests. In some cases
this isn't a reasonable proposition (e.g. the docs tutorials which need to download models).

In these cases you can re-host files in S3 for fast access in CI. A committer can upload a file,
specified by a name, hash, and path in S3, using the ``workflow_dispatch`` event on `the
upload_ci_resource.yml GitHub Actions workflow
<https://github.com/apache/tvm/actions/workflows/upload_ci_resource.yml>`_.  The sha256 must match
the file or it will not be uploaded. The upload path is user-defined so it can be any path (no
trailing or leading slashes allowed) but be careful not to collide with existing resources on
accident. Once uploaded you should send a PR to update the ``URL_MAP`` in
`request_hook.py <https://github.com/apache/tvm/blob/main/tests/scripts/request_hook/request_hook.py>`_
with the new URL.


Handle Integer Constant Expression
----------------------------------
We often need to handle constant integer expressions in TVM. Before we do so, the first question we want to ask is that is it really necessary to get a constant integer. If symbolic expression also works and let the logic flow, we should use symbolic expression as much as possible. So the generated code works for shapes that are not known ahead of time.

Note that in some cases we cannot know certain information, e.g. sign of symbolic variable, it is ok to make assumptions in certain cases. While adding precise support if the variable is constant.

If we do have to get constant integer expression, we should get the constant value using type ``int64_t`` instead of ``int``, to avoid potential integer overflow. We can always reconstruct an integer with the corresponding expression type via ``make_const``. The following code gives an example.

.. code:: c++

   Expr CalculateExpr(Expr value) {
     int64_t int_value = GetConstInt<int64_t>(value);
     int_value = CalculateExprInInt64(int_value);
     return make_const(value.type(), int_value);
   }
