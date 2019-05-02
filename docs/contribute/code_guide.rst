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

Python Code Styles
------------------
- The functions and classes are documented in `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_ format.
- Check your code style using ``make pylint``


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
