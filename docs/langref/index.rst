Language Reference
==================
This document provides references to
embedded languages and IRs in the TVM stack.

Introduction to Relay
---------------------

Relay is a functional, differentiable programming language
designed to be an expressive intermediate representation for machine
learning systems. Relay supports algebraic data types, closures,
control flow, and recursion, allowing it to directly represent more
complex models than computation graph-based IRs can.
Relay also includes a form of dependent typing using *type relations*
in order to handle shape analysis for operators with complex
requirements on argument shapes.

Relay is extensible by design and makes it easy for machine learning
researchers and practitioners to develop new large-scale program
transformations and optimizations.

The below pages describe the grammar, type system,
algebraic data types, and operators in Relay, respectively.

.. toctree::
   :maxdepth: 2

   relay_expr
   relay_type
   relay_adt
   relay_op

Hybrid Script
-------------

The below page describes the TVM hybrid script front-end,
which uses software emulation to support some constructs not
officially supported in TVM.

.. toctree::
   :maxdepth: 2

   hybrid_script
