=====================
Introduction to Relay
=====================

Relay is a functional, differentiable programming language that has been
designed to be an expressive intermediate representation for machine
learning systems. It features support for closures, control flow, and
recursion, thus allowing for representing more complex models directly
in Relay. Relay also includes a form of dependent typing using type
relations in order to handle shape analysis for operators with complex
requirements on argument shapes.

Additionally, Relay is designed to be easily extensible by
machine learning researchers and practitioners in order to facilitate
the inclusion and development of new large-scale program transformations
and optimizations.

The below pages describe the grammar, type system, and operators in Relay, respectively.

.. toctree::
   :maxdepth: 2

   relay_expr
   relay_type
   relay_op
