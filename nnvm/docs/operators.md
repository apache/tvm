NNVM Core Operator Specification
================================

Each operator attributes are stored in json format.
tuples are stored as json array.

## Tier 1: Basic Operators

***Enables fully connected nets***

- **dense**
  - attributes
     - units: int  Number of hidden units in the data.
     - use_bias: bool Whether use bias
  - inputs
     - data, 2D Tensor
     - weight, 2D Tensor
     - bias, optional, 1D Tensor
  - outputs
     - output, 2D Tensor

- **relu**
  - inputs
     - data, nD Tensor
  - outputs
     - output, nD Tensor
