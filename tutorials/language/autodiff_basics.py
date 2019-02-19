"""
Automatic Differentiation of Tensor Expressions
===============================================
**Author**: `Sergei Grechanik <https://github.com/sgrechanik-h>`_

This tutorial describes how to use automatic differentiation of tensor expressions.

Usually differentiation is done on the level of NNVM/Relay graphs. However there are some situations
when one might want to perform differentiation on the lower level of TVM tensor expressions, e.g.:
  - When you are experimenting with a completely new kind of operations.
  - When gradients for some operations haven't been implemented yet in NNVM/Relay.
  - When you are implementing gradients for a new operation manually and need a starting point.
  - When you want to train models in pure TVM without NNVM/Relay (if you really do, please tell us
    why).

.. note::

    - Automatic differentiation is still work in progress. Some operations are not differentiated
      very well yet.
    - Automatic differentiation doesn't perform scheduling. The generated code should be scheduled
      by hand or using some autoscheduling and autotuning methods (which may require manually
      writing schedule templates).

"""
from __future__ import absolute_import, print_function
import tvm
import topi

######################################################################
# How to use automatic differentiation
# ------------------------------------
#
# Basically, all you need is the function :any:`tvm.differentiate` (also known as
# :any:`tvm.autodiff.differentiate`) which takes a tensor, differentiates it with respect to other
# given tensors using reverse accumulation, and applies certain optimizations. Let's consider an
# example:

# inputs
X = tvm.placeholder((32, 100), name='X')
W = tvm.placeholder((10, 100), name='W')
B = tvm.placeholder((10,), name='B')

# forward computation, basically topi.nn.dense(X, W, B)
k = tvm.reduce_axis((0, 100))
T = tvm.compute((32, 10), lambda i, j: tvm.sum(X[i, k]*W[j, k], k))
Y = topi.add(T, B)
L = topi.sum(Y)

# gradients
[dL_dW, dL_dB] = tvm.differentiate(L, [W, B])

######################################################################
# `L` is a scalar, so the results are gradients, however in general the result is a full Jacobian.
# :any:`tvm.differentiate` also accepts the third parameter if you want to multiply the Jacobian by
# another tensor.

[dY_dW] = tvm.differentiate(Y, [W])
print("Y.shape", Y.shape)
print("W.shape", W.shape)
print("dY_dW.shape", dY_dW.shape)

[dL_dW] = tvm.differentiate(Y, [W], topi.full_like(Y, 1.0))

######################################################################
# The result of :any:`tvm.differentiate` mimics a list, however it is an object that also contains
# all intermediate adjoints. Note also that the list of input tensors may be omitted, in which case
# the output will be differentiated with respect to all the inputs:

res = tvm.differentiate(L)
dL_dX = res.adjoints[X]
dL_dT = res.adjoints[T]
dL_dY = res.adjoints[Y]

######################################################################
# Examples of generated gradients
# -------------------------------
#
# Let's print out some generated code. We'll start with the simple matrix multiplication
# we've already differentiated.

T1 = tvm.compute((32, 10), lambda i, j: tvm.sum(X[i, k]*W[j, k], k), name='matmul')
H1 = tvm.placeholder(T1.shape, name='H1')

[dW] = tvm.differentiate(T1, [W], H1)
print(tvm.PrintTensorRecursively(dW))

######################################################################
# (The only problem here is that an unnecessary intermediate tensor was extracted.)
#
# Now let's look at some problematic operations, like maxpool:

X1 = tvm.placeholder((64, 32, 28, 28), name='X1')
W1 = tvm.placeholder((64, 64, 3, 3), name='W1')
Y1 = topi.nn.pool(X1, [2, 2], [2, 2], [0, 0, 0, 0], 'max')
H1 = tvm.placeholder(Y1.shape, name='H1')

[dX1] = tvm.differentiate(Y1, [X1], H1)
print(tvm.PrintTensorRecursively(dX1))

######################################################################
# Here the elements of the adjoint `H1` are multiplied by the elements of a mask (computed with
# the tensor called `extracted_tensor`). The mask represents whether an element is the maximum of
# its neighborhood. This is not the optimal solution.

######################################################################
# Overriding the differentiation function
# ---------------------------------------
#
# :any:`tvm.differentiate` internally calls a function which performs differentiation of a given
# tensor with respect to one of its inputs. This functions may be overridden for every tensor or for
# some particular tensors, which is useful when the default differentiation function does a poor job
# and we need to provide some gradients manually. Let's define our own naive version of this
# function:

def custom_fdiff(out, inp, head):
    return topi.tensordot(head, tvm.autodiff.Jacobian(out, inp, False), len(out.shape))

######################################################################
# This function must take the tensors `out`, `inp` and `head` where `out` is the tensor that should
# be differentiated with respect to `inp`, `inp` is an immediate dependency of `out`, and `head` is
# the adjoint of `out` which should be multiplied by the result of differentiation. The
# differentiation itself is done using the function :any:`tvm.autodiff.Jacobian`, and the
# multiplication is done with :any:`topi.tensordot`. The default differentiation function
# :any:`tvm.autodiff.DiffBuildingBlock` does the same thing, but it also applies certain optimizing
# transformations.
#
# A custom differentiation function may be used like this:

res = tvm.differentiate(L, fdiff=custom_fdiff)

######################################################################
# A custom differentiation function may be used to override differentiation for certain operations
# by checking if `out` is the operation we want to differentiate differently. However, there is an
# alternative way: using the `override` keyword argument. `override` should be a dict mapping
# tensors to their dependencies and custom differentiation functions.
#
# Let's consider the following scenario: we want to block gradient flow from `Y` to `X` and compute
# gradients of `Y` wrt `B` and `W` using the unoptimized differentiation function `custom_fdiff`.
# Note that `W` and `X` are not immediate dependencies of `Y`.

def custom_fdiff_2(out, inputs, head):
    assert out == Y
    assert inputs == [X, W, B]
    # block gradients to X
    dX = topi.full(head.shape[:-len(out.shape)] + list(X.shape), head.dtype, 0)
    # use the custom unoptimized differentiation function for the rest
    return [dX] + list(tvm.differentiate(out, [W, B], head, fdiff=custom_fdiff))

res = tvm.differentiate(L, override={Y: ([X, W, B], custom_fdiff_2)})

######################################################################
# There are several things to note:
#   - For efficiency reasons the custom differentiation function used in `override` has a slightly
#     different interface than the custom differentiation functions used for `fdiff`, namely it
#     takes a list of inputs instead of a single input, and returns the list of the corresponding
#     adjoints.
#   - We had overridden the dependencies for `Y` (its immediate dependencies are `T` and `B`, but we
#     used `X`, `W` and `B` instead), so we couldn't use :any:`tvm.autodiff.Jacobian` or
#     `custom_fdiff` directly, since they expect the input to be an immediate dependency for the
#     output. That's why we had to wrap them in the call to :any:`tvm.differentiate`.
