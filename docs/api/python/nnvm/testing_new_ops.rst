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

Testing new operations
----------------------

When adding new operations, it is a good idea to test them. Testing
should be done with the function ``nnvm.testing.check_function``. You
should provide it with the symbol representing the result of a
computation and a reference numpy implementation. By default, it will
also check analytical gradients against numerical gradients if
analytical gradients are implemented for your operation. You can also
pass a reference implementation for the gradients, but numerical
gradients will still be checked. Numerical gradient checking may be
switched off explicitly, but doing this is not a good idea generally.
Here is an example testing the logarithm operation:

.. code:: python

    import numpy as np
    import nnvm
    import nnvm.symbol as sym
    from nnvm.testing.check_computation import check_function

    x = sym.Variable("x")
    y = sym.log(x)

    def forward(x):
        return np.log(x)

    def backward(head_grads, x):
        return [1. / x * head_grads]

    dtype = "float32"
    shape = {'x': (1, 3, 32, 32)}
    check_function(y, forward, backward, in_range=(0.001, 2.0), dtype=dtype, shape=shape)

If you run the code above, you might get an ``AssertionError`` in rare
cases. That’s why it is recommended to run new tests a lot of times.

.. code:: python

    for _ in range(10000):
        check_function(y, forward, backward, in_range=(0.001, 2.0), dtype=dtype, shape=shape)

If you run the code above then sooner or later you will get an exception
which may look like this:

.. code-block:: text

    AssertionError: Analytical and numerical grads wrt x differ too much
    analytical grad = [
            ...
        ]
    numerical grad = [
            ...
        ]
    distance > atol*sqrt(n) + rtol*grad_norm
    distance 308.50885009765625 > 0.01*55.42562584220407 + 0.1*2167.70703125

It means that either you have a mistake in the ``FGradient`` function or
the numerical error is too high. Generally, if you look at the printed
gradients and see that they differ only slightly or just in a single
position, then it is a numerical error. But if the gradients look
completely different, especially if many corresponding positions have
different signs, then it must be something wrong with the analytical
gradient implementation.

Then try to make this error reproducible, and also try to reduce the
shape of inputs, but not too much, a vector of 10 elements is a
reasonable choice. Also you won’t need reference functions ``forward``
and ``backward``, and restricting the number of targets might also be a
good idea. Since the error may manifest itself only in rare cases, you
might want to run it in a loop.

.. code:: python

    shape = {'x': (10,)}
    np.random.seed(42)

    for _ in range(1000):
        check_function(y, in_range=(0.001, 2.0), dtype=dtype, shape=shape,
                       numerical_grads=True, only_targets=['llvm'])

Running this code will result in the following:

.. code-block:: text

    check_function failed while checking gradients numerically, here is the main graph
    Graph(%x, %head_grads_0) {
      %x, shape=[10], dtype=0
      %head_grads_0, shape=[10], dtype=0
      %1 = log(%x), shape=[10], dtype=0
      %3 = elemwise_div(%head_grads_0, %x), shape=[10], dtype=0
      ret %1, %3, %head_grads_0
    }
    graph_attr_keys = [layout_inputs, dtype_num_unknown_nodes, dtype, shape_num_unknown_nodes, shape]

    Generated inputs:
    {'x': array([2.5660574e-01, 1.5313280e+00, 1.0232578e-03, 8.3371508e-01,
           1.0454979e+00, 1.1021420e-01, 1.9461832e+00, 4.5302454e-01,
           6.0909325e-01, 6.0858107e-01], dtype=float32), 'head_grads_0': array([0.4616029 , 0.00394617, 1.4589603 , 1.9337242 , 0.44936267,
           1.3264314 , 1.4840508 , 1.6970023 , 0.84583575, 0.60655886],
          dtype=float32)}

    ...

    AssertionError: Analytical and numerical grads wrt x differ too much
    analytical grad = [1.7988799e+00 2.5769596e-03 1.4257993e+03 2.3194065e+00 4.2980734e-01
     1.2035031e+01 7.6254421e-01 3.7459390e+00 1.3886802e+00 9.9667716e-01]
     numerical grad = [1.7948151e+00 1.9073486e-03 9.9268610e+02 2.3174286e+00 4.2915344e-01
     1.1980057e+01 7.6198578e-01 3.7412643e+00 1.3866425e+00 9.9563599e-01]
    distance > atol*sqrt(n) + rtol*grad_norm
    distance 433.11322021484375 > 0.01*3.1622776601683795 + 0.1*992.7716674804688

In this case the largest difference is in the 2nd position (starting
from 0) which corresponds to input value ``1.0232578e-03``. This value
is too close to the singularity, so the numerical derivative gets too
imprecise. The solution is to shrink the range for ``x``, here, for
example, ``(0.002, 2.0)`` turned out to be enough. Don’t forget to run
lots of tests, so that other people don’t get false positives.

.. code:: python

    for _ in range(100):
        check_function(y, in_range={x: (0.002, 2.0)}, dtype=dtype, shape=(1, 3, 32, 32),
                       numerical_grads=True, only_targets=['llvm'])

If you need a more precise control over which values get passed to the
checking function, you can use ``values={x: ...}``:

.. code:: python

    x_val = np.array([1.2594858e+00, 1.0960974e-01, 1.4975418e+00, 6.3585603e-01,
           1.2692513e-03, 1.0227472e+00, 9.4656967e-02, 5.5306298e-01,
           1.4142460e+00, 1.2631655e-01], dtype=np.float32)
    check_function(y, values={x: x_val}, dtype=dtype, shape=shape,
                   numerical_grads=True, only_targets=['llvm'])
