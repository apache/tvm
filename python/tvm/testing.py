""" TVM testing utilities """
import numpy as np

def assert_allclose(actual, desired, rtol=1e-7, atol=1e-7):
    """ Version of np.testing.assert_allclose with `atol` and `rtol` fields set
    in reasonable defaults.

    Arguments `actual` and `desired` are not interchangable, since the function
    compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
    often allow `desired` to be close to zero, we generally want non-zero `atol`.
    """
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=True)
