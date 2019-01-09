import numpy as np
import tvm
from tvm.testing import check_numerical_grads

def test_check_numerical_grads():
    # Functions and their derivatives
    functions = [
        lambda x: (x*x*x, 3*x*x),
        lambda x: (x*x, 2*x),
        lambda x: (np.abs(x), np.sign(x)),
        lambda x: (np.log(np.abs(x)), 1/x),
        lambda x: (np.sqrt(np.abs(x)), np.sign(x)/(2*np.sqrt(np.abs(x)))),
        lambda x: (1/x, -1/(x*x)),
        lambda x: (np.sign(np.sin(1/x)), np.zeros_like(x)),
        lambda x: (x*np.sin(1/x), np.sin(1/x) - np.cos(1/x)/x),
        lambda x: (np.sin(1/x), - np.cos(1/x)/(x*x)),
    ]

    # Avoid values too close to 0 since singularities of our functions are there
    min_x = 0.5

    for func in functions:
        x_input = np.random.uniform(min_x, 10, size=(3, 4))

        # We need a function returning a scalar, so sum the results
        func_forw = lambda x: np.sum(func(x)[0])
        grads = [func(x_input)[1]]

        check_numerical_grads(func_forw, [x_input], grads)

    # Check functions with multiple arguments
    for f1 in functions:
        for f2 in functions:
            x_input = np.random.uniform(min_x, 10, size=(3, 4))
            y_input = np.random.uniform(min_x, 10, size=(3, 4))

            func_forw = lambda x, y: np.sum(f1(x)[0] + f2(y)[0])
            grads = [f1(x_input)[1], f2(y_input)[1]]

            check_numerical_grads(func_forw, [x_input, y_input], grads)

            # Same thing but with keyword arguments
            func_forw = lambda x, y: np.sum(f1(x)[0] + f2(y)[0])
            grads = {'x': f1(x_input)[1], 'y': f2(y_input)[1]}

            check_numerical_grads(func_forw, {'x': x_input, 'y': y_input}, grads)

    def _noise1(x, atol=1e-2, rtol=0.1):
        # We go in random direction using twice the original tolerance to be sure this
        # results in an error
        sqrt_n = np.sqrt(float(np.prod(x.shape)))
        tol = 2*(np.linalg.norm(x)*rtol + atol*sqrt_n)
        noise = np.random.normal(size=x.shape)
        noise = tol * noise / np.linalg.norm(noise)
        return x + noise

    def _noise2(x, atol=1e-2, rtol=0.1):
        # This noise affects just a single component
        sqrt_n = np.sqrt(float(np.prod(x.shape)))
        tol = 2*(np.linalg.norm(x)*rtol + atol*sqrt_n)
        n = np.random.randint(np.prod(x.shape))
        noise = np.zeros_like(x)
        noise.reshape(-1)[n] = tol
        return x + noise

    # Add noise to gradients and check that the function throws
    for f1 in functions:
        for f2 in functions:
            x_input = np.random.uniform(min_x, 10, size=(3, 4))
            y_input = np.random.uniform(min_x, 10, size=(3, 4))

            func_forw = lambda x, y: np.sum(f1(x)[0] + f2(y)[0])
            grads = [_noise1(f1(x_input)[1]), _noise1(f2(y_input)[1])]

            try:
                check_numerical_grads(func_forw, [x_input, y_input], grads)
            except AssertionError as e:
                pass
            else:
                raise AssertionError("check_numerical_grads didn't raise an exception")

            func_forw = lambda x, y: np.sum(f1(x)[0] + f2(y)[0])
            grads = {'x': _noise2(f1(x_input)[1]), 'y': _noise2(f2(y_input)[1])}

            try:
                check_numerical_grads(func_forw, {'x': x_input, 'y': y_input}, grads)
            except AssertionError as e:
                pass
            else:
                raise AssertionError("check_numerical_grads didn't raise an exception")


if __name__ == "__main__":
    test_check_numerical_grads()

