""" TVM testing utilities """
import logging
import numpy as np
import tvm

def assert_allclose(actual, desired, rtol=1e-7, atol=1e-7):
    """ Version of np.testing.assert_allclose with `atol` and `rtol` fields set
    in reasonable defaults.

    Arguments `actual` and `desired` are not interchangable, since the function
    compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
    often allow `desired` to be close to zero, we generally want non-zero `atol`.
    """
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=True)


def check_numerical_grads(function, input_values, grad_values, function_value=None,
                          delta=1e-3, atol=1e-2, rtol=0.1, acceptable_fail_fraction=None):
    """A helper function that checks that numerical gradients of a function are
    equal to gradients computed in some different way (analytical gradients).

    Numerical gradients are computed using finite difference approximation. To
    reduce the number of function evaluations, the number of points used is
    gradually increased if the error value is too high (up to 5 points).

    Parameters
    ----------
    function
        A function that takes inputs either as positional or as keyword
        arguments (either `function(*input_values)` or `function(**input_values)`
        should be correct) and returns a scalar result. Should accept numpy
        ndarrays.

    input_values : Dict[str, numpy.ndarray] or List[numpy.ndarray]
        A list of values or a dict assigning values to variables. Represents the
        point at which gradients should be computed.

    grad_values : Dict[str, numpy.ndarray] or List[numpy.ndarray]
        Gradients computed using a different method.

    function_value : float, optional
        Should be equal to `function(**input_values)`.

    delta : float, optional
        A small number used for numerical computation of partial derivatives.
        The default 1e-3 is a good choice for float32.

    atol : float, optional
        Absolute tolerance. Gets multiplied by `sqrt(n)` where n is the size of a
        gradient.

    rtol : float, optional
        Relative tolerance.

    acceptable_fail_fraction : float, optional
        If not None, raise an error only when the fraction of wrong elements for a gradient is
        higher than this value.
    """
    # If input_values is a list then function accepts positional arguments
    # In this case transform it to a function taking kwargs of the form {"0": ..., "1": ...}
    if not isinstance(input_values, dict):
        input_len = len(input_values)
        input_values = {str(idx): val for idx, val in enumerate(input_values)}

        def _function(_input_len=input_len, _orig_function=function, **kwargs):
            return _orig_function(*(kwargs[str(i)] for i in range(input_len)))
        function = _function

        grad_values = {str(idx): val for idx, val in enumerate(grad_values)}

    if function_value is None:
        function_value = function(**input_values)

    # a helper to modify j-th element of val by a_delta
    def modify(val, j, a_delta):
        val = val.copy()
        val.reshape(-1)[j] = val.reshape(-1)[j] + a_delta
        return val

    # numerically compute a partial derivative with respect to j-th element of the var `name`
    def derivative(x_name, j, a_delta):
        modified_values = {n: modify(val, j, a_delta) if n == x_name else val
                           for n, val in input_values.items()}
        return (function(**modified_values) - function_value)/a_delta

    def compare_derivative(j, n_der, grad):
        der = grad.reshape(-1)[j]
        return np.abs(n_der - der) < atol + rtol*np.abs(n_der)

    for x_name, grad in grad_values.items():
        if grad.shape != input_values[x_name].shape:
            raise AssertionError(
                "Gradient wrt '{}' has unexpected shape {}, expected {} "
                .format(x_name, grad.shape, input_values[x_name].shape))

        ngrad = np.zeros_like(grad)

        wrong_positions = []

        # compute partial derivatives for each position in this variable
        for j in range(int(np.prod(grad.shape))):
            # forward difference approximation
            nder = derivative(x_name, j, delta)

            # if the derivative is not equal to the analytical one, try to use more
            # precise and expensive methods
            if not compare_derivative(j, nder, grad):
                # central difference approximation
                nder = (derivative(x_name, j, -delta) + nder)/2

                if not compare_derivative(j, nder, grad):
                    # central difference approximation using h = delta/2
                    cnder2 = (derivative(x_name, j, delta/2) + derivative(x_name, j, -delta/2))/2
                    # five-point derivative
                    nder = (4*cnder2 - nder)/3

            # if the derivatives still don't match, add this position to the
            # list of wrong positions
            if not compare_derivative(j, nder, grad):
                wrong_positions.append(np.unravel_index(j, grad.shape))

            ngrad.reshape(-1)[j] = nder

        wrong_fraction = len(wrong_positions)/np.prod(grad.shape)

        dist = np.sqrt(np.sum((ngrad - grad)**2))
        grad_norm = np.sqrt(np.sum(ngrad**2))

        if not (np.isfinite(dist) and np.isfinite(grad_norm)):
            raise ValueError(
                "NaN or infinity detected during numerical gradient checking wrt '{}'\n"
                "analytical grad = {}\n numerical grad = {}\n"
                .format(x_name, grad, ngrad))

        # we multiply atol by this number to make it more universal for different sizes
        sqrt_n = np.sqrt(float(np.prod(grad.shape)))

        if dist > atol*sqrt_n + rtol*grad_norm:
            enough_failures = (acceptable_fail_fraction is None or
                               wrong_fraction > acceptable_fail_fraction)
            if enough_failures:
                raise AssertionError(
                    "Analytical and numerical grads wrt '{}' differ too much\n"
                    "analytical grad = {}\n numerical grad = {}\n"
                    "{}% of elements differ, first 10 of wrong positions: {}\n"
                    "distance > atol*sqrt(n) + rtol*grad_norm\n"
                    "distance {} > {}*{} + {}*{}"
                    .format(x_name, grad, ngrad, int(100*wrong_fraction), wrong_positions[:10],
                            dist, atol, sqrt_n, rtol, grad_norm))
            else:
                logging.warning("Analytical and numerical grads wrt '%s' differ, however "
                                "there were not enough wrong elements to raise an error "
                                "(only %d%%)",
                                x_name, int(100*wrong_fraction))

        max_diff = np.max(np.abs(ngrad - grad))
        avg_diff = np.mean(np.abs(ngrad - grad))
        logging.info("Numerical grad test wrt '%s' of shape %s passes, "
                     "dist = %f, max_diff = %f, avg_diff = %f",
                     x_name, grad.shape, dist, max_diff, avg_diff)


class PerformanceEstimate:
    """A result of static performance estimation.

    Parameters
    ----------
    iterations : int
        The total number of iterations of all the loops.

    multiplications : int
        The total number of expensive operations like multiplications.

    memory : int
        The amount of memory to allocate.
    """
    def __init__(self, iterations=0, multiplications=0, memory=0):
        self.iterations = iterations
        self.multiplications = multiplications
        self.memory = memory

    def as_tuple(self):
        return (self.iterations, self.multiplications, self.memory)

    def __add__(self, other):
        return PerformanceEstimate(iterations=self.iterations + other.iterations,
                                   multiplications=self.multiplications + other.multiplications,
                                   memory=self.memory + other.memory)

    def max(self, other):
        return PerformanceEstimate(
            iterations=max(self.iterations, other.iterations),
            multiplications=max(self.multiplications, other.multiplications),
            memory=max(self.memory, other.memory))

    def times(self, iters):
        return PerformanceEstimate(iterations=self.iterations*iters,
                                   multiplications=self.multiplications*iters,
                                   memory=self.memory)

    def __repr__(self):
        return "PerformanceEstimate(iterations={}, multiplications={}, memory={})".format(
            self.iterations, self.multiplications, self.memory)

    def __le__(self, other):
        return \
            self.iterations <= other.iterations and \
            self.multiplications <= other.multiplications and \
            self.memory <= other.memory


def estimate_performance(s, param_values=None, processed_ops=None):
    """Statically estimate performance of statements, expressions and tensors. Note that the
    estimate is very rough, it mustn't be used to predict future performance, its only purpose is
    to detect possible performance regressions.

    Parameters
    ----------
    s
        A statement, an expression, a tensor, an operation, or a list
        of any of the above.

    param_values : Dict[tvm.expr.Var, int], optional
        Values for parameters (free variables).

    Returns
    -------
    estimate : PerformanceEstimate
    """
    from tvm import stmt
    from tvm import expr

    if param_values is None:
        param_values = {}

    if processed_ops is None:
        processed_ops = {}
        res = estimate_performance(s, param_values=param_values, processed_ops=processed_ops)
        for op_est in processed_ops.values():
            res += op_est
        return res

    def est(expression, param_values=param_values, processed_ops=processed_ops):
        return estimate_performance(expression,
                                    param_values=param_values,
                                    processed_ops=processed_ops)

    def _eval(expression, param_values=param_values):
        return tvm.ir_pass.Simplify(tvm.ir_pass.Substitute(expression, param_values)).value

    def _prod(elems):
        res = 1
        for x in elems:
            res *= x
        return res

    if s is None or isinstance(s, (stmt.AssertStmt, stmt.Free, stmt.Prefetch,
                                   expr.ConstExpr, expr.Var, tvm.tensor.PlaceholderOp)):
        return PerformanceEstimate()
    elif isinstance(s, list):
        res = PerformanceEstimate()
        for item in s:
            res += est(item)
        return res
    elif s in processed_ops:
        return PerformanceEstimate()
    elif isinstance(s, stmt.Allocate):
        mem = _prod([_eval(e) for e in s.extents])
        return est(s.condition) + est(s.body) + PerformanceEstimate(memory=mem)
    elif isinstance(s, stmt.Block):
        return est(s.first) + est(s.rest)
    elif isinstance(s, stmt.Evaluate):
        return est(s.value)
    elif isinstance(s, stmt.For):
        body_est = est(s.body)
        body_est.iterations = max(1, body_est.iterations)
        return body_est.times(_eval(s.extent))
    elif isinstance(s, stmt.IfThenElse):
        return est(s.condition) + est(s.then_case) + est(s.else_case)
    elif isinstance(s, stmt.LetStmt):
        return est(s.value) + est(s.body)
    elif isinstance(s, (stmt.ProducerConsumer, stmt.AttrStmt)):
        return est(s.body)
    elif isinstance(s, stmt.Provide):
        return est(s.value)
    elif isinstance(s, stmt.Realize):
        return est(s.condition) + est(s.body)
    elif isinstance(s, stmt.Store):
        return est(s.value) + est(s.index) + est(s.predicate)
    elif isinstance(s, (expr.Mul, expr.Div, expr.Mod)):
        return est(s.a) + est(s.b) + PerformanceEstimate(multiplications=1)
    elif isinstance(s, (expr.BinaryOpExpr, expr.CmpExpr, expr.LogicalExpr)):
        if not hasattr(s, 'b'):
            return est(s.a)
        return est(s.a) + est(s.b)
    elif isinstance(s, expr.Call):
        res = PerformanceEstimate()
        for a in s.args:
            res += est(a)
        if s.call_type == expr.Call.Halide:
            # The estimate is added to processed_ops, we don't need the result here
            est(s.func)
        elif s.name == "tvm_if_then_else":
            pass
        else:
            # expr.If it is a non-halide call (e.g. exp or log), consider it a mul
            res += PerformanceEstimate(multiplications=1)
        return res
    elif isinstance(s, expr.Cast):
        return est(s.value)
    elif isinstance(s, expr.Load):
        return est(s.index) + est(s.predicate)
    elif isinstance(s, expr.Select):
        return est(s.condition) + est(s.true_value) + est(s.false_value)
    elif isinstance(s, expr.Reduce):
        iterations = _prod([_eval(iv.dom.extent) for iv in s.axis])
        res = PerformanceEstimate()
        for id_elem in s.combiner.identity_element:
            res += est(id_elem)
        on_each_iter = est(s.condition)
        for src in s.source:
            on_each_iter += est(src)
        for comb_res in s.combiner.result:
            on_each_iter += est(comb_res)
        on_each_iter.iterations = max(1, on_each_iter.iterations)
        return res + on_each_iter.times(iterations)
    elif isinstance(s, tvm.tensor.Tensor):
        return est(s.op)
    elif isinstance(s, tvm.tensor.ComputeOp):
        iterations = _prod([_eval(iv.dom.extent) for iv in s.axis])
        if s.reduce_axis:
            res = est(s.body[0])
        else:
            res = PerformanceEstimate()
            for b in s.body:
                res += est(b)
        res.iterations = max(1, res.iterations)
        res = res.times(iterations) + PerformanceEstimate(memory=iterations*len(s.body))
        processed_ops[s] = res
        return PerformanceEstimate()

    raise ValueError("Don't know how to estimate performance of {} of type {}"
                     .format(s, type(s)))
