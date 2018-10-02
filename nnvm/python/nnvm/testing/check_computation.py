# pylint: disable=cell-var-from-loop,no-else-return
"""Helper utilities to check functions and their gradients."""
from __future__ import absolute_import as _abs

import logging
import numpy as np

import tvm
from tvm.contrib import graph_runtime

import nnvm
from nnvm.compiler import graph_util
from nnvm.compiler.graph_attr import TCODE_TO_DTYPE, DTYPE_TO_TCODE
from .config import ctx_list

def infer_shapes_dtypes(graph, shape=None, dtype=None, fallback_dtype=None):
    """Runs dtype and shape inference passes on a graph and returns the resulting graph
    along with the inferred information.

    Parameters
    ----------
    graph : nnvm.graph.Graph
        A graph we want to run inference on.

    shape : Dict[str, Tuple[int]] or Tuple[int], optional
        A dict mapping input variable names to shapes.
        By default shapes will be inferred from variables' attributes.
        Note that this parameter takes precedence over variables' attributes.

    dtype : Dict[str, str] or str, optional
        A dict mapping input variable names to dtypes, or just a single dtype.
        By default dtypes will be inferred from variables' attributes.
        Note that this parameter takes precedence over variables' attributes.

    fallback_dtype : str, optional
        A dtype that will be used for variables whose dtype can't be inferred from other
        variables' dtypes.

    Returns
    -------
    graph : nnvm.graph.Graph
        The resulting graph with dtype and shape information on its nodes.

    input_shapes : Dict[str, Tuple[int]]
        The inferred shapes of input variables merged with the `shape` dictionary.

    input_dtypes : Dict[str, str]
        The inferred dtypes of input variables merged with the `dtype` dictionary.

    output_shapes : List[Tuple[int]]
        The inferred shapes of outputs.

    output_dtypes : List[str]
        The inferred dtypes of outputs.
    """
    # Preprocess input parameters
    if shape is None:
        provided_shapes = {}
    elif isinstance(shape, dict):
        provided_shapes = shape
    else:
        provided_shapes = {x: shape for x in graph.symbol.list_input_variables()}

    if dtype is None:
        provided_dtypes = {}
    elif isinstance(dtype, dict):
        provided_dtypes = dtype
    else:
        provided_dtypes = {x: dtype for x in graph.symbol.list_input_variables()}

    provided_shapes = _dict_var_to_dict_str(provided_shapes)
    provided_dtypes = _dict_var_to_dict_str(provided_dtypes)

    # The graph may already contain shape and dtype info, so extract it and merge with
    # the user-specified shapes and dtypes (use the user-specified one on contradiction)
    preexisting_shapes = graph.json_attr('shape')
    preexisting_dtypes = graph.json_attr('dtype')

    if preexisting_shapes:
        for x in graph.index.input_names:
            if x not in provided_shapes:
                x_shape = tuple(preexisting_shapes[graph.index.entry_id(x)])
                provided_shapes[x] = x_shape

    if preexisting_dtypes:
        for x in graph.index.input_names:
            if x not in provided_dtypes:
                x_dtype = TCODE_TO_DTYPE[preexisting_dtypes[graph.index.entry_id(x)]]
                provided_dtypes[x] = x_dtype

    # Perform inference
    nnvm.compiler.graph_attr.set_shape_inputs(graph, provided_shapes)
    nnvm.compiler.graph_attr.set_dtype_inputs(graph, provided_dtypes)

    graph = graph.apply('InferShape').apply('InferType')

    inferred_shapes = graph.json_attr('shape')
    inferred_dtypes = graph.json_attr('dtype')

    index = graph.index

    output_shapes = [tuple(inferred_shapes[index.entry_id(entry)])
                     for entry in index.output_entries]
    output_dtypes = [TCODE_TO_DTYPE[inferred_dtypes[index.entry_id(entry)]]
                     for entry in index.output_entries]

    # Postprocess the results
    input_shapes = provided_shapes.copy()
    input_dtypes = provided_dtypes.copy()

    for x in graph.symbol.list_input_variables():
        x_name = x.attr('name')
        x_entry_id = graph.index.entry_id(x_name)
        input_shapes[x_name] = tuple(inferred_shapes[x_entry_id])
        input_dtypes[x_name] = TCODE_TO_DTYPE[inferred_dtypes[x_entry_id]]

    # Merge the original user-specified shapes in case some of them are specified for non-existing
    # variables
    for x_name, x_shape in provided_shapes.items():
        x_shape = tuple(x_shape)
        if input_shapes.get(x_name, x_shape) != x_shape:
            raise RuntimeError("Inferred shape differs from the provided shape.\n"
                               "Provided shapes: {}\nInferred shapes: {}"
                               .format(provided_shapes, input_shapes))
        else:
            input_shapes[x_name] = x_shape

    # Merge the original user-specified dtypes
    for x_name, x_dtype in provided_dtypes.items():
        if not isinstance(x_dtype, str):
            x_dtype = TCODE_TO_DTYPE[x_dtype]
        if input_dtypes.get(x_name, x_dtype) != x_dtype:
            raise RuntimeError("Inferred dtype differs from the provided dtype.\n"
                               "Provided dtypes: {}\nInferred dtypes: {}"
                               .format(provided_dtypes, input_dtypes))
        else:
            input_dtypes[x_name] = x_dtype

    # If some dtypes weren't inferred and there is a fallback dtype, assign it to those varibles
    # and repeat the inference
    if fallback_dtype is not None and not all(input_dtypes.values()):
        input_dtypes = {x: input_dtypes[x] if input_dtypes[x] else fallback_dtype
                        for x in input_dtypes}
        return infer_shapes_dtypes(graph, input_shapes, input_dtypes, fallback_dtype=None)

    return graph, input_shapes, input_dtypes, output_shapes, output_dtypes

def graph_to_function(graph, target, ctx, shape=None, dtype=None):
    """Convert a graph to a function taking a keyword args and returning a list of results
    (both args and results are numpy arrays).

    Example::

        fun = graph_to_function(graph, llvm, cpu(0))
        [res1, res2] = fun(x=np.zeros((1,2)), y=np.zeros((1,)))

    Parameters
    ----------
    graph : nnvm.graph.Graph
        A graph we want to convert to a function.

    target : str or :any:`tvm.target.Target`
        The build target

    ctx : TVMContext
        The context to deploy the module.

    shape : Dict[str, Tuple[int]], optional
        A dict mapping input variable names to shapes.
        By default shapes will be inferred from variables' attributes.
        Note that this parameter takes precedence over variables' attributes.

    dtype : Dict[str, str] or str, optional
        A dict mapping input variable names to dtypes, or just a single dtype.
        By default dtypes will be inferred from variables' attributes.
        Note that this parameter takes precedence over variables' attributes.

    Returns
    -------
    function : Callable[..., List[numpy.ndarray]]
    """
    # Infer missing shapes and dtypes
    graph, shape, dtype, output_shapes, output_dtypes = \
        infer_shapes_dtypes(graph, shape=shape, dtype=dtype)

    if None in dtype.values():
        raise ValueError("Input variables with no type: {}".format(dtype))

    if not all(shape.values()):
        raise ValueError("Input variables with no shape: {}".format(shape))

    compute_graph, lib, params = nnvm.compiler.build(graph, target, shape=shape, dtype=dtype)
    module = graph_runtime.create(compute_graph, lib, ctx)

    if params:
        module.set_inputs(**params)

    def run(**kwargs):
        module.run(**kwargs)
        res = []
        for i, (o_shape, o_dtype) in enumerate(zip(output_shapes, output_dtypes)):
            res.append(module.get_output(i, tvm.nd.empty(o_shape, o_dtype)).asnumpy())
        return res

    return run

def _dict_var_to_dict_str(dictionary):
    """Convert a Dict[nnvm.Symbol, T] to Dict[str, T]"""
    if isinstance(dictionary, dict):
        return {s.attr('name') if isinstance(s, nnvm.symbol.Symbol) else s:
                dictionary[s] for s in dictionary}
    else:
        return dictionary

def check_function(symbol, forward=None, backward=None, grad_input_vars=None,
                   shape=None, dtype=None, in_range=None, values=None,
                   exclude_targets=None, only_targets=None,
                   additional_params=None,
                   numerical_grads=None, numerical_grads_params=None,
                   atol=1e-5, rtol=1e-5, quiet=False):
    """Compute the function and/or its gradients on a random input and raise
    an exception if the result doesn't match the reference implementation.

    Parameters
    ----------
    symbol : nnvm.Symbol
        A symbol representing the output.

    forward : Callable[..., List[numpy.ndarray]], optional
        A reference implementation to compare with.

    backward : Callable[..., List[numpy.ndarray] or Dict[str, numpy.ndarray]], optional
        A reference implementation of gradients. Should also accept head_grads besides
        normal inputs which is a list of gradients of some scalar wrt the outputs or just a
        single gradient if there are multiple outputs.
        Should return either a dict mapping input variable names to the respective
        gradients or a list of gradients wrt variables from grad_input_vars in
        exactly the same order (in alphabetical order by default).

    grad_input_vars : List[nnvm.Symbol or str], optional
        A list of variables with respect to which the gradients will be computed.
        None (default) means that all input variables will be used in an alphabetical order.

    shape : Dict[nnvm.Symbol or str, Tuple[int]] or Tuple[int], optional
        A dict mapping input variable names to shapes, or just a single shape.
        By default shapes will be inferred from variables' attributes (see the Examples).
        Note that this parameter takes precedence over variables' attributes.

    dtype : Dict[nnvm.Symbol or str, str] or str, optional
        A dict mapping input variable names to dtypes, or just a single dtype.
        By default dtypes will be inferred from variables' attributes (see the Examples).
        If dtypes cannot be inferred for some variables then float32 will be used as a fallback.
        Note that this parameter takes precedence over variables' attributes.

    in_range : Dict[nnvm.Symbol or str, (float, float)] or (float, float), optional
        A dict mapping input variable names to ranges or just a single range
        (the same for all variables). Input values will be generated from
        uniform distributions on these ranges. `head_grads` can also be
        assigned a range this way.

    values : Dict[nnvm.Symbol or str, numpy.ndarray], optional
        A dict explicitly providing values for some variables instead of random generation.

    exclude_targets : Set[str], optional
        Skip compiling and running anything for these targets.

    only_targets : Set[str], optional
        Test only for those targets from `ctx_list()` that are also in this set.

    additional_params : dict, optional
        A dict of additional parameters which will be passed to forward and backward.

    numerical_grads : bool or 'if_possible', optional
        Whether to additionally check against numerically computed gradients. If 'if_possible' or
        None is passed (which is the default) then it will try to create a gradient computation
        graph and then check gradients numerically only if this graph can be created (i.e. if there
        are some operations with unimplemented gradients, it will just issue a warning).
        Checking against numerical gradients is done via the `check_numerical_grads` function.

    numerical_grads_params : dict, optional
        Additional parameters for `check_numerical_grads`.

    atol : float, optional
        Absolute tolerance for `np.testing.assert_allclose`. NOT used for numerical gradients.

    rtol : float, optional
        Relative tolerance for `np.testing.assert_allclose`. NOT used for numerical gradients.

    quiet : bool, optional
        Don't dump additional information to stdout on failure.

    Examples
    --------
    .. code-block:: python

        x = sym.Variable("x", shape=(1, 2))
        y = sym.Variable("y", shape=(1, 2))

        # check the function and its gradients both numerically and using a reference function
        check_function(x + 2*y,
                       lambda x, y: x + 2*y,
                       lambda x, y, head_grads: {'x': head_grads, 'y': 2*head_grads})

        # just check gradients numerically
        check_function(x + 2*y, numerical_grads=True)

        # just check the forward computation
        check_function(x + 2*y, lambda x, y: x + 2*y, numerical_grads=False)

        # specifying dtype
        check_function(x + 2*y, lambda x, y: x + 2*y, dtype='float64')

        # dtypes can also be specified during variable creation with dtype codes
        x = sym.Variable("x", dtype=0)
        check_function(x + 1, shape=(2, 2), numerical_grads=True)
    """
    # validate and preprocess the input params
    if numerical_grads is None and forward is None and backward is None:
        raise ValueError("No reference function was passed to check_function. If you only want to "
                         "check gradients numerically, pass numerical_grads=True explicitly.")

    if numerical_grads is None:
        numerical_grads = 'if_possible'

    if numerical_grads not in [False, True, 'if_possible']:
        raise ValueError("numerical_grads must be a bool or 'if_possible', not {}"
                         .format(numerical_grads))

    if additional_params is None:
        additional_params = {}

    input_vars = symbol.list_input_variables()
    input_dict = {x.attr('name'): x for x in input_vars}

    if grad_input_vars is None:
        grad_input_vars = sorted(input_vars, key=lambda x: x.attr('name'))
    else:
        grad_input_vars = [input_dict[x] if isinstance(x, str) else x for x in grad_input_vars]

    in_range = _dict_var_to_dict_str(in_range)
    values = _dict_var_to_dict_str(values)

    out_len = len(symbol.list_output_names())

    # Infer the output shapes and dtypes, and preprocess the shape and dtype params
    forward_graph, shape, dtype, out_shapes, out_dtypes = \
        infer_shapes_dtypes(nnvm.graph.create(symbol), shape=shape, dtype=dtype,
                            fallback_dtype='float32')

    if not all(out_shapes) or not all(out_dtypes):
        if not quiet:
            print(forward_graph.ir(join_node_attrs=['shape', 'dtype']))
        raise ValueError("Could not infer shapes or dtypes for outputs.\n"
                         "out_shapes = {}\nout_dtypes = {}".format(out_shapes, out_dtypes))

    backward_graph = None

    # If we want gradients, we have to recreate the graph, but now with gradient computations
    # Note that here we need out_shapes for defining the shape of head grads, so we have to
    # create the graph twice
    if backward is not None or numerical_grads:
        try:
            head_grads_symbols = [nnvm.symbol.Variable("head_grads_" + str(i),
                                                       shape=out_shapes[i],
                                                       dtype=DTYPE_TO_TCODE[out_dtypes[i]])
                                  for i in range(out_len)]
            grad_symbols = graph_util.gradients([symbol], grad_input_vars,
                                                grad_ys=head_grads_symbols)
            # Sometimes grads do not depend on head_grads, so head_grads does not appear
            # in the variable list; adding it manually prevents this, making things a bit easier
            backward_graph = \
                nnvm.graph.create(nnvm.symbol.Group([symbol] + grad_symbols + head_grads_symbols))

            backward_graph, shape, dtype, out_shapes, out_dtypes = \
                infer_shapes_dtypes(backward_graph, shape=shape, dtype=dtype,
                                    fallback_dtype='float32')
        except nnvm._base.NNVMError as err:
            if backward is None and numerical_grads == "if_possible":
                logging.warning("Won't check gradients because: %s", str(err).split('\n', 1)[0])
                numerical_grads = False
                backward_graph = None
            else:
                raise

    main_graph = backward_graph if backward_graph is not None else forward_graph

    # Generate random data for inputs (including head_grads)

    np_inputs = {}

    for x in main_graph.symbol.list_input_variables():
        x_name = x.attr('name')
        x_shape = shape[x_name]
        x_dtype = dtype[x_name]

        if values is not None and x_name in values:
            np_inputs[x_name] = values[x_name].astype(x_dtype)
            continue

        low = -1.0
        high = 1.0
        if in_range is not None:
            if isinstance(in_range, dict):
                if x_name in in_range:
                    low = in_range[x_name][0]
                    high = in_range[x_name][1]
            else:
                low = in_range[0]
                high = in_range[1]

        np_inputs[x_name] = np.random.uniform(size=x_shape, low=low, high=high).astype(x_dtype)

    np_inputs_without_head_grads = {k: np_inputs[k] for k in np_inputs
                                    if not k.startswith('head_grads_')}

    nothing_was_done = True

    # Compute and compare the results
    for target, ctx in ctx_list():
        if exclude_targets is not None:
            if target in exclude_targets or str(target) in exclude_targets:
                logging.info("Skipping target = %s, ctx = %s", target, ctx)
                continue
        if only_targets is not None:
            if target not in only_targets and str(target) not in only_targets:
                logging.info("Skipping target = %s, ctx = %s", target, ctx)
                continue

        logging.info("Checking computation on target = %s, ctx = %s", target, ctx)

        debug_stage = None

        try:
            nnvm_res = None

            debug_stage = "compiling"
            main_function = graph_to_function(main_graph, target, ctx)

            # nnvm_res contains the output and gradients (if they are needed)
            debug_stage = "running"
            nnvm_res = main_function(**np_inputs)

            if backward_graph is not None:
                grad_var_names = [x.attr('name') for x in grad_input_vars]
                nnvm_grads = {x: v for x, v in zip(grad_var_names, nnvm_res[out_len:])}

            if forward is not None:
                nothing_was_done = False
                debug_stage = "checking forward computation"
                logging.debug(debug_stage)

                params = {}
                params.update(np_inputs_without_head_grads)
                params.update(additional_params)
                numpy_res = forward(**params)

                if isinstance(numpy_res, tuple):
                    numpy_res = list(numpy_res)

                if not isinstance(numpy_res, list):
                    numpy_res = [numpy_res]

                if len(numpy_res) != out_len:
                    raise ValueError("Forward function returned {} values, but "
                                     "the nnvm graph returns {} values"
                                     .format(len(numpy_res), out_len))

                for i in range(out_len):
                    np.testing.assert_allclose(nnvm_res[i], numpy_res[i], atol=atol, rtol=rtol)

            if backward is not None:
                nothing_was_done = False
                debug_stage = "checking gradients"
                logging.debug(debug_stage)

                np_head_grads = [np_inputs["head_grads_" + str(i)] for i in range(out_len)]

                if out_len == 1:
                    np_head_grads = np_head_grads[0]

                params = {'head_grads': np_head_grads}
                params.update(np_inputs_without_head_grads)
                params.update(additional_params)
                numpy_grads = backward(**params)

                if not isinstance(numpy_grads, dict):
                    if isinstance(numpy_grads, tuple):
                        numpy_grads = list(numpy_grads)
                    if not isinstance(numpy_grads, list):
                        numpy_grads = [numpy_grads]
                    numpy_grads = {x: v for x, v in zip(grad_var_names, numpy_grads)}
                    if len(numpy_grads) != len(grad_var_names):
                        raise ValueError("The backward function returns a list of gradients which "
                                         "does not contain gradients for these variables: {}"
                                         .format(set(grad_var_names) - set(numpy_grads)))

                for x_name in numpy_grads:
                    np.testing.assert_allclose(nnvm_grads[x_name], numpy_grads[x_name],
                                               atol=atol, rtol=rtol)

            if numerical_grads:
                nothing_was_done = False
                debug_stage = "checking gradients numerically"
                logging.debug(debug_stage)

                forward_function = graph_to_function(forward_graph, target, ctx)

                # Since the result may be non-scalar, we have to put another operation on the top,
                # so we just multiple by the randomly generated head_grads and then sum everything.
                # This way we can reuse the gradient values which has been already computed.
                def scalar_function(**kwargs):
                    res = forward_function(**kwargs)
                    return np.sum([np.dot(np_inputs['head_grads_' + str(i)].ravel(), res[i].ravel())
                                   for i in range(out_len)])

                if numerical_grads_params is None:
                    numerical_grads_params = {}

                check_numerical_grads(
                    scalar_function,
                    input_values=np_inputs_without_head_grads,
                    grad_values=nnvm_grads,
                    **numerical_grads_params)

        except:
            if not quiet:
                print("\ncheck_function failed while {}, here is the main graph"
                      .format(debug_stage))
                print(main_graph.ir(join_node_attrs=['shape', 'dtype']))
                if nnvm_res is not None:
                    print("Generated inputs:")
                    print(np_inputs)
                    print()
            raise

    if nothing_was_done:
        logging.warning("Nothing was done in check_function. Check ctx_list().")


def check_numerical_grads(function, input_values, grad_values, function_value=None,
                          delta=1e-3, atol=1e-2, rtol=0.1):
    """A helper function that checks that numerical gradients of a function are equal to
    gradients computed in some different way (analytical gradients).

    Numerical gradients are computed using finite difference approximation. To reduce the number of
    function evaluations, the number of points used is gradually increased if the error value is
    too high (up to 5 points).

    Parameters
    ----------
    function
        A function that takes inputs as keyword arguments (like `function(**input_values)`) and
        returns a scalar result. Should accept numpy ndarrays.

    input_values : Dict[str, numpy.ndarray]
        A dict assigning values to variables. Represents the point at which gradients should be
        computed.

    grad_values : Dict[str, numpy.ndarray]
        Gradients computed using a different method.

    function_value : float, optional
        Should be equal to `function(**input_values)`.

    delta : float, optional
        A small number used for numerical computation of partial derivatives. The default 1e-3 is a
        good choice for float32.

    atol : float, optional
        Absolute tolerance.

    rtol : float, optional
        Relative tolerance.
    """

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

        # compute partial derivatives for each position in this variable
        for j in range(np.prod(grad.shape)):
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

            ngrad.reshape(-1)[j] = nder

        dist = np.sqrt(np.sum((ngrad - grad)**2))
        grad_norm = np.sqrt(np.sum(ngrad**2))

        if not (np.isfinite(dist) and np.isfinite(grad_norm)):
            raise ValueError(
                "NaN or infinity detected during numerical gradient checking wrt {}\n"
                "analytical grad = {}\n numerical grad = {}\n"
                .format(x_name, grad, ngrad))

        # we multiple atol by this number to make it more universal for different sizes
        sqrt_n = np.sqrt(float(np.prod(grad.shape)))

        if dist > atol*sqrt_n + rtol*grad_norm:
            raise AssertionError(
                "Analytical and numerical grads wrt {} differ too much\n"
                "analytical grad = {}\n numerical grad = {}\n"
                "distance > atol*sqrt(n) + rtol*grad_norm\n"
                "distance {} > {}*{} + {}*{}"
                .format(x_name, grad, ngrad,
                        dist, atol, sqrt_n, rtol, grad_norm))

        max_diff = np.max(np.abs(ngrad - grad))
        avg_diff = np.mean(np.abs(ngrad - grad))
        logging.info("Numerical grad test wrt %s of shape %s passes, "
                     "dist = %f, max_diff = %f, avg_diff = %f",
                     x_name, grad.shape, dist, max_diff, avg_diff)
