# pylint: disable=cell-var-from-loop
"""Helper utilities to check functions and their gradients."""
from __future__ import absolute_import as _abs

import logging
import numpy as np

import tvm
from tvm.contrib import graph_runtime

import nnvm
from nnvm.compiler import graph_util
from nnvm.compiler.graph_attr import TCODE_TO_DTYPE
from .config import ctx_list

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
        By default shapes will be inferred automatically.

    dtype : Dict[str, str] or str, optional
        A dict mapping input variable names to dtypes, or just a single dtype.
        By default dtypes will be inferred automatically.

    Returns
    -------
    function : Callable[..., List[numpy.ndarray]]
    """

    if shape is None or dtype is None:
        all_shapes = graph.json_attr('shape')
        all_dtypes = graph.json_attr('dtype')

        if all_shapes is None or all_dtypes is None:
            graph = graph.apply('InferShape').apply('InferType')
            all_shapes = graph.json_attr('shape')
            all_dtypes = graph.json_attr('dtype')

        all_dtypes = [None if t == -1 else TCODE_TO_DTYPE[t] for t in all_dtypes]

        if shape is None:
            shape = {x: all_shapes[graph.index.entry_id(x)] for x in graph.index.input_names}

        if dtype is None:
            dtype = {x: all_dtypes[graph.index.entry_id(x)] for x in graph.index.input_names}

    if None in dtype.values():
        raise ValueError("Input variables with no type: {}".format(dtype))

    compute_graph, lib, _ = nnvm.compiler.build(graph, target, shape=shape, dtype=dtype)
    module = graph_runtime.create(compute_graph, lib, ctx)

    all_shapes = compute_graph.json_attr('shape')
    all_dtypes = [TCODE_TO_DTYPE[dtype] for dtype in compute_graph.json_attr('dtype')]

    def run(**kwargs):
        module.run(**kwargs)
        res = []
        for i, out_entry in enumerate(compute_graph.index.output_entries):
            res.append(
                module.get_output(
                    i, tvm.nd.empty(all_shapes[out_entry[0]], all_dtypes[out_entry[0]])).asnumpy())
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
                   shape=None, dtype=None, in_range=None,
                   exclude_targets=None, only_targets=None,
                   numerical_grads='if_possible', delta=1e-3,
                   atol=1e-5, rtol=1e-5, ng_atol=1e-2, ng_rtol=1e-2,
                   ng_max_error=1e+3, ng_max_discarded_frac=0.1,
                   quiet=False, dump_graph=False):
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
        By default shapes will be inferred automatically.

    dtype : Dict[nnvm.Symbol or str, str] or str, optional
        A dict mapping input variable names to dtypes, or just a single dtype.
        By default dtypes will be inferred automatically.

    in_range : Dict[nnvm.Symbol or str, (float, float)] or (float, float), optional
        A dict mapping input variable names to ranges or just a single range
        (the same for all variables). Input values will be generated from
        uniform distributions on these ranges. `head_grads` can also be
        assigned a range this way.

    exclude_targets : Set[str]
        Skip compiling and running anything for these targets.

    only_targets : Set[str]
        Test only for those targets from `ctx_list()` that are also in this set.

    numerical_grads : bool or "if_possible"
        Whether to additionally check against numerically computed gradients. If 'if_possible' is
        passed (which is the default) then it will try to create a gradient computation graph and
        then check gradients numerically only if this graph can be created (i.e. if there are some
        operations with unimplemented gradients, it will just issue a warning).

    delta : float
        A small value used for numerical gradient computation (usually called h in textbooks).

    atol : float
        Absolute tolerance.

    rtol : float
        Relative tolerance.

    ng_atol : float
        Absolute tolerance for numerical grad checking.

    ng_rtol : float
        Relative tolerance for numerical grad checking.

    ng_max_error : float
        Discard numerical partial derivatives whose estimated error is larger than this value.

    ng_max_discarded_frac : float
        Allow discarding no more than this fraction of partial derivatives.

    quiet : bool
        Don't dump additional information to stdout on failure.

    dump_graph : bool
        Dump the graph even on success.
    """

    if numerical_grads not in [False, True, 'if_possible']:
        raise ValueError("numerical_grads must be a bool or 'if_possible', not {}"
                         .format(numerical_grads))

    input_vars = symbol.list_input_variables()
    input_dict = {x.attr('name'): x for x in input_vars}

    if grad_input_vars is None:
        grad_input_vars = sorted(input_vars, key=lambda x: x.attr('name'))
    else:
        grad_input_vars = [input_dict[x] if isinstance(x, str) else x for x in grad_input_vars]

    if shape is None:
        shape = {}

    if not isinstance(shape, dict):
        shape = {x: shape for x in input_dict}

    shape = _dict_var_to_dict_str(shape)
    dtype = _dict_var_to_dict_str(dtype)
    in_range = _dict_var_to_dict_str(in_range)

    # Infer the output shape and dtype by creating a graph and running passes

    forward_graph = nnvm.graph.create(symbol)

    if dtype is not None:
        nnvm.compiler.graph_attr.set_dtype_inputs(forward_graph, dtype)

    if shape is not None:
        nnvm.compiler.graph_attr.set_shape_inputs(forward_graph, shape)

    forward_graph = forward_graph.apply('InferShape').apply('InferType')
    shapes = forward_graph.json_attr('shape')
    dtypes = forward_graph.json_attr('dtype')

    out_len = len(symbol.list_output_names())

    out_shapes = [shapes[forward_graph.index.output_entries[i][0]] for i in range(out_len)]
    out_dtypes = [dtypes[forward_graph.index.output_entries[i][0]] for i in range(out_len)]

    backward_graph = None

    # If we want gradients, we have to recreate the graph, but now with gradient computations
    # Note that here we need out_shape for defining the shape of head grads, so we have to
    # create the graph twice
    if backward is not None or numerical_grads:
        try:
            head_grads_symbols = [nnvm.symbol.Variable("head_grads_" + str(i),
                                                       shape=out_shapes[i], dtype=out_dtypes[i])
                                  for i in range(out_len)]
            grad_symbols = graph_util.gradients([symbol], grad_input_vars,
                                                grad_ys=head_grads_symbols)
            # Sometimes grads do not depend on head_grads, so head_grads does not appear
            # in the variable list; adding it manually prevents this, making things a bit easier
            backward_graph = \
                nnvm.graph.create(nnvm.symbol.Group([symbol] + grad_symbols + head_grads_symbols))

            if dtype is not None:
                nnvm.compiler.graph_attr.set_dtype_inputs(backward_graph, dtype)

            if shape is not None:
                nnvm.compiler.graph_attr.set_shape_inputs(backward_graph, shape)

            backward_graph = backward_graph.apply('InferShape').apply('InferType')
            shapes = backward_graph.json_attr('shape')
            dtypes = backward_graph.json_attr('dtype')
        except nnvm._base.NNVMError as err:
            if backward is None and numerical_grads == "if_possible":
                logging.warning("Won't check gradients because: %s", str(err).split('\n', 1)[0])
                numerical_grads = False
                backward_graph = None
            else:
                raise

    main_graph = backward_graph if backward_graph is not None else forward_graph

    if dump_graph:
        print()
        print(main_graph.ir(join_node_attrs=['shape', 'dtype']))

    # Generate random data for inputs (including head_grads)

    np_inputs = {}

    for x in main_graph.symbol.list_input_variables():
        x_name = x.attr('name')

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

        x_node_id = main_graph.index.node_id(x_name)
        x_shape = shapes[x_node_id]
        x_dtype = dtypes[x_node_id]

        if not isinstance(x_dtype, str):
            x_dtype = TCODE_TO_DTYPE[x_dtype]

        x_value = np.random.uniform(size=x_shape, low=low, high=high).astype(x_dtype)

        np_inputs[x_name] = x_value

    np_inputs_without_head_grads = {k: np_inputs[k] for k in np_inputs
                                    if not k.startswith('head_grads_')}

    # Compute and compare the results
    for target, ctx in ctx_list():
        if (exclude_targets is not None and (target in exclude_targets or
                                             str(target) in exclude_targets)) or\
           (only_targets is not None and not (target in only_targets or
                                              str(target) in only_targets)):
            logging.info("Skipping target = %s, ctx = %s", target, ctx)
            continue

        logging.info("Checking computation on target = %s, ctx = %s", target, ctx)

        debug_stage = None

        try:
            debug_stage = "compiling"
            nnvm_res = None
            main_function = graph_to_function(main_graph, target, ctx)
            # nnvm_res contains the output and gradients (if they are needed)
            nnvm_res = main_function(**np_inputs)

            if backward_graph is not None:
                grad_var_names = [x.attr('name') for x in grad_input_vars]
                nnvm_grads = {x: v for x, v in zip(grad_var_names, nnvm_res[out_len:])}

            if forward is not None:
                debug_stage = "checking forward computation"
                logging.debug(debug_stage)

                numpy_res = forward(**np_inputs_without_head_grads)

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
                debug_stage = "checking gradients"
                logging.debug(debug_stage)

                np_head_grads = [np_inputs["head_grads_" + str(i)] for i in range(out_len)]

                if out_len == 1:
                    np_head_grads = np_head_grads[0]

                numpy_grads = backward(head_grads=np_head_grads, **np_inputs_without_head_grads)
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
                debug_stage = "checking gradients numerically"
                logging.debug(debug_stage)

                forward_function = graph_to_function(forward_graph, target, ctx)

                # Since the result may be non-scalar, we have to put another operation on the top,
                # so we just multiple by the randomly generated head_grads and then sum everything.
                # This way we can reuse the gradient values which has been already computed.
                def function(**kwargs):
                    res = forward_function(**kwargs)
                    return np.sum([np.dot(np_inputs['head_grads_' + str(i)].ravel(), res[i].ravel())
                                   for i in range(out_len)])

                check_numerical_grads(
                    function,
                    input_values=np_inputs_without_head_grads,
                    grad_values=nnvm_grads,
                    delta=delta,
                    max_error=ng_max_error,
                    max_discarded_frac=ng_max_discarded_frac,
                    atol=ng_atol, rtol=ng_rtol)

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


def check_numerical_grads(function, input_values, grad_values, function_value=None,
                          delta=1e-3, max_error=1e+3, max_discarded_frac=0.1,
                          atol=1e-2, rtol=1e-2):
    """A helper function that checks that numerical gradients of a function are equal to
    gradients computed in some different way.

    We compute two approximations for each gradient using forward and backward differences.
    Then we use the distance between these gradients as our estimate for the error and check if the
    provided analytical gradient lies within this distance from the central difference approximation
    of the gradient.

    Parameters
    ----------
    function
        A function that takes inputs as keyword arguments (like `function(**input_values)`) and
        returns a scalar result. Should accept and return numpy arrays.

    input_values : Dict[str, numpy.ndarray]
        A dict assigning values to variables. Represents the point at which gradients should be
        computed.

    grad_values : Dict[str, numpy.ndarray]
        Gradients computed using a different method.

    function_value : float, optional
        Should be equal to `function(**input_values)`.

    delta : float
        A small number used for numerical computation of partial derivatives.

    max_error : float
        Discard numerical partial derivatives whose estimated error is larger than this value.

    max_discarded_frac : float
        Allow discarding no more than this fraction of partial derivatives.

    atol : float
        Absolute tolerance.

    rtol : float
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

    for x_name, grad in grad_values.items():
        if grad.shape != input_values[x_name].shape:
            raise AssertionError(
                "Gradient wrt '{}' has unexpected shape {}, expected {} "
                .format(x_name, grad.shape, input_values[x_name].shape))

        discarded_count = 0
        nondiscarded_count = 0

        ngrad1 = np.zeros_like(grad)
        ngrad2 = np.zeros_like(grad)

        # compute partial derivatives for each position in this variable
        for j in range(np.prod(grad.shape)):
            ngrad1.reshape(-1)[j] = derivative(x_name, j, delta)
            ngrad2.reshape(-1)[j] = derivative(x_name, j, -delta)

            error_estimation = np.abs(ngrad1.reshape(-1)[j] - ngrad2.reshape(-1)[j])
            if error_estimation > max_error or not np.isfinite(error_estimation):
                logging.warning("Estimated error for the partial derivative number %i wrt %s "
                                "is too large (or nan): %f; This derivative will be discarded.",
                                j, x_name, error_estimation)
                ngrad1.reshape(-1)[j] = ngrad2.reshape(-1)[j] = grad.reshape(-1)[j]
                discarded_count += 1
            else:
                nondiscarded_count += 1

        if discarded_count / (discarded_count + nondiscarded_count) > max_discarded_frac:
            raise AssertionError(
                "Too many ({} out of {}) numerical derivatives wrt {} were discarded "
                "because of high estimated error. Try changing parameters like delta "
                "or in_range, or turn off numerical grad checking for this function."
                .format(discarded_count, discarded_count + nondiscarded_count, x_name))

        ngrad = (ngrad1 + ngrad2)/2
        dist_1_2 = np.linalg.norm(ngrad1 - ngrad2)
        dist_n_s = np.linalg.norm(ngrad - grad)

        if dist_n_s > dist_1_2 + atol + np.linalg.norm(ngrad)*rtol:
            raise AssertionError(
                "Analytical and numerical grads wrt {} differ too much\n"
                "analytical grad = {}\n numerical grad = {}\ndistance {} > {} + {} + {}*{}"
                .format(x_name, grad, ngrad,
                        dist_n_s, dist_1_2, atol, np.linalg.norm(ngrad), rtol))

        max_diff = np.max(np.abs(ngrad - grad))
        avg_diff = np.mean(np.abs(ngrad - grad))
        logging.info("Numerical grad test wrt %s of shape %s passes, "
                     "dist = %f, max_diff = %f, avg_diff = %f",
                     x_name, grad.shape, dist_n_s, max_diff, avg_diff)
