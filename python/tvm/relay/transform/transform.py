# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, missing-docstring, unused-import
"""
Relay pass transformation infrastructure.
"""
import types
import inspect
import functools

import tvm.ir
from tvm import te
from tvm.runtime import ndarray as _nd

from tvm import relay
from . import _ffi_api


def build_config(opt_level=2,
                 fallback_device=_nd.cpu(),
                 required_pass=None,
                 disabled_pass=None,
                 trace=None):
    """Configure the build behavior by setting config variables.

    Parameters
    ----------
    opt_level: int, optional
        Optimization level. The optimization pass name and level are as the
        following:

        .. code-block:: python

            OPT_PASS_LEVEL = {
                "SimplifyInference": 0,
                "OpFusion": 1,
                "FoldConstant": 2,
                "FoldScaleAxis": 3,
                "AlterOpLayout": 3,
                "CanonicalizeOps": 3,
                "CanonicalizeCast": 3,
                "EliminateCommonSubexpr": 3,
                "CombineParallelConv2D": 4,
                "CombineParallelDense": 4,
                "FastMath": 4
            }

    fallback_device : int, str, or tvmContext, optional
        The fallback device. It is also used as the default device for
        operators without specified device during heterogeneous execution.

    required_pass: set of str, optional
        Optimization passes that are required regardless of optimization level.

    disabled_pass: set of str, optional
        Optimization passes to be disabled during optimization.

    trace: Callable[[IRModule, PassInfo, bool], None]
        A tracing function for debugging or introspection.

    Returns
    -------
    pass_context: PassContext
        The pass context for optimizations.
    """
    return tvm.ir.transform.PassContext(
        opt_level, fallback_device, required_pass,
        disabled_pass, trace)


@tvm._ffi.register_object("relay.FunctionPass")
class FunctionPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relay.Function in a module. A function
    pass class should be created through `function_pass`.
    """


def InferType():
    """Infer the type of an expr.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered type inference pass.
    """
    return _ffi_api.InferType()


def FoldScaleAxis():
    """Fold the scaling of axis into weights of conv2d/dense. This pass will
    invoke both forward and backward scale folding.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to fold expressions.

    Note
    ----
    Internally, we will call backward_fold_scale_axis before using
    forward_fold_scale_axis as backward folding targets the common conv->bn
    pattern.
    """
    return _ffi_api.FoldScaleAxis()


def BackwardFoldScaleAxis():
    """Backward fold axis scaling into weights of conv2d/dense.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to backward fold expressions.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis as backward folding targets the common
    conv->bn pattern.
    """
    return _ffi_api.BackwardFoldScaleAxis()

def RemoveUnusedFunctions(entry_functions=None):
    """Remove unused global relay functions in a relay module.

    Parameters
    ----------
    entry_functions: list[string]
        The set of entry functions to start from.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to remove unused functions.
    """
    if entry_functions is None:
        entry_functions = ['main']
    return _ffi_api.RemoveUnusedFunctions(entry_functions)

def ForwardFoldScaleAxis():
    """Fold the scaling of axis into weights of conv2d/dense.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to forward fold expressions.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis, as backward folding targets the
    common conv->bn pattern.
    """
    return _ffi_api.ForwardFoldScaleAxis()


def SimplifyInference():
    """Simplify the data-flow graph for inference phase. An simplified expression
    which is semantically equal to the input expression will be returned.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass to perform operator simplification.
    """
    return _ffi_api.SimplifyInference()


def FastMath():
    """ Converts the expensive non linear functions to their fast but approximate counterparts.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass to perform fast math operations.
    """
    return _ffi_api.FastMath()


def CanonicalizeOps():
    """Canonicalize special operators to basic operators.
    This can simplify followed analysis, e.g. expanding bias_add to
    expand_dims and broadcast_add.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass performing the canonicalization.
    """
    return _ffi_api.CanonicalizeOps()


def DeadCodeElimination(inline_once=False):
    """Remove expressions that do not have any users (dead code).

    Parameters
    ----------
    inline_once: Optional[Bool]
        Whether to inline binding that occurs only once.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that eliminates the dead code in a Relay program.
    """
    return _ffi_api.DeadCodeElimination(inline_once)

def LazyGradientInit():
    """Reduces memory usage of gradient tensors

    Parameters
    ----------

    Returns
    -------
    ret: tvm.transform.Pass
        A pass which delays and/or reduces memory allocation,
        by lazily allocating 0 or one filled tensors.
    """
    return _ffi_api.LazyGradientInit()

def FoldConstant():
    """Fold the constant expressions in a Relay program.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for constant folding.
    """
    return _ffi_api.FoldConstant()


def FuseOps(fuse_opt_level=-1):
    """Fuse operators in an expr to a larger operator according to some rules.

    Parameters
    ----------
    fuse_opt_level : int
        The level of fuse optimization. -1 indicates that the level will be
        inferred from pass context.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for operator fusion.
    """
    return _ffi_api.FuseOps(fuse_opt_level)


def CombineParallelConv2D(min_num_branches=3):
    """Combine multiple conv2d operators into one.

    Parameters
    ----------
    min_num_branches : int
        The minimum number of required parallel branches for performing this
        optimization.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that combines parallel conv2d operators.
    """
    return _ffi_api.CombineParallelConv2D(min_num_branches)


def CombineParallelDense(min_num_branches=3):
    """Combine multiple dense operators into one. For example:

    .. code-block
                    data
            /              \
        dense (2,2)         dense (2,2)
            |                 |
        elemwise/bcast (2,2)  elemwise/bcast (2,2)

    Would become:

    .. code-block

                data
                |
            batch_matmul+elemwise/bcast (2,2,2)

    Parameters
    ----------
    min_num_branches : int
        The minimum number of required parallel branches for performing this
        optimization.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that combines parallel dense operators.
    """
    return _ffi_api.CombineParallelDense(min_num_branches)


def AlterOpLayout():
    """Alternate the layouts of operators or replace primitive operators with
    other expressions.
    This pass can be used for computing convolution in custom layouts or
    other general weight pre-transformation.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that alters the layout of operators.
    """
    return _ffi_api.AlterOpLayout()


def ConvertLayout(desired_layout):
    """ Given a dest layout, this pass transforms the expr such that most of the ops input data
    layout is changed to the dest layout. In ideal situation, there are only 2 layout transforms,
    one at the start and one at the end.

    This pass is not a part of relay.build and is expected to be called between framework-relay
    parser and relay.build call. This is very helpful for hardware backends that support/prefer only
    type of data layout.

    RFC - https://discuss.tvm.ai/t/layout-conversion-pass/4009

    This pass uses most of the AlterOpLayout and InferCorrectLayout infrastructure. We can define
    new layouts for conv2d ops for now. Most of the other operators try to adapt to their input
    layout using the InferCorrectLayout infrastructure.

    Parameters
    ----------
    desired_layout : str
      The desired layout for the transformed expr.

    Returns
    -------
    pass: FunctionPass
      The pass.
    """
    return _ffi_api.ConvertLayout(desired_layout)


def Legalize(legalize_map_attr_name="FTVMLegalize"):
    """Legalizes an expression with another expression.
    This pass can be used to replace an expr with another expr for target
    dependent optimizations. For example, one expr, though semnatically
    equivalent to the other, can have better performance on a target. This pass
    can be used to legalize the expr in a target-dependent manner.

    Parameters
    ----------
    legalize_map_attr_name : str
        The Op's attr name which corresponds to the legalize rule function.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that rewrites an expr.
    """
    return _ffi_api.Legalize(legalize_map_attr_name)


def MergeComposite(pattern_table):
    """Merge multiple operators into a single composite relay function.

    Parameters
    ----------
    pattern_table : list(tuple)
        A list of (pattern_name, pattern, check) tuples.
        The order of the patterns in the list will determine the order
        of priority in which they are matched.
        'check' is a function to check whether an extracted pattern matches.
        It can be implemented by pattern writer but if not specified it will
        always return True.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that merges operators into a single composite
        relay function.
    """
    pattern_names = []
    patterns = []
    checks = []
    for tup in pattern_table:
        if len(tup) == 2:
            pattern_name, pattern = tup
            check = lambda extract: True
        elif len(tup) == 3:
            pattern_name, pattern, check = tup

        pattern_names.append(pattern_name)
        patterns.append(pattern)
        checks.append(check)

    return _ffi_api.MergeComposite(pattern_names, patterns, *checks)


def MergeCompilerRegions():
    """Merge together compiler regions.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that merges compiler regions.
    """
    return _ffi_api.MergeCompilerRegions()


def RewriteAnnotatedOps(fallback_device):
    """Rewrite the annotated program where annotation operators, e.g.
    `on_deivce`, mark which device an expression should be scheduled to.
    This pass helps heterogeneous execution where different operators may need
    to be allocated on various devices.

    Parameters
    ----------
    fallback_device : int
        The fallback device type. It is also used as the default device for
        operators with no annotated device.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that rewrites an expression with annotated
        `on_device` operators.
    """
    return _ffi_api.RewriteDeviceAnnotation(fallback_device)


def ToANormalForm():
    """Turn Graph Normal Form expression into A Normal Form Expression.
    The scope of the root expression is the global scope.
    The scope of any non root expression is the least common ancestor of all it's scope.
    Values are ordered by post-DFS order in each scope.

    Returns
    -------
    ret: Union[tvm.transform.Pass, tvm.relay.Expr]
        The registered pass that transforms an expression into A Normal Form.
    """
    return _ffi_api.ToANormalForm()


def ToCPS(expr, mod=None):
    """
    Turn expression into continuation passing style(CPS).

    Every intermediate compute will be passed to a continuation.

    Returns
    -------
    result: tvm.transform.Pass
        The registered pass that transforms an expression into CPS.
    """
    return _ffi_api.to_cps(expr, mod)


def EtaExpand(expand_constructor=False, expand_global_var=False):
    """Add abstraction over a constructor or global variable bound to a function

    Parameters
    ----------
    expand_constructor: bool
        Whether to expand constructors.

    expand_global_var: bool
        Whether to expand global variables.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that eta expands an expression.
    """
    return _ffi_api.EtaExpand(expand_constructor, expand_global_var)


def ToGraphNormalForm():
    """Turn a Relay program in A Normal Form into Graph Normal Form

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that transforms an expression into Graph Normal Form.
    """
    return _ffi_api.ToGraphNormalForm()


def EliminateCommonSubexpr(fskip=None):
    """Eliminate common subexpressions.

    Parameters
    ----------
    fskip: Callable
        The callback function that decides whether an expression should be
        skipped.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that eliminates common subexpressions.
    """
    return _ffi_api.EliminateCommonSubexpr(fskip)


def PartialEvaluate():
    """Evaluate the static fragment of the code.

    Note
    ----
    This transformation could be either `Module -> Module` or `Expr -> Expr`.
    It will directly transform the input expression to a new one if the target
    expression is provided. Otherwise, it will rely on the pass manager to
    carry out transformation.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that performs partial evaluation on an expression.
    """
    return _ffi_api.PartialEvaluate()


def CanonicalizeCast():
    """
    Canonicalize cast expressions to make operator fusion more efficient.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that canonicalizes cast expression.
    """
    return _ffi_api.CanonicalizeCast()


def LambdaLift():
    """
    Lift the closure to global function.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that lifts the lambda function.
    """
    return _ffi_api.LambdaLift()


def PartitionGraph():
    """Partition a Relay program into regions that can be executed on different
    backends.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that partitions the Relay program.
    """
    return _ffi_api.PartitionGraph()



def AnnotateTarget(targets):
    """Annotate ops in an experession with a provied compiler/target and then
    use it for codegen.

    Parameters
    ----------
    targets : str or List[str]
        The list of target compilers used for codegen.

    Returns
    -------
    ret : tvm.transform.Pass
        The annotated pass that wrapps ops with subgraph_start and
        subgraph_end.
    """
    if isinstance(targets, str):
        targets = [targets]
    return _ffi_api.AnnotateTarget([tvm.runtime.container.String(t) for t in targets])


def Inline():
    """Perform inlining on the given Relay IR module. The global functions that
    are marked as `inline` should be always inlined. A cost model will be
    needed in the future to decide if it is profitable to inline the function.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass that performs inlining for a Relay IR module.
    """
    return _ffi_api.Inline()


def gradient(expr, mod=None, mode='higher_order'):
    """
    Transform the input function,
    returning a function that calculate the original result,
    paired with gradient of the input.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, which is a Function or a GlobalVar.

    mod : Optional[tvm.IRModule]

    mode : Optional[String]
        The mode of the automatic differentiation algorithm.
        'first_order' only works on first order code, but will not produce
        reference nor closure.
        'higher_order' works on all code using reference and closure.

    Returns
    -------
    expr : tvm.relay.Expr
      The transformed expression.
    """
    if mode == 'first_order':
        return _ffi_api.first_order_gradient(expr, mod)
    if mode == 'higher_order':
        return _ffi_api.gradient(expr, mod)
    raise Exception('unknown mode')


def to_cps(func, mod=None):
    """
    Turn expression into CPS expression.

    Every intermediate compute will be passed to a continuation.

    Parameters
    ----------
    func: tvm.relay.Function
        The input function.

    mod: Optional[tvm.IRModule]
        The global module.

    Returns
    -------
    result: tvm.relay.Function
      The output function.
    """
    return _ffi_api.to_cps(func, mod)


def un_cps(func):
    """
    Turn an cps function into a Function without the continuation argument.

    Note that this will not give the exact same interface as before cps:
      If the input/output is higher order, they will still be in cps form.

    Parameters
    ----------
    func: tvm.relay.Function
        The input function

    Returns
    -------
    result: tvm.relay.Function
        The output function
    """
    return _ffi_api.un_cps(func)


def _wrap_class_function_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""
    class PyFunctionPass(FunctionPass):
        """Internal wrapper class to create a class instance."""
        def __init__(self, *args, **kwargs):
            # initialize handle in cass pass_cls creation failed.fg
            self.handle = None
            inst = pass_cls(*args, **kwargs)
            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_function(func, mod, ctx)
            self.__init_handle_by_constructor__(
                _ffi_api.MakeFunctionPass, _pass_func, pass_info)
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyFunctionPass.__init__, pass_cls.__init__)
    PyFunctionPass.__name__ = pass_cls.__name__
    PyFunctionPass.__doc__ = pass_cls.__doc__
    PyFunctionPass.__module__ = pass_cls.__module__
    return PyFunctionPass


def function_pass(pass_func=None, opt_level=None, name=None, required=None):
    """Decorate a function pass.

    This function returns a callback when pass_func
    is provided. Otherwise, it returns the created function pass using the
    given optimization function.

    Parameters
    ----------
    pass_func : Optional[Callable[(Function, Module, PassContext) -> Function]]
        The transformation function or class.

    opt_level : int
        The optimization level of this module pass.

    name : Optional[str]
        The name of the function pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the module pass is dependent on.

    Returns
    -------
    create_function_pass : Union[Callable, FunctionPass]

        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new FunctionPass will be returned when we decorate a pass function.
        A new FunctionPass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a function pass class.

    .. code-block:: python

        @relay.transform.function_pass(opt_level=1)
        class TestReplaceFunc:
            def __init__(self, new_func):
                self.new_func = new_func

            def transform_function(self, func, mod, ctx):
                # just for demo purposes
                # transform func to new_func
                return self.new_func

        x = relay.var("x", shape=(10, 20))
        f1 = relay.Function([x], x)
        f2 = relay.Function([x], relay.log(x))
        # fpass is now a special pass that replaces every
        # function to f1
        fpass = TestReplaceFunc(f1)
        # now every function in input_mod is replaced by f1
        res_mod = fpass(input_mod)


    The following code creates a function pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relay.transform.function_pass(opt_level=2)
        def transform(func, mod, ctx):
            # my transformations here.
            return func

        function_pass = transform
        assert isinstance(function_pass, transform.FunctionPass)
        assert function_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = function_pass(m)
        # Now constant folding should have been applied to every function in
        # the provided module m. And the updated module will be returned.
    """

    if opt_level is None:
        raise ValueError("Please provide opt_level for the funtion pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " +
                        "list/tuple.")

    def create_function_pass(pass_arg):
        """Internal function that creates a function pass"""
        fname = name if name else pass_arg.__name__
        info = tvm.transform.PassInfo(opt_level, fname, required)
        if inspect.isclass(pass_arg):
            return _wrap_class_function_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for Module pass")
        return _ffi_api.MakeFunctionPass(pass_arg, info)

    if pass_func:
        return create_function_pass(pass_func)
    return create_function_pass


@function_pass(opt_level=1)
class ChangeBatch:
    """
    Change the batch size.

    Parameters
    ----------
    data: Dict[relay.Var, int]
      A dictionary of all the params to change.
      The keys are all params, and the values are which dimension hold the batch.

    batch_size: int
      The batch size to change to.

    Returns
    -------
    pass: FunctionPass
      The pass.
    """
    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size

    def transform_function(self, func, mod, ctx):
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        change_batch = self
        class ChangeBatchMutator(tvm.relay.ExprMutator):
            def visit_var(self, var):
                if var in change_batch.data:
                    ty = var.type_annotation
                    new_shape = list(ty.shape)
                    new_shape[change_batch.data[var]] = change_batch.batch_size
                    return relay.Var(var.name_hint, relay.TensorType(new_shape, ty.dtype))
                return var
        return ChangeBatchMutator().visit(func)


def DenseToSparse(weight_name, weight_shape):
    """
    Rewrite qualified ```nn.dense operation``` to ```nn.sparse_dense```
    This pass is used in ```data_dep_optimization.bsr_dense```
    Parameters of this pass is generated by ```analysis.sparse_dense.process_params```

    Parameters
    ----------
    weight_name: Array[String]
      Names of weights which qualified sparse contrains

    weight_shape: Array[Array[IntImm]]
      Weights shape in BSR format.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered DenseToSparse pass.
    """
    return _ffi_api.DenseToSparse(weight_name, weight_shape)

def SimplifyFCTranspose(target_weight_name):
    """
    Rewrite ```y = nn.dense(x, transpose(w, [1, 0]))``` to ```y = nn.dense(x, wt)```
    This pass is used in ```data_dep_optimization.simplify_fc_transpose```

    Parameters
    ----------
    weight_name: Array[String]
      Names of weights which qualified ```y = nn.dense(x, transpose(w, [1, 0]))```
      This parameter is generated by ```analysis.search_fc_transpose``` function

    Returns
    -------
    ret : tvm.transform.Pass
        The registered SimplifyFCTranspose pass.
    """
    return _ffi_api.SimplifyFCTranspose(target_weight_name)
