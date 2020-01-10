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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
"""
Relay pass transformation infrastructure.
"""
import types
import inspect
import functools

import tvm
from tvm._ffi.runtime_ctypes import TVMContext
from tvm import relay
from . import _transform
from .base import RelayNode, register_relay_node
from .. import nd as _nd


@register_relay_node
class PassInfo(RelayNode):
    """The class contains the meta data required by a pass. It is the
    container of information needed by running an optimization or analysis.
    This class can be extended by adding new members when more meta data is
    needed.

    Parameters
    ----------
    opt_level : int
        The optimization level of this pass.

    name : str
        The pass name.

    required : List[str]
        The list of passes that are required by a certain pass.
    """

    def __init__(self, opt_level, name, required=None):
        self.__init_handle_by_constructor__(
            _transform.PassInfo, opt_level, name, required)


@register_relay_node
class PassContext(RelayNode):
    """The basis where a Relay optimization/analysis runs on.
    Each pass context contains a number of auxiliary information that is used
    to help an optimization pass. Such information includes the error reporter
    to record the errors of during the optimization, etc.

    opt_level : Optional[int]
        The optimization level of this pass.

    fallback_device : Optional[Union[int, str, TVMContext]]
        The fallback device type. It is also used as the default device for
        operators that are not annotated during heterogeneous execution.

    required_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of passes that are required by a certain pass.

    disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of passes that are disabled.
    """
    def __init__(self,
                 opt_level=2,
                 fallback_device=_nd.cpu(),
                 required_pass=None,
                 disabled_pass=None):
        if isinstance(fallback_device, str):
            fallback_device = _nd.context(fallback_device).device_type
        elif isinstance(fallback_device, TVMContext):
            fallback_device = fallback_device.device_type
        if not isinstance(fallback_device, int):
            raise TypeError("required_pass is expected to be the type of " +
                            "int/str/TVMContext.")

        required = list(required_pass) if required_pass else []
        if not isinstance(required, (list, tuple)):
            raise TypeError("required_pass is expected to be the type of " +
                            "list/tuple/set.")

        disabled = list(disabled_pass) if disabled_pass else []
        if not isinstance(disabled, (list, tuple)):
            raise TypeError("disabled_pass is expected to be the type of " +
                            "list/tuple/set.")

        self.__init_handle_by_constructor__(_transform.PassContext, opt_level,
                                            fallback_device, required,
                                            disabled)

    def __enter__(self):
        _transform.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace):
        _transform.ExitPassContext(self)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _transform.GetCurrentPassContext()


def build_config(opt_level=2,
                 fallback_device=_nd.cpu(),
                 required_pass=None,
                 disabled_pass=None):
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
                "CombineParallelDense": 4
            }

    fallback_device : int, str, or tvm.TVMContext, optional
        The fallback device. It is also used as the default device for
        operators without specified device during heterogeneous execution.

    required_pass: set of str, optional
        Optimization passes that are required regardless of optimization level.

    disabled_pass: set of str, optional
        Optimization passes to be disabled during optimization.

    Returns
    -------
    pass_context: PassContext
        The pass context for optimizations.
    """
    return PassContext(opt_level, fallback_device, required_pass,
                       disabled_pass)


@register_relay_node
class Pass(RelayNode):
    """The base class of all passes. All methods here are just simple wrappers
    that are implemented in the backend. They are defined for users to
    conveniently interact with the base class.
    """

    @property
    def info(self):
        """Get the pass meta."""
        return _transform.Info(self)

    def __call__(self, mod):
        """Execute the pass. Note that for sequential pass, the dependency among
        different passes will be resolved in the backend.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that a certain optimization is performed on.

        Returns
        -------
        mod : tvm.relay.Module
            The updated module after applying this pass.
        """
        return _transform.RunPass(self, mod)


@register_relay_node
class ModulePass(Pass):
    """A pass that works on tvm.relay.Module. Users don't need to interact with
    this class directly. Instead, a module pass should be created through
    `module_pass`, because the design of the `module_pass` API is flexible
    enough to handle the creation of a module pass in different manners. In
    addition, all members of a module pass can be accessed from the base class.
    The same rule applies to FunctionPass as well.
    """


@register_relay_node
class FunctionPass(Pass):
    """A pass that works on each tvm.relay.Function in a module. A function
    pass class should be created through `function_pass`.
    """


@register_relay_node
class Sequential(Pass):
    """A pass that works on a sequence of pass objects. Multiple passes can be
    executed sequentially using this class.

    Some typical usage of the sequential pass are:
    1. Users provide a list of passes for optimization.
    2. Only an optimization level is provided so that the backend system has
       to glob all passes at this level and below to perform the optimizations.
    Note that users can also provide a series of passes that they don't want to
    apply when running a sequential pass. Pass dependency will be resolved in
    the backend as well.

    Parameters
    ----------
    passes : Optional[List[Pass]]
        A sequence of passes candidate for optimization.

    opt_level : Optional[int]
        The optimization level of this sequential pass.

    name : Optional[str]
        The name of the sequential pass.

    required : Optional[List[str]]
        The list of passes that the sequential pass is dependent on.
    """

    def __init__(self,
                 passes=None,
                 opt_level=2,
                 name="sequential",
                 required=None):
        passes = passes if passes else []
        if not isinstance(passes, (list, tuple)):
            raise TypeError("passes must be a list of Pass objects.")

        required = required if required else []
        if not isinstance(required, (list, tuple)):
            raise TypeError("Required is expected to be the type of list/tuple.")

        self.__init_handle_by_constructor__(_transform.Sequential,
                                            passes, opt_level, name, required)


def InferType():
    """Infer the type of an expr.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered type inference pass.
    """
    return _transform.InferType()


def FoldScaleAxis():
    """Fold the scaling of axis into weights of conv2d/dense. This pass will
    invoke both forward and backward scale folding.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass to fold expressions.

    Note
    ----
    Internally, we will call backward_fold_scale_axis before using
    forward_fold_scale_axis as backward folding targets the common conv->bn
    pattern.
    """
    return _transform.FoldScaleAxis()


def BackwardFoldScaleAxis():
    """Backward fold axis scaling into weights of conv2d/dense.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass to backward fold expressions.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis as backward folding targets the common
    conv->bn pattern.
    """
    return _transform.BackwardFoldScaleAxis()

def RemoveUnusedFunctions(entry_functions=None):
    """Remove unused global relay functions in a relay module.

    Parameters
    ----------
    entry_functions: list[string]
        The set of entry functions to start from.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass to remove unused functions.
    """
    if entry_functions is None:
        entry_functions = ['main']
    return _transform.RemoveUnusedFunctions(entry_functions)

def ForwardFoldScaleAxis():
    """Fold the scaling of axis into weights of conv2d/dense.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass to forward fold expressions.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis, as backward folding targets the
    common conv->bn pattern.
    """
    return _transform.ForwardFoldScaleAxis()


def SimplifyInference():
    """Simplify the data-flow graph for inference phase. An simplified expression
    which is semantically equal to the input expression will be returned.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered to perform operator simplification.
    """
    return _transform.SimplifyInference()


def CanonicalizeOps():
    """Canonicalize special operators to basic operators.
    This can simplify followed analysis, e.g. expanding bias_add to
    expand_dims and broadcast_add.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass performing the canonicalization.
    """
    return _transform.CanonicalizeOps()


def DeadCodeElimination(inline_once=False):
    """Remove expressions that do not have any users (dead code).

    Parameters
    ----------
    inline_once: Optional[Bool]
        Whether to inline binding that occurs only once.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass that eliminates the dead code in a Relay program.
    """
    return _transform.DeadCodeElimination(inline_once)


def FoldConstant():
    """Fold the constant expressions in a Relay program.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass for constant folding.
    """
    return _transform.FoldConstant()


def FuseOps(fuse_opt_level=-1):
    """Fuse operators in an expr to a larger operator according to some rules.

    Parameters
    ----------
    fuse_opt_level : int
        The level of fuse optimization. -1 indicates that the level will be
        inferred from pass context.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass for operator fusion.
    """
    return _transform.FuseOps(fuse_opt_level)


def CombineParallelConv2D(min_num_branches=3):
    """Combine multiple conv2d operators into one.

    Parameters
    ----------
    min_num_branches : int
        The minimum number of required parallel branches for performing this
        optimization.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass that combines parallel conv2d operators.
    """
    return _transform.CombineParallelConv2D(min_num_branches)


def CombineParallelDense(min_num_branches=3):
    """Combine multiple dense operators into one. For example:

                data
          /              \
     dense (2,2)         dense (2,2)
         |                 |
    elemwise/bcast (2,2)  elemwise/bcast (2,2)

    Would become:

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
    ret: tvm.relay.Pass
        The registered pass that combines parallel dense operators.
    """
    return _transform.CombineParallelDense(min_num_branches)


def AlterOpLayout():
    """Alternate the layouts of operators or replace primitive operators with
    other expressions.
    This pass can be used for computing convolution in custom layouts or
    other general weight pre-transformation.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that alters the layout of operators.
    """
    return _transform.AlterOpLayout()


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
    ret : tvm.relay.Pass
        The registered pass that rewrites an expr.
    """
    return _transform.Legalize(legalize_map_attr_name)


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
    ret: tvm.relay.Pass
        The registered pass that rewrites an expression with annotated
        `on_device` operators.
    """
    return _transform.RewriteDeviceAnnotation(fallback_device)


def ToANormalForm():
    """Turn Graph Normal Form expression into A Normal Form Expression.
    The scope of the root expression is the global scope.
    The scope of any non root expression is the least common ancestor of all it's scope.
    Values are ordered by post-DFS order in each scope.

    Returns
    -------
    ret: Union[tvm.relay.Pass, tvm.relay.Expr]
        The registered pass that transforms an expression into A Normal Form.
    """
    return _transform.ToANormalForm()


def ToCPS(expr, mod=None):
    """
    Turn expression into continuation passing style(CPS).

    Every intermediate compute will be passed to a continuation.

    Returns
    -------
    result: tvm.relay.Pass
        The registered pass that transforms an expression into CPS.
    """
    return _transform.to_cps(expr, mod)


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
    ret: tvm.relay.Pass
        The registered pass that eta expands an expression.
    """
    return _transform.EtaExpand(expand_constructor, expand_global_var)


def ToGraphNormalForm():
    """Turn a Relay program in A Normal Form into Graph Normal Form

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that transforms an expression into Graph Normal Form.
    """
    return _transform.ToGraphNormalForm()


def EliminateCommonSubexpr(fskip=None):
    """Eliminate common subexpressions.

    Parameters
    ----------
    fskip: Callable
        The callback function that decides whether an expression should be
        skipped.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that eliminates common subexpressions.
    """
    return _transform.EliminateCommonSubexpr(fskip)


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
    ret: tvm.relay.Pass
        The registered pass that performs partial evaluation on an expression.
    """
    return _transform.PartialEvaluate()


def CanonicalizeCast():
    """
    Canonicalize cast expressions to make operator fusion more efficient.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that canonicalizes cast expression.
    """
    return _transform.CanonicalizeCast()


def LambdaLift():
    """
    Lift the closure to global function.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that lifts the lambda function.
    """
    return _transform.LambdaLift()


def PrintIR(show_meta_data=True):
    """
    Print the IR for a module to help debugging.

    Parameters
    ----------
    show_meta_data : bool
        A boolean flag to indicate if meta data should be printed.

    Returns
    -------
    ret : tvm.relay.Pass
        The registered pass that prints the module IR.
    """
    return _transform.PrintIR(show_meta_data)


def gradient(expr, mod=None, mode='higher_order'):
    """
    Transform the input function,
    returning a function that calculate the original result,
    paired with gradient of the input.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, which is a Function or a GlobalVar.

    mod : Optional[tvm.relay.Module]

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
        return _transform.first_order_gradient(expr, mod)
    if mode == 'higher_order':
        return _transform.gradient(expr, mod)
    raise Exception('unknown mode')


def to_cps(func, mod=None):
    """
    Turn expression into CPS expression.

    Every intermediate compute will be passed to a continuation.

    Parameters
    ----------
    func: tvm.relay.Function
        The input function.

    mod: Optional[tvm.relay.Module]
        The global module.

    Returns
    -------
    result: tvm.relay.Function
      The output function.
    """
    return _transform.to_cps(func, mod)


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
    return _transform.un_cps(func)


def _wrap_class_module_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""
    class PyModulePass(ModulePass):
        """Internal wrapper class to create a class instance."""
        def __init__(self, *args, **kwargs):
            # initialize handle in cass pass_cls creation failed.fg
            self.handle = None
            inst = pass_cls(*args, **kwargs)
            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(mod, ctx):
                return inst.transform_module(mod, ctx)
            self.__init_handle_by_constructor__(
                _transform.MakeModulePass, _pass_func, pass_info)
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyModulePass.__init__, pass_cls.__init__)
    PyModulePass.__name__ = pass_cls.__name__
    PyModulePass.__doc__ = pass_cls.__doc__
    PyModulePass.__module__ = pass_cls.__module__
    return PyModulePass


def module_pass(pass_func=None, opt_level=None, name=None, required=None):
    """Decorate a module pass.

    This function returns a callback when pass_func is provided.
    Otherwise, it serves a decorator function.

    pass_func can also be a class type with a method transform_module.
    This function will create a decorated ModulePass using transform_module
    as the pass function.

    Parameters
    ----------
    pass_func : Optional[Callable[(Module, PassContext) ->Module]]
        The transformation function or class.

    opt_level : int
        The optimization level of this module pass.

    name : Optional[str]
        The name of the module pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the module pass is dependent on.

    Returns
    -------
    create_module_pass : Union[Callable, ModulePass]
        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new ModulePass will be returned when we decorate a pass function.
        A new ModulePass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a module pass class.

    .. code-block:: python

        @relay.transform.module_pass
        class CustomPipeline:
            def __init__(self, enable_fold):
                self.enable_fold = enable_fold
                self.cse = relay.transform.EliminateCommonSubexpr()
                self.const_fold = relay.transform.FoldConstant()

            def transform_module(self, mod, ctx):
                mod = self.cse(mod, ctx)
                if self.enable_fold:
                    mod = self.const_fold(mod, ctx)
                return mod

        # create an instance of customized pipeline
        pipeline = CustomPipeline(enable_fold=False)
        assert isinstance(pipeline, transform.ModulePass)
        # run the pipeline.
        output_module = pipeline(input_module)

    The following code creates a module pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relay.transform.module_pass(opt_level=2)
        def transform(mod, ctx):
            tp = relay.TensorType((10,), "float32")
            x = relay.var("x", tp)
            gv = relay.GlobalVar("var")
            func = relay.Function([x], relay.abs(x))
            new_mod = relay.Module({gv: func})
            new_mod.update(mod)
            return new_mod

        module_pass = transform
        assert isinstance(module_pass, transform.ModulePass)
        assert module_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = module_pass(m)
        # Now a function abs should be added to the module m.
    """
    if opt_level is None:
        raise ValueError("Please provide opt_level for the module pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " +
                        "list/tuple.")

    def create_module_pass(pass_arg):
        """Internal function that creates a module pass"""
        fname = name if name else pass_arg.__name__
        info = PassInfo(opt_level, fname, required)
        if inspect.isclass(pass_arg):
            return _wrap_class_module_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for Module pass")
        return _transform.MakeModulePass(pass_arg, info)

    if pass_func:
        return create_module_pass(pass_func)
    return create_module_pass


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
                _transform.MakeFunctionPass, _pass_func, pass_info)
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
        info = PassInfo(opt_level, fname, required)
        if inspect.isclass(pass_arg):
            return _wrap_class_function_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for Module pass")
        return _transform.MakeFunctionPass(pass_arg, info)

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
                else:
                    return var
        return ChangeBatchMutator().visit(func)
