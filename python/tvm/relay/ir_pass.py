# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
This file contains:
1. The set of passes for Relay, which exposes an interface for configuring the
   passes and scripting them in Python.

2. The pass manager for Relay which exposes different granularity of interfaces
   for users to implement and use passes more conveniently.
"""
import types

from . import _ir_pass
from . import _make
from .expr import Expr
from .ty import Type
from .base import RelayNode, register_relay_node
from .module import Module


@register_relay_node
class PassContext(RelayNode):
    """The basis where a Relay optimization/analysis runs on.
    Each pass context contains a number of auxiliary information that is used
    to help an optimization pass. Such information includes the error reporter
    to record the errors of during the performing the optimization, etc.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ir_pass.PassContext)


@register_relay_node
class Pass(RelayNode):
    """The base class of all passes. This class is designed as a pure virtual
    class that will be implemented by the subclasses.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.
    """

    def set_pass_context(self, pass_ctx):
        """Setup the pass context for analysis and optimizations. This context
        could be shared by different passes for sequential passes.

        Parameters
        ----------
        pass_ctx : PassContext
            The context that is used to help perform a certain pass or a series
            of passes.

        Returns
        -------
        pass : Pass
            The updated pass.
        """
        if not isinstance(pass_ctx, PassContext):
            raise TypeError("pass_ctx is expected to be the PassContext type")
        return _ir_pass.SetContext(self, pass_ctx)

    def __call__(self, mod):
        """Execute the pass. It is an abstract function that will be
        implemented by subclasses.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that a certain optimization is performed on.

        Returns
        -------
        mod : tvm.relay.Module
            The updated module after applying this pass.
        """
        raise NotImplementedError("Pure virtual function is not implemented.")


@register_relay_node
class ModulePass(Pass):
    """A pass that works on tvm.relay.Module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_func : Callable[PassContext: tvm.relay.Module -> tvm.relay.Module]
        The curried callback that sketches a certain optimization.
    """

    def __init__(self, name, opt_level, pass_func):
        self.__init_handle_by_constructor__(_ir_pass.CreateModulePass, name,
                                            opt_level, pass_func)

    def __call__(self, mod):
        """Execute a module pass.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the module pass is executed on.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _ir_pass.RunModulePass(self, mod)


@register_relay_node
class FunctionPass(Pass):
    """A pass that works on each tvm.relay.Function in a module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    pass_func : Callable[PassContext: tvm.relay.Function -> tvm.relay.Function]
        The curried callback that sketches a certain optimization.
    """

    def __init__(self, name, opt_level, pass_func):
        self.__init_handle_by_constructor__(_ir_pass.CreateFunctionPass, name,
                                            opt_level, pass_func)

    def __call__(self, mod):
        """Execute a function pass.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the function pass is executed on.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _ir_pass.RunFunctionPass(self, mod)


@register_relay_node
class SequentialPass(Pass):
    """A pass that works on each tvm.relay.Function in a module.

    Parameters
    ----------
    name : str
        The pass name.

    opt_level : int
        The optimization level of this pass.

    passes : List[Pass]
        The pass candidates to be executed.

    disabled : Optional[List[str]]
        The list of passes that are disabled.
    """

    def __init__(self, name, opt_level, passes, disabled=None):
        disabled = disabled if disabled else []
        if not isinstance(disabled, (list, tuple)):
            raise TypeError("disabled must be a list or tuple of pass names")
        self.__init_handle_by_constructor__(_ir_pass.CreateSequentialPass,
                                            name, opt_level, passes, disabled)

    def __call__(self, mod):
        """Execute a sequence of passes.

        Parameters
        ----------
        mod : tvm.relay.Module
            The module that the function pass is executed on.

        Returns
        -------
        ret : tvm.relay.Module
            The updated module.
        """
        return _ir_pass.RunSequentialPass(self, mod)


def create_module_pass(pass_name, opt_level, pass_func):
    """Create a module pass using a defined optimization function from Python.

    Parameters
    ----------
    pass_name : str
        The name of the pass.

    opt_level : int
        The optimization level of this pass.

    pass_func : Optional[Callable[PassContext: Module/Function ->
                Module/Function]]
        The implemented optimization pass.

    Returns
    -------
    ret : Pass
        A module level pass built through pass_func.
    """
    if not isinstance(pass_func, (types.FunctionType, types.LambdaType)):
        raise TypeError("pass_func must be a callable for Module pass")

    return _ir_pass.CreateModulePass(pass_name, opt_level, pass_func)


def create_function_pass(pass_name, opt_level, pass_func):
    """Create a function pass using a defined optimization function from
    Python.

    Parameters
    ----------
    pass_name : str
        The name of the pass.

    opt_level : int
        The optimization level of this pass.

    pass_func : Optional[Callable[PassContext: Module/Function ->
                Module/Function]]
        The implemented optimization pass.

    Returns
    -------
    ret : Pass
        A function level pass built through pass_func.
    """
    if not isinstance(pass_func, (types.FunctionType, types.LambdaType)):
        raise TypeError("pass_func must be a callable for Module pass")

    return _ir_pass.CreateFunctionPass(pass_name, opt_level, pass_func)


def create_sequential_pass(pass_name, opt_level, sequential_passes,
                           disabled=None):
    """Create a sequential pass using a defined optimization function from
    Python.

    Parameters
    ----------
    pass_name : str
        The name of the pass.

    opt_level : int
        The optimization level of this pass.

    sequential_passes : Optional[List[Pass]]
        A sequence of passes candidate for optimization.

    disabled : Optional[List[str]]
        A list of disabled passes.

    Returns
    -------
    ret : Pass
        A sequential pass built through pass_func.
    """
    if not isinstance(sequential_passes, (list, tuple)):
        raise TypeError("sequential_passes must be a list of Pass objects.")

    disabled = disabled if disabled else []
    if not isinstance(disabled, (list, tuple)):
        raise TypeError("disabled must be a list or tuple of pass names")

    return _ir_pass.CreateSequentialPass(pass_name, opt_level,
                                         sequential_passes, disabled)


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fvisit : function
        The visitor function to be applied.
    """
    return _ir_pass.post_order_visit(expr, fvisit)

def infer_type(expr, mod=None):
    """Infer the type of expr under the context of mod.

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression.

    mod: Optional[tvm.relay.Module]
        The global module.

    Returns
    -------
    checked_expr : tvm.relay.Expr
        The checked expression.
    """
    return _ir_pass.infer_type(expr, mod)


def backward_fold_scale_axis(expr):
    """Backward fold axis scaling into weights of conv2d/dense.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, we expect that expr's types
        should be fully inferred by infer_type.

    Returns
    -------
    folded_expr : tvm.relay.Expr
        The folded expression after transformation.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis.
    As backward folding targets common conv-bn pattern.
    """
    return _ir_pass.backward_fold_scale_axis(expr)


def forward_fold_scale_axis(expr):
    """Fold the scaling of axis into weights of conv2d/dense.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, we expect that expr's types
        should be fully inferred by infer_type.

    Returns
    -------
    folded_expr : tvm.relay.Expr
        The folded expression after transformation.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis.
    As backward folding targets common conv-bn pattern.
    """
    return _ir_pass.forward_fold_scale_axis(expr)


def well_formed(expr):
    """Check that each Var is only bound once (well formed).

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression

    Returns
    -------
    well_form : bool
        Whether the input expression is well formed
    """
    return _ir_pass.well_formed(expr)


def check_kind(t, mod=None):
    """Check that the type is well kinded and return the kind.
    For example, this mean type cannot has tensor of tensor, or is a tuple type of 2 shapes.

    Parameters
    ----------
    t : tvm.relay.Type
        The type to check

    mod : Optional[tvm.relay.Module]
        The global module.

    Returns
    -------
    kind : Kind
        the kind of t

    Examples
    --------
    .. code:: python

        assert check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Shape)])) == Shape
        assert check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Type)])) == Type
    """
    if mod is not None:
        return _ir_pass.check_kind(t, mod)
    else:
        return _ir_pass.check_kind(t)


def free_vars(expr):
    """Get free Vars from expression expr in Post DFS order.

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of free variables in post DFS order.

    Note
    ----
    The fact that Vars are post-DFS ordred are useful in
    neural networks: usually this means weights of previous
    are ordered first.
    """
    return _ir_pass.free_vars(expr)


def bound_vars(expr):
    """Get bound vars from expression expr in post-DFS order.

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of bound variables in post-DFS order.
    """
    return _ir_pass.bound_vars(expr)


def all_vars(expr):
    """Get all vars from expression expr in post-DFS order.

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of all variables in post-DFS order.
    """
    return _ir_pass.all_vars(expr)


def free_type_vars(expr, mod=None):
    """Get free type variables from expression/type e

    Parameters
    ----------
    expr: Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type
    mod: tvm.relay.Module, optional
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of free type variables in post-DFS order
    """
    use_mod = mod if mod is not None else Module()
    return _ir_pass.free_type_vars(expr, use_mod)


def bound_type_vars(expr, mod=None):
    """Get bound type variables from expression/type e

    Parameters
    ----------
    expr: Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type
    mod: tvm.relay.Module, optional
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of bound type variables in post-DFS order
    """
    use_mod = mod if mod is not None else Module()
    return _ir_pass.bound_type_vars(expr, use_mod)


def all_type_vars(expr, mod=None):
    """Get all type variables from expression/type e

    Parameters
    ----------
    expr: Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type
    mod: tvm.relay.Module, optional
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of all type variables in post-DFS order
    """
    use_mod = mod if mod is not None else Module()
    return _ir_pass.all_type_vars(expr, use_mod)


def simplify_inference(expr):
    """ Simplify the data-flow graph for inference phase.

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
        An expression which is semantically equal to the input expression,
        but with some simplification
    """
    return _ir_pass.simplify_inference(expr)


def canonicalize_ops(expr):
    """ Canonicalize special operators to basic operators.
    This can simplify latter analysis. (e.g. Expand bias_add to expand_dims and broadcast_add.)

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
        An expression without bias_add
    """
    return _ir_pass.canonicalize_ops(expr)


def dead_code_elimination(expr):
    """ Remove expressions which does not effect the program result (dead code).

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
        An expression which is semantically equal to the input expression,
        but with dead code removed.
    """
    return _ir_pass.dead_code_elimination(expr)


def alpha_equal(lhs, rhs):
    """Compare two Relay expr for structural equivalence (alpha equivalence).

    Parameters
    ----------
    lhs: tvm.relay.Expr
        One of the input Expression.

    rhs: tvm.relay.Expr
        One of the input Expression.

    Returns
    -------
    result: bool
        True iff lhs is alpha equal to rhs.
    """
    return bool(_make._alpha_equal(lhs, rhs))


def graph_equal(lhs, rhs):
    """Compare two Relay expr for data-flow equivalence.
    The difference between this and alpha-equality is that
    variables are not expected to match between lhs and rhs;
    they are treated as sources and are mapped between each other.

    Parameters
    ----------
    lhs: tvm.relay.Expr
      One of the input Expression.

    rhs: tvm.relay.Expr
      One of the input Expression.

    Returns
    -------
    result: bool
      True iff lhs is data-flow equivalent to rhs.
    """
    return bool(_make._graph_equal(lhs, rhs))


def structural_hash(value):
    """Hash a Relay expression structurally.

    Parameters
    ----------
    expr: tvm.relay.Expr or tvm.relay.Type
      The expression to hash.

    Returns
    -------
    result: int
      The hash value
    """
    if isinstance(value, Expr):
        return int(_ir_pass._expr_hash(value))
    elif isinstance(value, Type):
        return int(_ir_pass._type_hash(value))
    else:
        msg = ("found value of type {0} expected" +
               "relay.Expr or relay.Type").format(type(value))
        raise TypeError(msg)


def fold_constant(expr):
    """Fold the constant expression in expr.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        The transformed expression.
    """
    return _ir_pass.FoldConstant(expr)


def fuse_ops(expr, opt_level=1):
    """Fuse operators in expr together.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    opt_level : int
        The level of fuse optimization.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        Transformed expression, containing fused result.
    """
    return _ir_pass.FuseOps(expr, opt_level)


def combine_parallel_conv2d(expr):
    """Fold multiple conv2d into one.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        Transformed expression
    """
    return _ir_pass.CombineParallelConv2D(expr)


def alter_op_layout(expr):
    """Alternate the layouts of operators or replace primitive operators with
    other expressions.
    This pass can be used for computing convolution in custom layouts or
    other general weight pre-transformation.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        Transformed expression with alternated layout.
    """
    return _ir_pass.AlterOpLayout(expr)


def rewrite_annotated_ops(expr, fallback_device):
    """Rewrite the annotated program where annotation operators, e.g.
    `on_deivce`, mark which device an expression should be scheduled to.
    This pass helps heterogeneous execution where different operators may need
    to be allocated on various devices.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fallback_device : int
        The fallback device type. It is also used as the default device for
        operators with no annotated device.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        Transformed expression with cross device data copy operators.
    """
    return _ir_pass.RewriteDeviceAnnotation(expr, fallback_device)


def collect_device_info(expr):
    """Collect the device allocation map for the given expression. The device
    ids are propagated from the `device_copy` operators.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    ret : Dict[tvm.relay.expr, int]
        A dictionary mapping tvm.relay.Expr to device type.
    """
    return _ir_pass.CollectDeviceInfo(expr)


def collect_device_annotation_ops(expr):
    """Collect the device annotation ops for the given expression.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    ret : Dict[tvm.relay.expr, int]
        A dictionary mapping tvm.relay.Expr to device type where the keys are
        annotation expressions.
    """
    return _ir_pass.CollectDeviceAnnotationOps(expr)


def to_a_normal_form(expr, mod=None):
    """
    Turn Graph Normal Form expression into A Normal Form Expression.

    The scope of the root expression is the global scope.

    The scope of any non root expression is the least common ancestor of all it's scope.

    Values are ordered by post-DFS order in each scope.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    mod: Optional[tvm.relay.Module]
        The global module.

    Returns
    -------
    expr: tvm.relay.Expr
      The output expression.
    """
    return _ir_pass.to_a_normal_form(expr, mod)


def to_graph_normal_form(expr):
    """Turn A Normal Form expression into Graph Normal Form expression
    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression
    Returns
    -------
    expr : tvm.relay.Expr
      The output expression
    """
    return _ir_pass.to_graph_normal_form(expr)


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
        'first_order' only work on first order code, but will not produce reference nor closure.
        'higher_order' work on all code using reference and closure.

    Returns
    -------
    expr : tvm.relay.Expr
      The transformed expression.
    """
    if mode == 'first_order':
        return _ir_pass.first_order_gradient(expr, mod)
    elif mode == 'higher_order':
        return _ir_pass.gradient(expr, mod)
    else:
        raise Exception('unknown mode')



def get_total_mac_number(expr):
    """
    Count the number of MACs (multiply-accumulate) of a model

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    ret : int64
      The number of MACs (multiply-accumulate) of a model
    """
    return _ir_pass.GetTotalMacNumber(expr)


def eliminate_common_subexpr(expr, fskip=None):
    """
    Eliminate common subexpressions.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fskip: function
        The callback function that decides whether an expression should be skipped.

    Returns
    -------
    expr : tvm.relay.Expr
      The output expression.
    """
    return _ir_pass.eliminate_common_subexpr(expr, fskip)
