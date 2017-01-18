# pylint: disable=protected-access, no-member
"""Collection structure in the high level DSL."""
from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import _function_internal
from . import tensor as _tensor

@register_node
class Split(NodeBase):
    """Split operation on axis."""
    pass

@register_node
class Fuse(NodeBase):
    """Fuse operation on axis."""
    pass


@register_node
class Schedule(NodeBase):
    """Schedule for all the stages."""
    def __getitem__(self, k):
        if isinstance(k, _tensor.Tensor):
            k = k.op
        if not isinstance(k, _tensor.Operation):
            raise ValueError("Expect schedule key to be Tensor or Operation")
        if k not in self.stage_map:
            raise ValueError("Cannot find the operation %s in schedule" % (str(k)))
        return self.stage_map[k]

@register_node
class Stage(NodeBase):
    """A Stage represents schedule for one operation."""
    def split(self, parent, factor=None, outer=None):
        """Split the stage either by factor providing outer scope, or both

        Parameters
        ----------
        parent : IterVar
             The parent iter var.

        factor : Expr, optional
             The splitting factor

        outer : IterVar, optional
             The outer split variable

        Returns
        -------
        outer : IterVar
            The outer variable of iteration.

        inner : IterVar
            The inner variable of iteration.
        """
        if outer is not None:
            if outer.thread_tag == '':
                raise ValueError("split by outer must have special thread_tag")
            inner = _function_internal._StageSplitByOuter(self, parent, outer, factor)
        else:
            if factor is None:
                raise ValueError("either outer or factor need to be provided")
            outer, inner = _function_internal._StageSplitByFactor(self, parent, factor)
        return outer, inner

    def fuse(self, inner, outer):
        """Fuse inner and outer to a single iteration variable.

        Parameters
        ----------
        outer : IterVar
            The outer variable of iteration.

        inner : IterVar
            The inner variable of iteration.

        Returns
        -------
        inner : IterVar
            The fused variable of iteration.
        """
        return _function_internal._StageFuse(self, inner, outer)

    def set_scope(self, scope):
        """Set the thread scope of this stage

        Parameters
        ----------
        scope : str
            The thread scope of this stage
        """
        return _function_internal._StageSetScope(self, scope)

    def compute_at(self, parent, scope):
        """Attach the stage at parent's scope

        Parameters
        ----------
        parent : Stage
            The parent stage

        scope : IterVar
            The loop scope t be attached to.
        """
        _function_internal._StageComputeAt(self, parent, scope)

    def compute_inline(self):
        """Mark stage as inline

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _function_internal._StageComputeInline(self)

    def compute_root(self):
        """Attach the stage at parent, and mark it as root

        Parameters
        ----------
        parent : Stage
            The parent stage
        """
        _function_internal._StageComputeInline(self)

    def reorder(self, *args):
        """reorder the arguments in the specified order.

        Parameters
        ----------
        args : list of IterVar
            The order to be ordered
        """
        _function_internal._StageReorder(self, args)

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions

        The final loop order from outmost to inner most are
        [x_outer, y_outer, x_inner, y_inner]

        Parameters
        ----------
        x_parent : IterVar
            The original x dimension
        y_parent : IterVar
            The original y dimension
        x_factor : Expr
            The stride factor on x axis
        y_factor : Expr The stride factor on y axis

        Returns
        -------
        x_outer : IterVar
            Outer axis of x dimension
        y_outer : IterVar
            Outer axis of y dimension
        x_inner : IterVar
            Inner axis of x dimension
        p_y_inner : IterVar
            Inner axis of y dimension
        """
        x_outer, y_outer, x_inner, y_inner = _function_internal._StageTile(
            self, x_parent, y_parent, x_factor, y_factor)
        return x_outer, y_outer, x_inner, y_inner
