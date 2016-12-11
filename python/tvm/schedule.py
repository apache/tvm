"""Collection structure in the high level DSL."""
from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import _function_internal

@register_node
class Split(NodeBase):
    pass

@register_node
class Fuse(NodeBase):
    pass

@register_node
class Schedule(NodeBase):
    def split(self, parent, factor=None, outer=None):
        """Split the schedule either by factor providing outer scope, or both

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
            if outer.dom is None:
                raise ValueError("split by outer must have specified domain")
            inner = _function_internal._ScheduleSplitByOuter(self, parent, outer, factor)
        else:
            if factor is None:
                raise ValueError("either outer or factor need to be provided")
            outer, inner = _function_internal._ScheduleSplitByFactor(self, parent, factor)
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
        return _function_internal._ScheduleFuse(self, inner, outer)

    def compute_at(self, parent, scope):
        """Attach the schedule at parent's scope

        Parameters
        ----------
        parent : Schedule
            The parent schedule

        scope : IterVar
            The loop scope t be attached to.
        """
        _function_internal._ScheduleComputeAt(self, parent, scope)

    def compute_inline(self, parent):
        """Attach the schedule at parent, and mark it as inline

        Parameters
        ----------
        parent : Schedule
            The parent schedule
        """
        _function_internal._ScheduleComputeInline(self, parent)

    def compute_root(self, parent):
        """Attach the schedule at parent, and mark it as root

        Parameters
        ----------
        parent : Schedule
            The parent schedule
        """
        _function_internal._ScheduleComputeInline(self, parent)

    def reorder(self, *args):
        """reorder the arguments in the specified order.

        Parameters
        ----------
        args : list of IterVar
            The order to be ordered
        """
        _function_internal._ScheduleReorder(self, args)

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        x_outer, y_outer, x_inner, y_inner = _function_internal._ScheduleTile(
            self, x_parent, y_parent, x_factor, y_factor)
        return x_outer, y_outer, x_inner, y_inner
