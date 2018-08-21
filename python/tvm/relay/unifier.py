"""The Python interface to Relay's UnionFind and TypeUnifier."""

from typing import Dict
from .ir import register_relay_node, NodeBase
from . import ir
from . import _unifier

@register_relay_node
class UnionFind(NodeBase):
    """Python API for UnionFind.

       The UnionFind maintains equality classes of type variables, the
       representative of an equality class may be a type (which can)
       contain type variables. The TypeUnifier uses this to build a
       unification procedure between types.
    """
    uf_map: Dict[ir.IncompleteType, ir.IncompleteType]

    def insert(self, var: ir.IncompleteType) -> None:
        """Insert a type variable into the union find.

           :param: var: The variable to be inserted.
        """
        return _unifier.UnionFind_insert(self, var)

    def unify(self, var: ir.IncompleteType, typ: ir.Type) -> None:
        """Unify a type variable with an arbitrary type.

           :param: var: A type variable to be unified.
           :param: typ: The type to be unified with.
        """
        return _unifier.UnionFind_unify(self, var, typ)

    def find(self, var: ir.IncompleteType) -> ir.IncompleteType:
        """Find the representative element of the type var.

           :param: var: The variable to lookup in the union find.
        """
        return _unifier.UnionFind_find(self, var)

@register_relay_node
class TypeUnifier(NodeBase):
    """Python API for the TypeUnifier."""
    #pylint: disable=invalid-name
    uf: UnionFind
    eq_map: Dict[ir.TypeParam, ir.TypeParam]

    def insert(self, var: ir.IncompleteType) -> None:
        return _unifier.TypeUnifier_insert(self, var)

    def unify(self, type1: ir.Type, type2: ir.Type) -> ir.Type:
        """Unify two types producing the unified type as a result.

           :param: type1: The first type to be unified.
           :param: type2: The second type to be unified.
           :returns: The unified type.
        """
        return _unifier.TypeUnifier_unify(self, type1, type2)

    def subst(self, type1: ir.Type) -> ir.Type:
        return _unifier.TypeUnifier_subst(self, type1)
