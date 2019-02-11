# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""Adds certain standard global functions and ADT definitions to the module."""
from .ty import GlobalTypeVar, TypeVar, FuncType
from .expr import Var, Function, GlobalVar
from .adt import Constructor, TypeData, Clause, Match
from .adt import PatternConstructor, PatternVar, PatternWildcard

class Prelude:
    """Contains standard definitions."""

    def define_list_adt(self):
        """Defines a LISP-style list ADT. An empty list is
        represented by nil(). A member x can be appended to the
        front of a list l via the constructor cons(x, l)."""
        self.l = GlobalTypeVar("list")
        a = TypeVar("a")
        self.nil = Constructor("nil", [], self.l)
        self.cons = Constructor("cons", [a, self.l(a)], self.l)
        self.mod[self.l] = TypeData(self.l, [a], [self.nil, self.cons])

    def define_list_map(self):
        """Defines a function for mapping a function over a list's
        elements. That is, map(f, l) returns a new list where
        the ith member is f applied to the ith member of l.

        map(f, l) : fn<a, b>(fn(a) -> b, list[a]) -> list[b]
        """
        self.map = GlobalVar("map")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a], b))
        x = Var("x", self.l(a))
        y = Var("y")
        z = Var("z")
        nil_case = Clause(PatternConstructor(self.nil), self.nil())
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]),
                           self.cons(f(y), self.map(f, z)))
        self.mod[self.map] = Function([f, x], Match(x, [nil_case, cons_case]), None, [a, b])

    def define_list_foldl(self):
        """Defines a left-way fold over a list.

        foldl(f, z, l) : fn<a, b>(fn(b, a) -> b, b, list[a]) -> b

        foldl(f, z, cons(a1, cons(a2, cons(a3, cons(..., nil)))))
        evaluates to f(...f(f(f(z, a1), a2), a3)...)
        """
        self.foldl = GlobalVar("foldl")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a, b], a))
        av = Var("av", a)
        bv = Var("bv", self.l(b))
        y = Var("y")
        z = Var("z")
        nil_case = Clause(PatternConstructor(self.nil), av)
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]),
                           self.foldl(f, f(av, y), z))
        self.mod[self.foldl] = Function([f, av, bv],
                                        Match(bv, [nil_case, cons_case]), None, [a, b])

    def define_list_foldr(self):
        """Defines a right-way fold over a list.

        foldr(f, l, z) : fn<a, b>(fn(a, b) -> b, list[a], b) -> b

        foldr(f, cons(a1, cons(a2, cons(..., cons(an, nil)))), z)
        evalutes to f(a1, f(a2, f(..., f(an, z)))...)
        """
        self.foldr = GlobalVar("foldr")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a, b], b))
        av = Var("av", self.l(a))
        bv = Var("bv", b)
        y = Var("y")
        z = Var("z")
        nil_case = Clause(PatternConstructor(self.nil), bv)
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(y), PatternVar(z)]),
                           f(y, self.foldr(f, bv, z)))
        self.mod[self.foldr] = Function([f, bv, av],
                                        Match(av, [nil_case, cons_case]), None, [a, b])

    def define_optional_adt(self):
        """Defines an optional ADT, which can either contain some other
        type or nothing at all."""
        self.optional = GlobalTypeVar("optional")
        a = TypeVar("a")
        self.some = Constructor("some", [a], self.optional)
        self.none = Constructor("none", [], self.optional)
        self.mod[self.optional] = TypeData(self.optional, [a], [self.some, self.none])

    def define_nat_adt(self):
        """Defines a Peano (unary) natural number ADT.
        Zero is represented by z(). s(n) adds 1 to a nat n."""
        self.nat = GlobalTypeVar("nat")
        self.z = Constructor("z", [], self.nat)
        self.s = Constructor("s", [self.nat()], self.nat)
        self.mod[self.nat] = TypeData(self.nat, [], [self.z, self.s])

    def define_nat_double(self):
        """Defines a function that doubles a nat."""
        self.double = GlobalVar("double")
        x = Var("x", self.nat())
        y = Var("y")
        z_case = Clause(PatternConstructor(self.z), self.z())
        s_case = Clause(PatternConstructor(self.s, [PatternVar(y)]),
                        self.s(self.s(self.double(y))))
        self.mod[self.double] = Function([x], Match(x, [z_case, s_case]))

    def define_nat_add(self):
        """Defines a function that adds two nats."""
        self.add = GlobalVar("add")
        x = Var("x", self.nat())
        y = Var("y", self.nat())
        a = Var("a")
        z_case = Clause(PatternConstructor(self.z), y)
        s_case = Clause(PatternConstructor(self.s, [PatternVar(a)]),
                        self.s(self.add(a, y)))
        self.mod[self.add] = Function([x, y], Match(x, [z_case, s_case]))

    def define_list_sum(self):
        """Defines a function that computes the sum of a list of nats."""
        self.sum = GlobalVar("sum")
        a = Var("a", self.l(self.nat()))
        self.mod[self.sum] = Function([a], self.foldl(self.add, self.z(), a))

    def define_list_length(self):
        """Defines a function that returns the length of a list as a nat"""
        self.length = GlobalVar("length")
        a = TypeVar("a")
        x = Var("x", self.l(a))
        y = Var("y")
        nil_case = Clause(PatternConstructor(self.nil), self.z())
        cons_case = Clause(PatternConstructor(self.cons, [PatternWildcard(), PatternVar(y)]),
                           self.s(self.length(y)))
        self.mod[self.length] = Function([x],
                                         Match(x, [nil_case, cons_case]), None, [a])

    def define_tree_adt(self):
        """Defines a tree ADT. A tree can contain any type.
        It has only one constructor, rose(x, l), where x is the content
        of that point of the tree and l is a list of more trees of the
        same type. A leaf is thus rose(x, nil()).
        """
        self.tree = GlobalTypeVar("tree")
        a = TypeVar("a")
        self.rose = Constructor("rose", [a, self.l(self.tree(a))], self.tree)
        self.mod[self.tree] = TypeData(self.tree, [a], [self.rose])

    def define_tree_map(self):
        """Defines a function that maps over a tree. The function
        is applied to each subtree's contents.

        Signature: fn<a, b>(f : fn(a) -> b, t : tree[a]) -> tree[b]
        """
        self.tmap = GlobalVar("tmap")
        a = TypeVar("a")
        b = TypeVar("b")
        t = Var("t", self.tree(a))
        f = Var("f", FuncType([a], b))
        x = Var("x", self.tree(a))
        y = Var("y")
        z = Var("z")
        rose_case = Clause(PatternConstructor(self.rose, [PatternVar(y), PatternVar(z)]),
                           self.rose(f(y), self.map(Function([x], self.tmap(f, x)), z)))
        self.mod[self.tmap] = Function([f, t],
                                       Match(t, [rose_case]), self.tree(b), [a, b])

    def define_tree_size(self):
        """Defines a function that computes the size of a tree as a nat."""
        self.size = GlobalVar("size")
        a = TypeVar("a")
        t = Var("t", self.tree(a))
        x = Var("x", self.tree(a))
        z = Var("z")
        rose_case = Clause(PatternConstructor(self.rose, [PatternWildcard(), PatternVar(z)]),
                           self.s(self.sum(self.map(Function([x], self.size(x)), z))))
        self.mod[self.size] = Function([t],
                                       Match(t, [rose_case]), self.nat(), [a])

    def __init__(self, mod):
        self.mod = mod
        self.define_list_adt()
        self.define_list_map()
        self.define_list_foldl()
        self.define_list_foldr()

        self.define_optional_adt()

        self.define_nat_adt()
        self.define_nat_double()
        self.define_nat_add()
        self.define_list_length()
        self.define_list_sum()

        self.define_tree_adt()
        self.define_tree_map()
        self.define_tree_size()
