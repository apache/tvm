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

        map(f, l) : fun<a, b>(fun(a) -> b, list<a>) -> list<b>
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

        foldl(f, z, l) : fun<a, b>(fun(b, a) -> b, b, list<a>) -> b

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

        foldr(f, l, z) : fun<a, b>(fun(a, b) -> b, list<a>, b) -> b

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

    def __init__(self, mod):
        self.mod = mod
        self.define_list_adt()
        self.define_list_map()
        self.define_list_foldl()
        self.define_list_foldr()
