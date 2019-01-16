# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""Include some preloaded term/type definitions."""
from .ty import GlobalTypeVar, TypeVar, FuncType
from .expr import Var, Function, GlobalVar
from .adt import Constructor, TypeData, Clause, Match
from .adt import PatternConstructor, PatternVar, PatternWildcard

class Prelude:
    """Contain standard definitions."""
    def __init__(self, mod):
        self.mod = mod
        self.nat = GlobalTypeVar("nat")
        self.z = Constructor("z", [], self.nat)
        self.s = Constructor("s", [self.nat()], self.nat)
        mod[self.nat] = TypeData(self.nat, [], [self.z, self.s])

        self.double = GlobalVar("double")
        x = Var("x", self.nat())
        y = Var("y")
        z_case = Clause(PatternConstructor(self.z), self.z())
        s_case = Clause(PatternConstructor(self.s, [PatternVar(y)]), self.s(self.s(self.double(y))))
        mod[self.double] = Function([x], Match(x, [z_case, s_case]))

        self.add = GlobalVar("add")
        x = Var("x", self.nat())
        y = Var("y", self.nat())
        a = Var("a")
        z_case = Clause(PatternConstructor(self.z), y)
        s_case = Clause(PatternConstructor(self.s, [PatternVar(a)]), self.s(self.add(a, y)))
        mod[self.add] = Function([x, y], Match(x, [z_case, s_case]))

        self.l = GlobalTypeVar("list")
        a = TypeVar("a")
        self.nil = Constructor("nil", [], self.l)
        self.cons = Constructor("cons", [a, self.l(a)], self.l)
        mod[self.l] = TypeData(self.l, [a], [self.nil, self.cons])

        self.length = GlobalVar("length")
        a = TypeVar("a")
        x = Var("x", self.l(a))
        y = Var("y")
        nil_case = Clause(PatternConstructor(self.nil), self.z())
        cons_case = Clause(PatternConstructor(self.cons, [PatternWildcard(), PatternVar(y)]),
                           self.s(self.length(y)))
        mod[self.length] = Function([x], Match(x, [nil_case, cons_case]), None, [a])

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
        mod[self.map] = Function([f, x], Match(x, [nil_case, cons_case]), None, [a, b])

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
        mod[self.foldl] = Function([f, av, bv], Match(bv, [nil_case, cons_case]), None, [a, b])

        self.tree = GlobalTypeVar("tree")
        a = TypeVar("a")
        self.rose = Constructor("rose", [a, self.l(self.tree(a))], self.tree)
        mod[self.tree] = TypeData(self.tree, [a], [self.rose])

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
        mod[self.foldr] = Function([f, bv, av], Match(av, [nil_case, cons_case]), None, [a, b])

        self.sum = GlobalVar("sum")
        a = Var("a", self.l(self.nat()))
        mod[self.sum] = Function([a], self.foldl(self.add, self.z(), a))

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
        mod[self.tmap] = Function([f, t], Match(t, [rose_case]), self.tree(b), [a, b])

        self.size = GlobalVar("size")
        a = TypeVar("a")
        t = Var("t", self.tree(a))
        x = Var("x", self.tree(a))
        z = Var("z")
        rose_case = Clause(PatternConstructor(self.rose, [PatternWildcard(), PatternVar(z)]),
                           self.s(self.sum(self.map(Function([x], self.size(x)), z))))
        mod[self.size] = Function([t], Match(t, [rose_case]), self.nat(), [a])
        # cannot infer return type here
