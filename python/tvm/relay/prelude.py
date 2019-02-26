# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""Adds certain standard global functions and ADT definitions to the module."""
from .ty import GlobalTypeVar, TypeVar, FuncType, TupleType, scalar_type
from .expr import Var, Function, GlobalVar, Let, If, Tuple, TupleGetItem
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
        self.mod[self.map] = Function([f, x], Match(x, [nil_case, cons_case]), self.l(b), [a, b])

    def define_list_foldl(self):
        """Defines a left-way fold over a list.

        foldl(f, z, l) : fn<a, b>(fn(a, b) -> a, a, list[b]) -> a

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
                                        Match(bv, [nil_case, cons_case]), a, [a, b])

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
                                        Match(av, [nil_case, cons_case]), b, [a, b])

    def define_list_concat(self):
        """Defines a function that concatenates two lists.

        concat(l1, l2) : fn<a>(list[a], list[a]) -> list[a]"""
        self.concat = GlobalVar("concat")
        a = TypeVar("a")
        l1 = Var("l1", self.l(a))
        l2 = Var("l2", self.l(a))
        h = Var("h")
        t = Var("t")
        updater = Function([h, t], self.cons(h, t))
        self.mod[self.concat] = Function([l1, l2],
                                         self.foldr(updater, l2, l1),
                                         self.l(a), [a])

    def define_list_filter(self):
        """Defines a function that filters a list.

        filter(f, l) : fn<a>(fn(a) -> Tensor[(), bool], list[a]) -> list[a]

        It returns the sublist of l consisting of the elements for which f returns true.
        """
        self.filter = GlobalVar("filter")
        a = TypeVar("a")
        f = Var("f", FuncType([a], scalar_type("bool")))
        l = Var("l", self.l(a))
        h = Var("h")
        t = Var("t")
        nil_case = Clause(PatternConstructor(self.nil), self.nil())
        cons_case = Clause(PatternConstructor(self.cons, [PatternVar(h), PatternVar(t)]),
                           If(f(h), self.cons(h, self.filter(f, t)), self.filter(f, t)))
        self.mod[self.filter] = Function([f, l], Match(l, [nil_case, cons_case]), self.l(a), [a])

    def define_list_zip(self):
        """Defines a function that combines two lists into a list of tuples of their elements.

        zip(l, m) : fn<a, b>(list[a], list[b]) -> list[(a, b)]

        The zipped list will be the length of the shorter list.
        """
        self.zip = GlobalVar("zip")
        a = TypeVar("a")
        b = TypeVar("b")
        nil_case = Clause(PatternConstructor(self.nil), self.nil())
        l1 = Var("l1")
        l2 = Var("l2")
        h1 = Var("h1")
        h2 = Var("h2")
        t1 = Var("t1")
        t2 = Var("t2")
        inner_cons_case = Clause(PatternConstructor(self.cons, [PatternVar(h2), PatternVar(t2)]),
                                 self.cons(Tuple([h1, h2]), self.zip(t1, t2)))
        outer_cons_case = Clause(PatternConstructor(self.cons, [PatternVar(h1), PatternVar(t1)]),
                                 Match(l2, [nil_case, inner_cons_case]))
        self.mod[self.zip] = Function([l1, l2], Match(l1, [nil_case, outer_cons_case]),
                                      self.l(TupleType([a, b])), [a, b])

    def define_list_rev(self):
        """Defines a function that reverses a list.

        rev(l) : fn<a>(list[a]) -> list[a]
        """
        self.rev = GlobalVar("rev")
        a = TypeVar("a")
        l = Var("l", self.l(a))
        x = Var("x")
        y = Var("y")
        updater = Function([y, x], self.cons(x, y))
        self.mod[self.rev] = Function([l],
                                      self.foldl(updater, self.nil(), l),
                                      self.l(a), [a])

    def define_list_map_accumr(self):
        """Defines an accumulative map, which is a fold that simulataneously updates
        an accumulator value and a list of results.

        map_accumr(f, s, l) : fn<a, b, c>(fn(a, b) -> (a, c), a, list[b]) -> (a, list[c])

        This map proceeds through l from right to left.
        """
        self.map_accumr = GlobalVar("map_accumr")
        a = TypeVar("a")
        b = TypeVar("b")
        c = TypeVar("c")
        f = Var("f", FuncType([a, b], TupleType([a, c])))
        acc = Var("acc", a)
        l = Var("l", self.l(b))
        v = Var("v", b)
        p = Var("p", TupleType([a, self.l(c)]))
        f_out = Var("f_out", TupleType([a, c]))
        updater = Function([v, p],
                           Let(f_out, f(TupleGetItem(p, 0), v),
                               Tuple([TupleGetItem(f_out, 0),
                                      self.cons(TupleGetItem(f_out, 1),
                                                TupleGetItem(p, 1))])),
                           TupleType([a, self.l(c)]))
        self.mod[self.map_accumr] = Function([f, acc, l],
                                             self.foldr(updater, Tuple([acc, self.nil()]), l),
                                             TupleType([a, self.l(c)]),
                                             [a, b, c])

    def define_list_map_accuml(self):
        """Defines an accumulative map, which is a fold that simulataneously updates
        an accumulator value and a list of results.

        map_accuml(f, s, l) : fn<a, b, c>(fn(a, b) -> (a, c), a, list[b]) -> (a, list[c])

        This map proceeds through l from left to right.
        """
        self.map_accuml = GlobalVar("map_accuml")
        a = TypeVar("a")
        b = TypeVar("b")
        c = TypeVar("c")
        f = Var("f", FuncType([a, b], TupleType([a, c])))
        acc = Var("acc", a)
        l = Var("l", self.l(b))
        v = Var("v", b)
        p = Var("p", TupleType([a, self.l(c)]))
        f_out = Var("f_out", TupleType([a, c]))
        updater = Function([p, v],
                           Let(f_out, f(TupleGetItem(p, 0), v),
                               Tuple([TupleGetItem(f_out, 0),
                                      self.cons(TupleGetItem(f_out, 1),
                                                TupleGetItem(p, 1))])),
                           TupleType([a, self.l(c)]))
        self.mod[self.map_accuml] = Function([f, acc, l],
                                             self.foldl(updater, Tuple([acc, self.nil()]), l),
                                             TupleType([a, self.l(c)]),
                                             [a, b, c])


    def define_optional_adt(self):
        """Defines an optional ADT, which can either contain some other
        type or nothing at all."""
        self.optional = GlobalTypeVar("optional")
        a = TypeVar("a")
        self.some = Constructor("some", [a], self.optional)
        self.none = Constructor("none", [], self.optional)
        self.mod[self.optional] = TypeData(self.optional, [a], [self.some, self.none])

    def define_list_unfoldr(self):
        """Defines a function that builds up a list starting from a seed value.

        unfoldr(f, s) : fn<a, b>(fn(a) -> Optional[(a, b)], a) -> list[b]

        f returns an option containing a new seed and an output value. f will
        continue to be called on the new seeds until it returns None. All the
        output values will be combined into a list, right to left.
        """
        self.unfoldr = GlobalVar("unfoldr")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a], self.optional(TupleType([a, b]))))
        s = Var("s", a)
        p = Var("p", TupleType([a, b]))
        none_case = Clause(PatternConstructor(self.none), self.nil())
        some_case = Clause(PatternConstructor(self.some, [PatternVar(p)]),
                           self.cons(TupleGetItem(p, 1),
                                     self.unfoldr(f, TupleGetItem(p, 0))))
        self.mod[self.unfoldr] = Function([f, s], Match(f(s), [none_case, some_case]),
                                          self.l(b), [a, b])

    def define_list_unfoldl(self):
        """Defines a function that builds up a list starting from a seed value.

        unfoldl(f, s) : fn<a, b>(fn(a) -> Optional[(a, b)], a) -> list[b]

        f returns an option containing a new seed and an output value. f will
        continue to be called on the new seeds until it returns None. All the
        output values will be combined into a list, left to right.
        """
        self.unfoldl = GlobalVar("unfoldl")
        a = TypeVar("a")
        b = TypeVar("b")
        f = Var("f", FuncType([a], self.optional(TupleType([a, b]))))
        s = Var("s", a)
        # easiest way to implement is to do a right unfold and reverse
        self.mod[self.unfoldl] = Function([f, s],
                                          self.rev(self.unfoldr(f, s)),
                                          self.l(b), [a, b])

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
        """Defines a function that computes the size of a tree as a nat.

        Signature: fn<a>(t : tree[a]) -> nat
        """
        self.size = GlobalVar("size")
        a = TypeVar("a")
        t = Var("t", self.tree(a))
        x = Var("x", self.tree(a))
        z = Var("z")
        rose_case = Clause(PatternConstructor(self.rose, [PatternWildcard(), PatternVar(z)]),
                           self.s(self.sum(self.map(Function([x], self.size(x)), z))))
        self.mod[self.size] = Function([t],
                                       Match(t, [rose_case]), self.nat(), [a])

    def define_id(self):
        """Defines a function that return it's argument.

        Signature: fn<a>(x : a) -> a
        """
        self.id = GlobalVar("id")
        a = TypeVar("a")
        x = Var("x", a)
        self.mod[self.id] = Function([x], x, a, [a])


    def define_compose(self):
        """Defines a function that compose two function.

        Signature: fn<a, b, c>(f : fn(b) -> c, g : fn(a) -> b) -> fn(a) -> c
        """
        self.compose = GlobalVar("compose")
        a = TypeVar("a")
        b = TypeVar("b")
        c = TypeVar("c")
        f = Var("f", FuncType([b], c))
        g = Var("g", FuncType([a], b))
        x = Var("x")
        self.mod[self.compose] = Function([f, g],
                                          Function([x], f(g(x))),
                                          FuncType([a], c),
                                          [a, b, c])


    def define_iterate(self):
        """Define a function that take a number n, a function f,
        and return a closure that apply f n time on it's argument.

        Signature: fn<a>(n : nat, f : fn(a) -> a) -> fn(a) -> a
        """
        self.iterate = GlobalVar("iterate")
        a = TypeVar("a")
        f = Var("f", FuncType([a], a))
        x = Var("x", self.nat())
        y = Var("y", self.nat())
        z_case = Clause(PatternConstructor(self.z), self.id)
        s_case = Clause(PatternConstructor(self.s, [PatternVar(y)]),
                        self.compose(f, self.iterate(f, y)))
        self.mod[self.iterate] = Function([f, x],
                                          Match(x, [z_case, s_case]),
                                          FuncType([a], a),
                                          [a])

    def __init__(self, mod):
        self.mod = mod
        self.define_list_adt()
        self.define_list_map()
        self.define_list_foldl()
        self.define_list_foldr()
        self.define_list_concat()
        self.define_list_filter()
        self.define_list_zip()
        self.define_list_rev()
        self.define_list_map_accumr()
        self.define_list_map_accuml()

        self.define_optional_adt()
        self.define_list_unfoldr()
        self.define_list_unfoldl()

        self.define_nat_adt()
        self.define_nat_double()
        self.define_nat_add()
        self.define_list_length()
        self.define_list_sum()

        self.define_tree_adt()
        self.define_tree_map()
        self.define_tree_size()

        self.define_id()
        self.define_compose()
        self.define_iterate()
