from __future__ import absolute_import

class Target(object):
    str2type = {'x86': 1, 'cuda': 2, 'rasp': 3}
    type2str = {1: 'x86', 2: 'cuda', 3: 'rasp'}
    def __init__(self, target_type):
        """Constructs a context."""
        default_target = None
        if isinstance(target_type, Target):
            self.target_typeid = target_type.target_typeid
        else:
            self.target_typeid = Target.str2type[target_type]

    @property
    def target_type(self):
        """Returns the target type of current target."""
        return Target.type2str[self.target_typeid]

    def __hash__(self):
        """Compute hash value of target for dictionary lookup"""
        return hash(self.target_typeid)

    def __eq__(self, other):
        """Compares two targets. Two targets are equal if they
        have the same target type.
        """
        return isinstance(other, Target) and \
            self.target_typeid == other.target_typeid

    def __str__(self):
        return '%s' % (self.target_type)

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        self._old_target = Target.default_target
        Target.default_target = self
        return self

    def __exit__(self, ptype, value, trace):
        Target.default_target = self._old_target

Target.default_target = Target('x86')

def x86():
    return Target('x86')

def cuda():
    return Target('cuda')

def rasp():
    return Target('rasp')

def current_target():
    return Target.default_target
