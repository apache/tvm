"""Name manager to make sure name is unique."""
from __future__ import absolute_import as _abs

class NameManager(object):
    """NameManager to do automatic naming.

    User can also inherit this object to change naming behavior.
    """
    current = None

    def __init__(self):
        self._counter = {}
        self._old_manager = None

    def get(self, hint):
        """Get the canonical name for a symbol.

        This is default implementation.
        When user specified a name,
        the user specified name will be used.

        When user did not, we will automatically generate a
        name based on hint string.

        Parameters
        ----------
        hint : str
            A hint string, which can be used to generate name.

        Returns
        -------
        full_name : str
            A canonical name for the user.
        """
        if hint not in self._counter:
            self._counter[hint] = 0
        name = '%s%d' % (hint, self._counter[hint])
        self._counter[hint] += 1
        return name

    def __enter__(self):
        self._old_manager = NameManager.current
        NameManager.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_manager
        NameManager.current = self._old_manager

# initialize the default name manager
NameManager.current = NameManager()
