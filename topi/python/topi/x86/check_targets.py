# pylint: disable=invalid-name,unused-variable,invalid-name,unused-argument
"""Checks different x86 targets for target specific schedules"""

def check_skylake(target):
    """
    Checks if the target is skylake
    """

    for opt in target.options:
        if opt == '-mcpu=skylake-avx512':
            return True
    return False
