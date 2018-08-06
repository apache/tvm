def get_factor(num):
    """Get all factors of a number.

    Parameters
    ----------
    num : int
        Input number.

    Returns
    -------
    out : list of int
        Factors of input number.
    """
    rtv = []
    for i in range(1, num + 1):
        if num % i == 0:
            rtv.append(i)
    return rtv


