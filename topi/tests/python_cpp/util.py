import topi

def make_vector(elements):
    if elements is None:
        return topi.cpp.create_IntVector()
    elif isinstance(elements, int):
        return topi.cpp.create_IntVector(elements)
    else:
        return topi.cpp.create_IntVector(*elements)