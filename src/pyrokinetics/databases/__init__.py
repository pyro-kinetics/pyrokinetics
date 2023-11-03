from platform import python_version_tuple

__all__ = []

if tuple(int(x) for x in python_version_tuple()[:2]) >= (3, 9):
    from .imas import ids_to_pyro, pyro_to_ids

    __all__.extend(["pyro_to_ids", "ids_to_pyro"])
