def not_implemented(func):
    def wrapper_not_implemented(*args, **kwargs):
        self = args[0]
        raise NotImplementedError(
            f"{func.__name__} method not yet implemented for GK code {self.__class__.__name__}"
        )

    return wrapper_not_implemented
