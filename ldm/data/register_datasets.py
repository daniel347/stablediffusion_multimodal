__DATASETS__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASETS__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASETS__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, **kwargs):
    if __DATASETS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASETS__[name](**kwargs)